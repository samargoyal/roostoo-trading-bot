"""H20: microstructure, volume-surface, Kalman and Bayesian signals, screened by IC on the six
folds before any of them may enter a strategy test.

Written down before running:
  Universe  the bot's rule: the top 45 crypto pairs by 30-day USD volume each month (frozen
            candidates, spread <= 0.1%), PAXG left out, since these are cross-sectional.
  Targets   the next 24-hour and next 168-hour log returns, from 00:00 UTC each day.
  Score     the Spearman rank IC each day, averaged within each fold (October to October).
  Pass      the same sign in at least 5 of 6 folds and a mean |IC| of at least 0.02, at
            either horizon; and, with the defensive book's low-volatility ranking partialled
            out each day, the same sign in at least 4 of 6 folds. Only passing signals go on
            to strategy tests.
  Signals   taker_flow_24 / _168   share of quote volume bought by takers, minus 0.5
                                   (order-flow imbalance at slow horizons)
            trade_size             log mean quote per trade, last 24 hours minus the 30 days
                                   before (larger trades: bigger participants)
            amihud_168             mean |hourly log return| / quote volume (illiquidity)
            cs_spread_168          Corwin-Schultz bid-ask spread from hourly highs and lows
            abnormal_volume        last 24 hours' quote volume over what the coin's
                                   hour-of-week volume surface (previous 8 weeks) expects
            kalman_slope           local-linear-trend Kalman filter on log price: slope over
                                   its standard deviation (level noise r/24^2, slope noise
                                   r/336^4, r the trailing hourly return variance)
            bayes_momentum         336-hour return shrunk towards the cross-sectional mean by
                                   empirical Bayes, less for coins whose return is less noisy
            low_vol_168, mom_336   for reference: the defensive book's and the rotation's
                                   rankings

    python -m research.h20_screen
"""
import numpy as np
import pandas as pd

from bot.config import UniverseConfig, load_config
from research import fullbars
from research.folds import FOLDS, candidates
from research.panel import DEFENSIVE, monthly_universe

HORIZONS = (24, 168)


def rank_ic(feature: pd.DataFrame, target: pd.DataFrame, control: pd.DataFrame = None) -> pd.Series:
    """Daily Spearman IC across the universe (cells outside it are NaN); with `control`, the IC
    of the feature's rank after regressing out the control's rank."""
    fr, tr = feature.rank(axis=1), target.rank(axis=1)
    valid = fr.notna() & tr.notna()
    if control is not None:
        cr = control.rank(axis=1)
        valid &= cr.notna()
        cr = cr.where(valid)
    fr, tr = fr.where(valid), tr.where(valid)
    fd = fr.sub(fr.mean(axis=1), axis=0)
    td = tr.sub(tr.mean(axis=1), axis=0)
    if control is not None:
        cd = cr.sub(cr.mean(axis=1), axis=0)
        beta = (fd * cd).sum(axis=1) / (cd ** 2).sum(axis=1)
        fd = fd - cd.mul(beta, axis=0)
    ic = (fd * td).sum(axis=1) / np.sqrt((fd ** 2).sum(axis=1) * (td ** 2).sum(axis=1))
    return ic[valid.sum(axis=1) >= 10]


def in_fold(series: pd.Series, start: str, end: str) -> pd.Series:
    t = series.index
    return series[(t >= pd.Timestamp(start, tz="UTC")) & (t < pd.Timestamp(end, tz="UTC"))]


def corwin_schultz(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Spread estimate from each pair of consecutive bars (Corwin and Schultz, 2012)."""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def kalman_slope(logp: pd.DataFrame, level_hours: float = 24.0, slope_hours: float = 336.0) -> pd.DataFrame:
    """Local linear trend filter run forward over every column; returns slope / sd(slope)."""
    y = logp.values
    r_all = logp.diff().rolling(720, min_periods=200).var().values
    n, m = y.shape
    level, slope = np.full(m, np.nan), np.zeros(m)
    p11, p12, p22 = np.ones(m), np.zeros(m), np.ones(m)
    out = np.full((n, m), np.nan)
    for t in range(n):
        obs, r = y[t], r_all[t]
        ok = np.isfinite(obs) & np.isfinite(r) & (r > 0)
        start = ok & ~np.isfinite(level)
        level[start], slope[start] = obs[start], 0.0
        p11[start], p12[start], p22[start] = r[start], 0.0, r[start] / slope_hours ** 2
        run = ok & ~start
        if run.any():
            ql, qs = r[run] / level_hours ** 2, r[run] / slope_hours ** 4
            # predict: x = F x, P = F P F' + Q with F = [[1, 1], [0, 1]]
            lv, sl = level[run] + slope[run], slope[run]
            a11 = p11[run] + 2 * p12[run] + p22[run] + ql
            a12 = p12[run] + p22[run]
            a22 = p22[run] + qs
            # update with the observed log price
            s = a11 + r[run]
            k1, k2 = a11 / s, a12 / s
            innov = obs[run] - lv
            level[run], slope[run] = lv + k1 * innov, sl + k2 * innov
            p11[run], p12[run], p22[run] = (1 - k1) * a11, (1 - k1) * a12, a22 - k2 * a12
        out[t, run] = slope[run] / np.sqrt(p22[run])
    return pd.DataFrame(out, index=logp.index, columns=logp.columns)


def bayes_momentum(ret336: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes posterior mean of each coin's 336-hour return, row by row: a normal prior
    across the coins (mean and spread estimated from them), each coin's own noise its variance."""
    m = ret336.mean(axis=1)
    tau2 = (ret336.var(axis=1) - noise.mean(axis=1)).clip(lower=1e-6)
    w = 1.0 / (1.0 + noise.div(tau2, axis=0))  # tau2 / (tau2 + noise)
    return ret336.mul(w).add((1 - w).mul(m, axis=0))


def build(full: dict, mask: pd.DataFrame) -> dict:
    close, high, low = full["close"], full["high"], full["low"]
    qv, trades, tbq = full["quote_volume"], full["trades"], full["taker_buy_quote"]
    logp = np.log(close)
    ret = logp.diff()
    sig = {}
    for k in (24, 168):
        sig["taker_flow_%d" % k] = tbq.rolling(k).sum() / qv.rolling(k).sum() - 0.5
    size_now = qv.rolling(24).sum() / trades.rolling(24).sum()
    size_before = qv.shift(24).rolling(720).sum() / trades.shift(24).rolling(720).sum()
    sig["trade_size"] = np.log(size_now / size_before)
    sig["amihud_168"] = (ret.abs() / qv.where(qv > 0)).rolling(168, min_periods=120).mean()
    sig["cs_spread_168"] = corwin_schultz(high, low).rolling(168, min_periods=120).mean()
    expected = sum(qv.shift(168 * k) for k in range(1, 9)) / 8.0
    sig["abnormal_volume"] = np.log(qv.rolling(24).sum() / expected.rolling(24).sum())
    sig["kalman_slope"] = kalman_slope(logp)
    sig["low_vol_168"] = -ret.rolling(168, min_periods=150).std()
    sig["mom_336"] = logp - logp.shift(336)
    days = close.index[close.index.hour == 0]
    r336 = sig["mom_336"].loc[days].where(mask.loc[days])
    noise = (ret.rolling(336, min_periods=300).var() * 336).loc[days].where(mask.loc[days])
    sig["bayes_momentum"] = bayes_momentum(r336, noise)
    return {k: v.loc[days].where(mask.loc[days]) for k, v in sig.items()}


def main() -> None:
    cfg = load_config()
    pairs = sorted(p for p in candidates(cfg))
    full = fullbars.load(pairs)
    pairs = list(full["close"].columns)
    mask = monthly_universe(full, UniverseConfig(), pairs)
    mask[DEFENSIVE] = False
    signals = build(full, mask)
    days = signals["mom_336"].index
    logp = np.log(full["close"])
    fwd = {h: (logp.shift(-h) - logp).loc[days].where(mask.loc[days]) for h in HORIZONS}
    years = [f[0][:4] for f in FOLDS]

    rows = []
    for name, f in signals.items():
        row = {"signal": name}
        passes = False
        for h in HORIZONS:
            ic = rank_ic(f, fwd[h])
            partial = rank_ic(f, fwd[h], signals["low_vol_168"])
            means = [in_fold(ic, s, e).mean() for s, e in FOLDS]
            pmeans = [in_fold(partial, s, e).mean() for s, e in FOLDS]
            overall = float(np.nanmean(means))
            sign = np.sign(overall)
            same = sum(np.sign(x) == sign for x in means)
            psame = sum(np.sign(x) == sign for x in pmeans)
            row["%dh IC by fold" % h] = " ".join("%+.3f" % x for x in means)
            row["%dh mean" % h] = round(overall, 3)
            row["%dh same sign" % h] = "%d/6" % same
            row["%dh partial same" % h] = "%d/6" % psame
            if same >= 5 and abs(overall) >= 0.02 and psame >= 4 and name not in ("low_vol_168", "mom_336"):
                passes = True
        row["passes"] = "YES" if passes else ""
        rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 20)
    print("Daily rank IC with the next 24h and 168h returns, per fold (folds start in October %s)"
          % ", ".join(years))
    print(table.to_string(index=False))
    table.to_csv("runs/research/h20_screen.csv", index=False)


if __name__ == "__main__":
    main()
