"""H22: Bayesian and implied-volatility market timing, against the rotation sleeve's EMA filter.

The rotation sleeve (most of the bot's return) is only in the market while BTC's 168-hour EMA
is above its 672-hour EMA. Two other ways to read the regime, tested on BTC (the sleeve's
switch) and on every coin in the monthly universe (each coin timed by its own filter):

  BOCPD    Bayesian online changepoint detection (Adams and MacKay, 2007) on BTC's daily log
           returns: a normal model with unknown mean and variance (normal-inverse-gamma prior,
           mean 0, prior daily volatility 3.5%), restarting at a changepoint with probability
           1/30 a day. In the market while the posterior mean of the current regime's drift
           is positive.
  DVOL     Deribit's 30-day at-the-money implied volatility of BTC (the short end of the
           volatility surface), from March 2021. The variance risk premium is implied variance
           minus the last 30 days' realised variance; a high premium has been read both as
           fear that pays later and as a warning.

Written down before running:
  Timing test   hold the coin while its filter is on, cash while off, switching at 00:00 UTC
                with 0.1% per switch; for all coins, an equal 1/N share each of the month's
                universe (top 45 crypto, PAXG out), cash for those whose filter is off. The
                competition's composite score per fold. BOCPD replaces the EMA filter in a
                strategy test only if it has a higher composite in at least 4 of 6 folds and
                a higher median, on BTC for the sleeve's switch or on all coins for a per-coin
                filter.
  DVOL test     the Spearman correlation between the premium at 00:00 and BTC's next 7-day
                return in each fold from 2021 (5 folds); worth a strategy test only with the
                same sign in all 5 and |correlation| >= 0.05 on average.

    python -m research.h22_regimes
"""
import math
import os

import numpy as np
import pandas as pd
import requests
from scipy.special import gammaln

from bot.config import UniverseConfig, load_config
from bot.metrics import summarize
from research.folds import FOLDS, candidates
from research.panel import DEFENSIVE, load_panel, monthly_universe

DVOL_CACHE = os.path.join("data", "deribit_dvol_btc.csv")
FEE = 0.001


def bocpd_drift(x: np.ndarray, hazard: float = 1 / 30, mu0: float = 0.0, kappa0: float = 1.0,
                alpha0: float = 2.0, beta0: float = 0.035 ** 2, max_run: int = 1500) -> np.ndarray:
    """Posterior mean of the current regime's mean after each observation."""
    mu, kappa = np.array([mu0]), np.array([kappa0])
    alpha, beta = np.array([alpha0]), np.array([beta0])
    run = np.array([1.0])
    out = np.zeros(len(x))
    for t, xt in enumerate(x):
        # Student-t predictive of xt under each run length
        nu = 2 * alpha
        scale2 = beta * (kappa + 1) / (alpha * kappa)
        z = (xt - mu) ** 2 / (nu * scale2)
        logpdf = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
                  - 0.5 * np.log(nu * math.pi * scale2) - (nu + 1) / 2 * np.log1p(z))
        pred = np.exp(logpdf - logpdf.max())
        growth = run * pred * (1 - hazard)
        change = (run * pred * hazard).sum()
        run = np.concatenate([[change], growth])
        run /= run.sum()
        # posterior parameters: a fresh prior for run length 0, every other run updated by xt
        mu_new = (kappa * mu + xt) / (kappa + 1)
        beta_new = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[(kappa0 * mu0 + xt) / (kappa0 + 1)], mu_new])
        beta = np.concatenate([[beta0 + kappa0 * (xt - mu0) ** 2 / (2 * (kappa0 + 1))], beta_new])
        kappa = np.concatenate([[kappa0 + 1], kappa + 1])
        alpha = np.concatenate([[alpha0 + 0.5], alpha + 0.5])
        if len(run) > max_run:
            run, mu, kappa, alpha, beta = (a[:max_run] for a in (run, mu, kappa, alpha, beta))
            run /= run.sum()
        out[t] = float((run * mu).sum())
    return out


def timing_curve(daily_ret: pd.Series, on: pd.Series, start: str, end: str):
    """Hold BTC on days after the filter said on at 00:00; pay FEE on each switch."""
    days = daily_ret.loc[start:end].index
    days = days[days < pd.Timestamp(end, tz="UTC")]
    equity, held, curve = 100000.0, 0.0, []
    for d in days:
        want = 1.0 if on.get(d - pd.Timedelta(days=1), False) else 0.0
        if want != held:
            equity *= 1 - FEE
            held = want
        equity *= 1 + held * daily_ret[d]
        curve.append((int(d.value // 10 ** 6) + 86_400_000, equity))
    return curve


def basket_curve(rets: pd.DataFrame, on: pd.DataFrame, member: pd.DataFrame, start: str, end: str):
    """An equal 1/N share of each universe coin, held while its own filter (as of the previous
    00:00) is on, cash otherwise; FEE on every change of weight."""
    days = rets.loc[start:end].index
    days = days[days < pd.Timestamp(end, tz="UTC")]
    equity, held, curve = 100000.0, pd.Series(0.0, index=rets.columns), []
    for d in days:
        prev = d - pd.Timedelta(days=1)
        if prev not in member.index:
            continue
        inside = member.loc[prev] & rets.loc[d].notna()
        n = int(inside.sum())
        want = (on.loc[prev].reindex(rets.columns).fillna(False) & inside).astype(float) / max(n, 1)
        equity *= 1 - FEE * float((want - held).abs().sum())
        held = want
        equity *= 1 + float((held * rets.loc[d].fillna(0.0)).sum())
        curve.append((int(d.value // 10 ** 6) + 86_400_000, equity))
    return curve


def dvol() -> pd.Series:
    if not os.path.exists(DVOL_CACHE):
        rows, start = [], 1616544000000
        end = int(pd.Timestamp("2026-10-01", tz="UTC").value // 10 ** 6)
        while start < end:
            r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                             params={"currency": "BTC", "start_timestamp": start,
                                     "end_timestamp": min(start + 900 * 86_400_000, end), "resolution": "1D"},
                             timeout=30).json()["result"]["data"]
            if not r:
                break
            rows += r
            start = int(r[-1][0]) + 86_400_000
        pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts").to_csv(DVOL_CACHE, index=False)
    df = pd.read_csv(DVOL_CACHE)
    return pd.Series(df["close"].values, index=pd.to_datetime(df["ts"], unit="ms", utc=True))


def main() -> None:
    close = load_panel(["BTC/USD"])["close"]["BTC/USD"].dropna()
    # Daily closes at 00:00 UTC: the close of the 23:00 bar, labelled with the next day.
    daily = close[close.index.hour == 23]
    daily.index = daily.index + pd.Timedelta(hours=1)
    ret = np.log(daily).diff().dropna()
    drift = pd.Series(bocpd_drift(ret.values), index=ret.index)
    bocpd_on = drift > 0
    ema_f = close.ewm(span=168, adjust=False).mean()
    ema_s = close.ewm(span=672, adjust=False).mean()
    ema_on = (ema_f > ema_s)[close.index.hour == 23]
    ema_on.index = ema_on.index + pd.Timedelta(hours=1)
    simple = daily.pct_change().dropna()

    rows = []
    for name, on in [("EMA 168/672 (current)", ema_on), ("BOCPD drift > 0", bocpd_on), ("always in", None)]:
        row = {"filter": name}
        comps = []
        for start, end in FOLDS:
            flag = on if on is not None else pd.Series(True, index=simple.index)
            st = summarize(timing_curve(simple, flag, start, end), 100000.0)
            row[start[:4]] = "%+.0f%% (%.0f%%) %.2f" % (st["total_return"] * 100, st["max_drawdown"] * 100, st["composite"])
            comps.append(st["composite"])
        row["median"] = round(float(np.median(comps)), 2)
        row["_comps"] = comps
        rows.append(row)
    base = rows[0]["_comps"]
    for row in rows:
        row["better than EMA"] = "%d/6" % sum(a > b for a, b in zip(row.pop("_comps"), base))
    pd.set_option("display.width", 250)
    print("Hold BTC while the filter is on: return (max drawdown) composite per fold")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Share of days on: EMA %.0f%%, BOCPD %.0f%%; switches per year: EMA %.1f, BOCPD %.1f" % (
        ema_on.mean() * 100, bocpd_on.mean() * 100,
        ema_on.astype(int).diff().abs().sum() / (len(ema_on) / 365),
        bocpd_on.astype(int).diff().abs().sum() / (len(bocpd_on) / 365)))

    # Every coin in the monthly universe, each timed by its own filter.
    cfg = load_config()
    pairs = sorted(p for p in candidates(cfg) if p != DEFENSIVE)
    panel = load_panel(pairs)
    hourly = panel["close"][pairs]
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    closes = hourly[hourly.index.hour == 23]
    closes.index = closes.index + pd.Timedelta(hours=1)
    member = mask[mask.index.hour == 23]
    member.index = member.index + pd.Timedelta(hours=1)
    logret = np.log(closes).diff()
    simple_all = closes.pct_change()
    ef = hourly.ewm(span=168, adjust=False).mean()
    es = hourly.ewm(span=672, adjust=False).mean()
    ema_all = (ef > es).where(hourly.notna(), False)[hourly.index.hour == 23]
    ema_all.index = ema_all.index + pd.Timedelta(hours=1)
    bocpd_all = pd.DataFrame(False, index=logret.index, columns=pairs)
    for pair in pairs:
        x = logret[pair].dropna()
        if len(x) > 30:
            bocpd_all.loc[x.index, pair] = bocpd_drift(x.values) > 0
    always = pd.DataFrame(True, index=logret.index, columns=pairs)
    rows = []
    for name, on in [("EMA 168/672 per coin", ema_all), ("BOCPD drift > 0 per coin", bocpd_all),
                     ("always in (equal weight)", always)]:
        row = {"filter, all coins": name}
        comps = []
        for start, end in FOLDS:
            st = summarize(basket_curve(simple_all, on, member, start, end), 100000.0)
            row[start[:4]] = "%+.0f%% (%.0f%%) %.2f" % (st["total_return"] * 100, st["max_drawdown"] * 100, st["composite"])
            comps.append(st["composite"])
        row["median"] = round(float(np.median(comps)), 2)
        row["_comps"] = comps
        rows.append(row)
    base = rows[0]["_comps"]
    for row in rows:
        row["better than EMA"] = "%d/6" % sum(a > b for a, b in zip(row.pop("_comps"), base))
    print("\nEvery coin in the universe, each timed by its own filter (1/N each, cash when off)")
    print(pd.DataFrame(rows).to_string(index=False))

    iv = dvol() / 100.0
    rv = ret.rolling(30).var() * 365
    vrp = (iv ** 2 - rv).dropna()
    fwd7 = np.log(daily).shift(-7) - np.log(daily)
    print("\nDVOL variance risk premium vs BTC's next 7 days (Spearman, per fold):")
    corrs = []
    for start, end in FOLDS[1:]:
        both = pd.concat([vrp, fwd7], axis=1).dropna().loc[start:end]
        both = both[both.index < pd.Timestamp(end, tz="UTC")]
        c = both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")
        corrs.append(c)
        print("  %s  %+.3f  (%d days)" % (start[:4], c, len(both)))
    same = max(sum(c > 0 for c in corrs), sum(c < 0 for c in corrs))
    print("  same sign in %d/5, mean %+.3f -> %s" % (same, float(np.mean(corrs)),
                                                    "worth a strategy test" if same == 5 and abs(np.mean(corrs)) >= 0.05 else "no"))


if __name__ == "__main__":
    main()
