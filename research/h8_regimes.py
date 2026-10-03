"""H8 and H9: market timing with time-series models.

H8, regimes: a 2-state Gaussian hidden Markov model on BTC's 4-hour log returns, refitted
monthly on all earlier data (five random starts, best likelihood kept). Each 4 hours the
forward algorithm gives the filtered probability of the calmer, higher-mean state from
past data only (no smoothing, which would peek ahead). Compared with the bot's rule,
BTC above its 200-hour EMA, by the market's average return in the following 24h / 72h
when each switch says risk-on versus risk-off. Saves the regime series for research.sim.

H9, return prediction: do past returns predict future returns over time? Pooled AR(7) on
daily returns of the universe coins, and ARIMA(1,0,1) on BTC daily returns, both refitted
monthly. Scored by out-of-sample R-squared and how often the sign is right.

    python -m research.h8_regimes
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.arima.model import ARIMA

from bot.config import UniverseConfig
from research.panel import cached_pairs, load_panel, monthly_universe

OUT = os.path.join("runs", "research", "regimes.pkl")
FIRST_MONTH, LAST_MONTH = "2024-10-01", "2026-09-01"
PERIODS = {"Y0 Oct24-Sep25": ("2024-10-01", "2025-10-01"),
           "Y1 Oct25-Jun26": ("2025-10-01", "2026-07-01"),
           "Jul-Sep26 (seen)": ("2026-07-01", "2026-10-01")}


def fit_hmm(x: np.ndarray) -> GaussianHMM:
    best, best_score = None, -np.inf
    for seed in range(5):
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=300, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x)
            score = model.score(x)
        if score > best_score:
            best, best_score = model, score
    return best


def filtered_probability(model: GaussianHMM, x: np.ndarray, state: int, prior: np.ndarray):
    """Forward-algorithm probability of `state` after each observation, starting from `prior`."""
    means = model.means_[:, 0]
    sds = np.sqrt(model.covars_[:, 0, 0])
    trans = model.transmat_
    p = prior.copy()
    out = np.empty(len(x))
    for t, obs in enumerate(x[:, 0]):
        p = p @ trans
        like = np.exp(-0.5 * ((obs - means) / sds) ** 2) / sds
        p = p * like
        p = p / p.sum()
        out[t] = p[state]
    return out, p


def hmm_regime(btc_close: pd.Series) -> pd.Series:
    """Hourly boolean risk-on series from the walk-forward HMM (updated at 4-hour closes)."""
    bars = np.log(btc_close.resample("4h", label="right", closed="left").last()).diff().dropna()
    x_all = (bars * 100).to_frame().values
    months = list(pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS", tz="UTC"))
    probs = pd.Series(np.nan, index=bars.index)
    for month, next_month in zip(months, months[1:] + [bars.index[-1] + pd.Timedelta(hours=4)]):
        train = x_all[bars.index <= month]
        model = fit_hmm(train)
        means, sds = model.means_[:, 0], np.sqrt(model.covars_[:, 0, 0])
        calm = int(np.argmax(means / sds))     # the state with the better return per unit risk
        _, prior = filtered_probability(model, train, calm, model.startprob_)
        test = (bars.index > month) & (bars.index <= next_month)
        p, _ = filtered_probability(model, x_all[test], calm, prior)
        probs[test] = p
    # A 4-hour bar labelled T closes at T; it is usable from the hourly bar that opens at T.
    hourly = probs.reindex(btc_close.index, method="ffill")
    return hourly > 0.5


def timing_table(name: str, regime: pd.Series, mean_fwd: dict) -> None:
    cells = []
    for start, end in PERIODS.values():
        sel = (regime.index >= start) & (regime.index < end) & (regime.index.hour == 0)
        on = regime[sel].astype(bool)
        parts = []
        for h, fwd in mean_fwd.items():
            f = fwd[sel]
            parts.append("%+.2f/%+.2f%%" % (f[on].mean() * 100, f[~on].mean() * 100))
        cells.append("%3.0f%% on  %s" % (on.mean() * 100, "  ".join(parts)))
    print("%-14s" % name + "".join("%34s" % c for c in cells))


def ar_tests(panel, mask) -> None:
    logp = np.log(panel["close"])
    daily = logp[logp.index.hour == 0]
    r = daily.diff()
    fwd = r.shift(-1)
    lags = pd.concat({k: r.shift(k - 1) for k in range(1, 8)}, axis=1)
    rows = []
    for pair in r.columns:
        m = mask.loc[r.index, pair]
        df = pd.concat([lags.xs(pair, axis=1, level=1), fwd[pair].rename("y")], axis=1)[m.values]
        df["pair"] = pair
        rows.append(df)
    data = pd.concat(rows).dropna()
    months = list(pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS", tz="UTC"))
    preds = []
    for month, nxt in zip(months, months[1:] + [data.index.max() + pd.Timedelta(days=1)]):
        train = data[data.index < month - pd.Timedelta(days=1)]
        test = data[(data.index >= month) & (data.index < nxt)]
        X = np.column_stack([np.ones(len(train)), train[list(range(1, 8))].values])
        beta, *_ = np.linalg.lstsq(X, train["y"].values, rcond=None)
        p = np.column_stack([np.ones(len(test)), test[list(range(1, 8))].values]) @ beta
        preds.append(pd.DataFrame({"pred": p, "y": test["y"].values}, index=test.index))
    pooled = pd.concat(preds)

    btc = r["BTC/USD"].dropna()
    arima_preds = []
    for month, nxt in zip(months, months[1:] + [btc.index[-1] + pd.Timedelta(days=1)]):
        train = btc[btc.index < month]
        test = btc[(btc.index >= month) & (btc.index < nxt)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(train.values, order=(1, 0, 1)).fit()
            # One-day-ahead forecasts through the month, updating with each new observation.
            res_ext = res.extend(test.values) if len(test) else None
        if res_ext is None:
            continue
        f = np.concatenate([[res.forecast(1)[0]], res_ext.predict(start=0, end=len(test) - 1)[1:]])
        arima_preds.append(pd.DataFrame({"pred": f[:len(test)], "y": test.values}, index=test.index))
    arima = pd.concat(arima_preds)

    print("\nH9: predicting the next day's return from past returns (out of sample)")
    print("%-26s" % "model" + "".join("%26s" % p for p in PERIODS))
    for name, df in (("pooled AR(7), all coins", pooled), ("ARIMA(1,0,1), BTC", arima)):
        cells = []
        for start, end in PERIODS.values():
            x = df[(df.index >= start) & (df.index < end)]
            r2 = 1 - np.sum((x.y - x.pred) ** 2) / np.sum(x.y ** 2)  # against a zero forecast
            hit = np.mean(np.sign(x.pred) == np.sign(x.y))
            cells.append("R2 %+.3f  sign %.1f%%" % (r2, hit * 100))
        print("%-26s" % name + "".join("%26s" % c for c in cells))


def main() -> None:
    pairs = cached_pairs()
    panel = load_panel(pairs)
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    close = panel["close"]
    btc = close["BTC/USD"]
    ema200 = btc.ewm(span=200, adjust=False).mean()
    ema_rule = btc > ema200
    hmm = hmm_regime(btc)
    with open(OUT, "wb") as f:
        pickle.dump({"hmm": hmm, "ema200": ema_rule}, f)

    logp = np.log(close)
    mean_fwd = {h: (logp.shift(-h) - logp).where(mask).mean(axis=1) for h in (24, 72)}
    print("H8: average next 24h / 72h return of the universe when risk-on / risk-off")
    print("%-14s" % "switch" + "".join("%34s" % p for p in PERIODS))
    timing_table("BTC > EMA200", ema_rule, mean_fwd)
    timing_table("HMM 2-state", hmm, mean_fwd)
    timing_table("both agree", ema_rule & hmm, mean_fwd)
    flips = lambda s: int((s.astype(int).diff().abs() > 0).loc["2024-10-01":].sum())
    print("switches since Oct 2024: EMA200 %d, HMM %d" % (flips(ema_rule), flips(hmm)))
    ar_tests(panel, mask)


if __name__ == "__main__":
    main()
