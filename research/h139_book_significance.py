"""H139: an EMA trend book, 100% of the account, whose edge is significant (p < 0.05) and which
holds up in crashes (the user's brief, after H137 gave the 24/72-hour book p = 0.25).

Fixed before running. Every design is the long-short book alone (no rotation): every coin long
while its fast EMA is above its slow one and short while below, weighted by inverse volatility
times its 30-day efficiency ratio, shorts with a 10-ATR trailing stop and none where funding shows
shorts crowded (config/comp_book.json), with
  EMA pairs (hours): 72/288, 120/480, 168/672, 240/960, 336/1344
  shorts on the same EMAs, or on 24/72-hour EMAs (out of longs and into shorts within a day)
10 designs, run through the production backtester on the eight yearly folds, October 2018 to
October 2026 (taker fees), chained.

  selection: the highest Sharpe ratio of daily returns in development, October 2020 to October
      2024; nothing else is chosen from data
  significance, on the chosen design: the mean daily return above zero by a moving-block bootstrap
      (7-day blocks, 10,000 resamples), one-sided, in validation (October 2024 to October 2026, not
      used to choose) and in the holdout (October 2018 to October 2020); pass at p < 0.05 in
      validation. The deflated Sharpe ratio over validation with N = 10 (the designs tried here).
  crashes: every spell from October 2018 in which BTC closed 25% or more below its running high,
      peak to trough (research/h98_crash_validation.py); the mean return over them must be positive.
Reported beside the 24/72-hour book that was live from 9 to 12 October 2026 and B1 at 60/40.

    python -m research.h139_book_significance
"""
import json
import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from statistics import NormalDist

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h61_swing import load
from research.h95_aggressive_sleeve import bot_curve
from research.h98_crash_validation import chained, crashes
from research.holdout2018 import HOLDOUT

BOOK = json.load(open("config/comp_book.json"))["strategy"]
PAIRS = ([72, 288], [120, 480], [168, 672], [240, 960], [336, 1344])
DESIGNS = {}
for f, s in PAIRS:
    DESIGNS["H139 book %d/%d" % (f, s)] = dict(BOOK, ls_trend=[f, s])
    DESIGNS["H139 book %d/%d, shorts 24/72" % (f, s)] = dict(BOOK, ls_trend=[f, s], ls_short_trend=[24, 72])
REFERENCE = {"the 24/72-hour book (live 9-12 Oct)": json.load(open("config/comp_book_cap3.json"))["strategy"],
             "B1 60/40 (live now)": json.load(open("config/comp_b1_60.json"))["strategy"]}
DEV = ("2020-10-01", "2024-10-01")
VAL = ("2024-10-01", "2026-10-01")
HOLD = ("2018-10-01", "2020-10-01")
RNG = np.random.default_rng(0)


def daily(eq, a, b):
    d = eq.resample("1D").last()
    d = d[(d.index >= pd.Timestamp(a, tz="UTC")) & (d.index < pd.Timestamp(b, tz="UTC"))]
    return d.pct_change().dropna().values


def p_mean(r, block=7, n=10_000):
    x, k = r - r.mean(), -(-len(r) // block)
    starts = RNG.integers(0, len(r) - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :len(r)]
    return (np.sum(x[idx].mean(axis=1) >= r.mean()) + 1) / (n + 1)


def sharpe(r):
    return r.mean() / r.std(ddof=1) * math.sqrt(365) if len(r) > 1 and r.std() > 0 else 0.0


def deflated(r, n_trials, var_sr):
    z, g = NormalDist().inv_cdf, 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - g) * z(1 - 1 / n_trials) + g * z(1 - 1 / (n_trials * math.e)))
    sr = r.mean() / r.std(ddof=1)
    skew = ((r - r.mean()) ** 3).mean() / r.std() ** 3
    kurt = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    return NormalDist().cdf((sr - sr0) * math.sqrt(len(r) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    every = dict(DESIGNS, **REFERENCE)
    jobs = [(n, st, f) for n, st in every.items() for f in folds]
    with ProcessPoolExecutor(max_workers=12) as pool:
        got = {(n, s): c for n, s, c in pool.map(bot_curve, jobs)}
    eq = {n: chained([got[(n, s)] for s, _ in folds]) for n in every}
    D = load()
    btc = D.c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)

    def crash_returns(e):
        out = []
        for a, b, _ in events:
            seg = e[(e.index >= a) & (e.index <= b + pd.Timedelta(days=1))]
            out.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
        return out

    val_sr = np.array([sharpe(daily(eq[n], *VAL)) / math.sqrt(365) for n in DESIGNS])
    var_sr = val_sr.var(ddof=1)
    print("H139: %d crashes since October 2018: %s" % (len(events), ", ".join(
        "%s -%.0f%%" % (a.strftime("%Y-%m"), 100 * f) for a, _, f in events)))
    print("\n%-38s %8s | %8s %7s %8s | %8s %7s | %8s %s" % ("design", "dev SR", "val SR", "val p", "val ret",
                                                        "hold SR", "hold p", "crash", "per crash"))
    rows = {}
    for n in every:
        dv, vl, ho = daily(eq[n], *DEV), daily(eq[n], *VAL), daily(eq[n], *HOLD)
        cr = crash_returns(eq[n])
        rows[n] = (sharpe(dv), sharpe(vl), p_mean(vl), np.prod(1 + vl) - 1, sharpe(ho), p_mean(ho), np.nanmean(cr), cr)
        r = rows[n]
        print("%-38s %8.2f | %8.2f %7.4f %+7.0f%% | %8.2f %7.4f | %+7.1f%% %s" % (
            n, r[0], r[1], r[2], 100 * r[3], r[4], r[5], 100 * r[6], " ".join("%+.0f%%" % (100 * x) for x in cr)))
    pick = max(DESIGNS, key=lambda n: rows[n][0])
    r = rows[pick]
    vl = daily(eq[pick], *VAL)
    print("\nChosen on development Sharpe: %s" % pick)
    print("  validation: Sharpe %.2f, return %+.0f%%, p(mean > 0) = %.4f -> %s at 0.05" % (
        r[1], 100 * r[3], r[2], "passes" if r[2] < 0.05 else "fails"))
    print("  deflated Sharpe over validation, N = %d: %.2f" % (len(DESIGNS), deflated(vl, len(DESIGNS), var_sr)))
    print("  holdout: Sharpe %.2f, p = %.4f; crashes: mean %+.1f%% (%s)" % (
        r[4], r[5], 100 * r[6], "positive" if r[6] > 0 else "negative"))
    print("  config:", json.dumps(DESIGNS[pick], sort_keys=True))


if __name__ == "__main__":
    main()
