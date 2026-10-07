"""Round 110: the speed of B1's switch in hours (the user: "use the EMA, but of hours").

B1 (live) moves the account to the book alone while BTC's 168-hour EMA is below its 672-hour
EMA, and back to 55/45 when it recovers; the same EMAs gate the rotation. Written down before
running: the pair at 12/48, 24/96, 48/192, 72/288, 120/480, 168/672 (live) and 240/960 hours,
over every BTC crash since 2018 (H98's method), every year, the worst drawdown, the 14-day
yardstick and the number of switches a year.

    python -m research.round110_hourly_ema
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import curve_stats
from research.h61_swing import load
from research.h95_aggressive_sleeve import bot_curve
from research.h98_crash_validation import chained, crashes
from research.holdout2018 import HOLDOUT

LIVE = json.load(open("config/comp.json"))["strategy"]
PAIRS = [(12, 48), (24, 96), (48, 192), (72, 288), (120, 480), (168, 672), (240, 960)]
DESIGNS = {("B1 55/45, book in bears" if p == (168, 672) else "EMA %dh/%dh" % p):
           dict(LIVE, rotation_trend_fast=p[0], rotation_trend_slow=p[1]) for p in PAIRS}


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    with ProcessPoolExecutor(max_workers=8) as pool:
        got = {(n, s): c for n, s, c in pool.map(bot_curve, [(n, st, f) for n, st in DESIGNS.items() for f in folds])}
    D = load()
    btc_h = D.c["BTC/USD"].dropna()
    btc = btc_h.resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    dev = btc_h[(btc_h.index >= pd.Timestamp(FOLDS[0][0], tz="UTC")) & (btc_h.index < pd.Timestamp(FOLDS[-1][1], tz="UTC"))]
    print("Return over each of the %d crashes | mean | yearly (%s) | worst DD | six years | mean median 14d | switches a year"
          % (len(events), " ".join(s[:4] for s, _ in folds)))
    for (n, st), p in zip(DESIGNS.items(), PAIRS):
        eq = chained([got[(n, s)] for s, _ in folds])
        crash = []
        for a, b, _ in events:
            seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
            crash.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
        stats = [curve_stats(got[(n, s)]) for s, _ in folds]
        six = [x for (s, _), x in zip(folds, stats) if s >= FOLDS[0][0]]
        on = btc_h.ewm(span=p[0], adjust=False).mean() > btc_h.ewm(span=p[1], adjust=False).mean()
        flips = on.loc[dev.index].astype(int).diff().abs().sum() / 6
        print("  %-26s %s | %+5.1f%% | %s | %3.0f%% | %+8.0f%% | %5.2f | %4.0f" % (
            n, " ".join("%+5.0f%%" % (x * 100) for x in crash), np.nanmean(crash) * 100,
            " ".join("%+6.0f%%" % (x["ret"] * 100) for x in stats), max(x["mdd"] for x in six) * 100,
            (np.prod([1 + x["ret"] for x in six]) - 1) * 100, np.mean([x["w14"] for x in six]), flips), flush=True)


if __name__ == "__main__":
    main()
