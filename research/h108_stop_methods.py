"""H108: more stop methods on the live book (the user: "use more stop-loss methods"): percentage
trails on longs (ls_long_stop_pct) and shorts (short_stop_pct, beside the 10-ATR trail), and the
wait before a stopped coin may be re-entered (stop_cooldown_hours, 24 live).
On the book alone on 24/72-hour EMAs (deployed 7 October
2026), backtested after the user deployed it: every BTC crash since 2018 (H98's method), every
year, against the book alone with its usual shorts and against B1 with the same fast shorts.

    python -m research.h108_stop_methods
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
DESIGNS = {"live (shorts 10 ATR, 24h cooldown)": LIVE}
for p in (0.05, 0.10, 0.15):
    DESIGNS["longs trailed %.0f%%" % (p * 100)] = dict(LIVE, ls_long_stop_pct=p)
for p in (0.05, 0.10, 0.15):
    DESIGNS["shorts trailed %.0f%%" % (p * 100)] = dict(LIVE, short_stop_pct=p)
DESIGNS["both trailed 10%"] = dict(LIVE, ls_long_stop_pct=0.10, short_stop_pct=0.10)
for h in (6, 12, 72):
    DESIGNS["cooldown %dh after a stop" % h] = dict(LIVE, stop_cooldown_hours=h)


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    with ProcessPoolExecutor(max_workers=8) as pool:
        got = {(n, s): c for n, s, c in pool.map(bot_curve, [(n, st, f) for n, st in DESIGNS.items() for f in folds])}
    btc = load().c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    print("Return over each of the %d crashes | mean | yearly (%s) | worst DD | six years | mean median 14d"
          % (len(events), " ".join(s[:4] for s, _ in folds)))
    for n in DESIGNS:
        eq = chained([got[(n, s)] for s, _ in folds])
        crash = []
        for a, b, _ in events:
            seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
            crash.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
        stats = [curve_stats(got[(n, s)]) for s, _ in folds]
        six = [x for (s, _), x in zip(folds, stats) if s >= FOLDS[0][0]]
        print("  %-38s %s | %+5.1f%% | %s | %3.0f%% | %+8.0f%% | %5.2f" % (
            n, " ".join("%+5.0f%%" % (x * 100) for x in crash), np.nanmean(crash) * 100,
            " ".join("%+6.0f%%" % (x["ret"] * 100) for x in stats), max(x["mdd"] for x in six) * 100,
            (np.prod([1 + x["ret"] for x in six]) - 1) * 100, np.mean([x["w14"] for x in six])), flush=True)


if __name__ == "__main__":
    main()
