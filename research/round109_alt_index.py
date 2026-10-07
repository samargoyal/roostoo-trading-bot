"""Round 109: an altcoin index for the switch (the user: "can we track the market's bleeding
through crypto indexes?").

B1 (live) moves the account to the book alone while BTC's 168h EMA is below its 672h EMA. The
coins the rotation and the book trade are mostly altcoins, which often fall before and harder
than BTC. Written down before running: the same switch (and the rotation's filter) read from an
equal-weight index of the bot's coins other than BTC and PAXG ("alts"), or off when either
index is off ("either"); against B1 and the fixed 55/45, over every BTC crash since 2018 (H98's
method) and every year.

    python -m research.round109_alt_index
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
DESIGNS = {
    "55/45": json.load(open("config/comp_er_55.json"))["strategy"],
    "B1 55/45, book in bears": LIVE,
    "A-alts: alt index switch": dict(LIVE, regime_index="alts"),
    "A-either: BTC or alt index": dict(LIVE, regime_index="either"),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    with ProcessPoolExecutor(max_workers=8) as pool:
        got = {(n, s): c for n, s, c in pool.map(bot_curve, [(n, st, f) for n, st in DESIGNS.items() for f in folds])}
    btc = load().c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    print("Crashes: %s" % ", ".join("%s to %s (-%.0f%%)" % (a.date(), b.date(), f * 100) for a, b, f in events))
    print("Return over each crash | mean | yearly returns (%s) | worst DD | six years | mean median 14d composite"
          % " ".join(s[:4] for s, _ in folds))
    for n in DESIGNS:
        eq = chained([got[(n, s)] for s, _ in folds])
        crash = []
        for a, b, _ in events:
            seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
            crash.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
        st = [curve_stats(got[(n, s)]) for s, _ in folds]
        six = np.prod([1 + x["ret"] for (s, _), x in zip(folds, st) if s >= FOLDS[0][0]]) - 1
        dev = [x for (s, _), x in zip(folds, st) if s >= FOLDS[0][0]]
        print("  %-28s %s | %+5.1f%% | %s | %3.0f%% | %+8.0f%% | %.2f" % (
            n, " ".join("%+5.0f%%" % (x * 100) for x in crash), np.nanmean(crash) * 100,
            " ".join("%+6.0f%%" % (x["ret"] * 100) for x in st), max(x["mdd"] for x in dev) * 100, six * 100,
            np.mean([x["w14"] for x in dev])), flush=True)


if __name__ == "__main__":
    main()
