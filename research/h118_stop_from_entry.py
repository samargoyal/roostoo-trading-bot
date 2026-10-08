"""H118: the trailing stops capped at the entry price (the user: "keep the trailing stop but don't go
beyond my initial amount"), option stop_cap_entry_atr: beside the live trailing stops, a short is
also covered once it closes b ATRs above its entry price and a long once b ATRs below, b = 0, 1,
2 and 3; and stops from the entry instead of trailing (stop_from_entry), shorts at 3 and 5 ATRs. H114: the live book as deployed (HAR weights, 10% / 10-ATR short and 8-ATR long trails) with a
6-hour block after a stop (the user's choice), against 12 and 24 hours. H113: the best stops (shorts 10% + 10 ATR, a 24-hour block) with the HAR volatility forecast
or signal-strength weights (full from a 4% gap), for the user's recovery. H111: as H110 with the usual 24-hour block (H110 found the 1-hour block undid the 10% short
stop's gain). H110: the live book with both stops as deployed on 8 October 2026 (shorts also trailed at 10%,
longs at 8 ATRs, a 1-hour block), against the same without them and with the short stop alone.
On the book alone on 24/72-hour EMAs (deployed 7 October
2026), backtested after the user deployed it: every BTC crash since 2018 (H98's method), every
year, against the book alone with its usual shorts and against B1 with the same fast shorts.

    python -m research.h118_stop_from_entry
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
BEFORE = json.load(open("config/comp_book_band_1h.json"))["strategy"]
DESIGNS = {"live (trailing from the best close)": LIVE}
for b in (0.0, 1.0, 2.0, 3.0):
    DESIGNS["trailing + cap at entry + %.0f ATR" % b] = dict(LIVE, stop_cap_entry_atr=b)
for k in (3.0, 5.0):
    DESIGNS["from entry only, shorts %.0f ATR" % k] = dict(LIVE, stop_from_entry=True, short_stop_atr=k, short_stop_pct=0.0)


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
