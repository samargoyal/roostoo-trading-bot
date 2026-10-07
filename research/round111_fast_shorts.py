"""Round 111: a 24/72-hour switch, and very fast EMAs for the book's shorts (the user: "test
24/72, and also try very small EMAs for shorting").

On B1 (live), written down before running: the switch's BTC EMAs at 24/72 hours; the book's
shorts on 12/48, 24/72 and 48/144-hour EMAs (longs on 240/960, flat while they disagree; round
64's option); and the 24/72 switch with 24/72 shorts. Over every BTC crash since 2018, every
year, the worst drawdown, the 14-day yardstick and the switches a year.

    python -m research.round111_fast_shorts
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
    "B1 55/45, book in bears": LIVE,
    "switch 24h/72h": dict(LIVE, rotation_trend_fast=24, rotation_trend_slow=72),
    "B1, shorts on 12h/48h": dict(LIVE, ls_short_trend=[12, 48]),
    "B1, shorts on 24h/72h": dict(LIVE, ls_short_trend=[24, 72]),
    "B1, shorts on 48h/144h": dict(LIVE, ls_short_trend=[48, 144]),
    "switch 24h/72h + shorts 24h/72h": dict(LIVE, rotation_trend_fast=24, rotation_trend_slow=72,
                                            ls_short_trend=[24, 72]),
}
PAIRS = [(st.get("rotation_trend_fast", 168), st.get("rotation_trend_slow", 672)) for st in DESIGNS.values()]


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
