"""H98: long-short designs validated on the crashes themselves (the user: "make it short and long
and test it on the crash year to validate").

Every crash is found from the data, not picked: each spell, October 2018 to October 2026, in
which BTC's daily close fell 25% or more below its running high, measured from that high to the
lowest close before BTC made a new high. Each design's equity (the full backtester, the eight
yearly folds chained) is measured over every crash, peak to trough, beside its yearly returns,
so a crash fix that gives up the bull years shows. Written down before running:

  book alone (live now), and the same with
  F1  shorts on a faster trend, 168h/672h (longs on 240h/960h, flat while they disagree); and
      shorts on 120h/480h
  F2  the whole book on 120h/480h; and on 72h/288h
  55/45 (the rotation at 55% beside the book), and the same with
  B1  the rotation's share running the book while its BTC filter is off (round 107)
  B1F B1 with F1's faster shorts

    python -m research.h98_crash_validation
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
from research.holdout2018 import HOLDOUT

BOOK = json.load(open("config/comp.json"))["strategy"]
MIX = json.load(open("config/comp_er_55.json"))["strategy"]
DESIGNS = {
    "book alone (live)": BOOK,
    "F1 book, shorts 168h/672h": dict(BOOK, ls_short_trend=[168, 672]),
    "F1 book, shorts 120h/480h": dict(BOOK, ls_short_trend=[120, 480]),
    "F2 book on 120h/480h": dict(BOOK, ls_trend=[120, 480]),
    "F2 book on 72h/288h": dict(BOOK, ls_trend=[72, 288]),
    "55/45": MIX,
    "B1 55/45, book in bears": dict(MIX, ls_absorb_rotation=1.0),
    "B1F B1 + shorts 168h/672h": dict(MIX, ls_absorb_rotation=1.0, ls_short_trend=[168, 672]),
}
THRESHOLD = 0.25


def crashes(btc):
    """(peak, trough, fall) for every spell BTC fell THRESHOLD or more below its running high."""
    out, peak_t, peak, low_t, low = [], btc.index[0], btc.iloc[0], None, None
    for t, p in btc.items():
        if p >= peak:
            if low is not None and 1 - low / peak >= THRESHOLD:
                out.append((peak_t, low_t, 1 - low / peak))
            peak_t, peak, low_t, low = t, p, None, None
        elif low is None or p < low:
            low_t, low = t, p
    if low is not None and 1 - low / peak >= THRESHOLD:
        out.append((peak_t, low_t, 1 - low / peak))
    return out


def chained(curves):
    parts, level = [], 1.0
    for c in curves:
        parts.append(c * level)
        level = float(c.iloc[-1] * level)
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep="last")]


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    jobs = [(n, st, f) for n, st in DESIGNS.items() for f in folds]
    with ProcessPoolExecutor(max_workers=8) as pool:
        got = {(n, s): c for n, s, c in pool.map(bot_curve, jobs)}
    D = load()
    btc = D.c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    print("Crashes (BTC 25%+ below its high, peak to trough):")
    for a, b, f in events:
        print("  %s to %s  BTC -%.0f%%" % (a.date(), b.date(), f * 100))
    print("\nReturn over each crash, then yearly returns (%s) and six-year total" % " ".join(s[:4] for s, _ in folds))
    for n in DESIGNS:
        eq = chained([got[(n, s)] for s, _ in folds])
        crash = []
        for a, b, _ in events:
            seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
            crash.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
        years = [curve_stats(got[(n, s)])["ret"] for s, _ in folds]
        six = np.prod([1 + r for (s, _), r in zip(folds, years) if s >= FOLDS[0][0]]) - 1
        print("  %-28s crashes %s | mean %+5.1f%% | years %s | six %+8.0f%%" % (
            n, " ".join("%+5.0f%%" % (x * 100) for x in crash), np.nanmean(crash) * 100,
            " ".join("%+6.0f%%" % (r * 100) for r in years), six * 100), flush=True)


if __name__ == "__main__":
    main()
