"""H99: how much of B1's return comes from the rotation and how much from the book (the user:
"tell me how much return is produced by the book and the momentum").

B1 (the live settings since 7 October 2026) holds the rotation at 55% and the efficiency book at
45% while BTC's 168h EMA is above its 672h EMA, and the book alone while it is below. Each
sleeve's daily return comes from the full backtester run on that sleeve alone (the rotation at
99%, the book at 100%); each day's contribution is its share that day times its return, summed
by fold. The sum of both approximates B1's own return (shown beside it); the difference is the
daily rebalancing and fees between sleeves that the approximation leaves out.

    python -m research.h99_attribution
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from research.folds import FOLDS
from research.h60_swing_mft import curve_stats
from research.h61_swing import load
from research.h95_aggressive_sleeve import bot_curve
from research.holdout2018 import HOLDOUT

LIVE = json.load(open("config/comp.json"))["strategy"]
TAGS = {"rotation alone": dict(LIVE, rotation_weight=0.99, ls_absorb_rotation=0.0),
        "book alone (live)": dict(LIVE, rotation_weight=0.0, ls_absorb_rotation=0.0),
        "B1 55-45, book in bears": LIVE}


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    with ProcessPoolExecutor(max_workers=8) as pool:
        got = {(t, s): c for t, s, c in pool.map(bot_curve, [(t, st, f) for t, st in TAGS.items() for f in folds])}
    D = load()
    btc = D.c["BTC/USD"]
    on = (btc.ewm(span=168, adjust=False).mean() > btc.ewm(span=672, adjust=False).mean()).resample("1D").last()
    print("By fold: B1's return | rotation's contribution + book's contribution (summed daily) | days in bear mode")
    tot_r = tot_b = 0.0
    for s, _ in folds:
        rot = got[("rotation alone", s)].resample("1D").last().pct_change().dropna()
        book = got[("book alone (live)", s)].resample("1D").last().pct_change().dropna()
        idx = rot.index.intersection(book.index)
        rot, book = rot.loc[idx], book.loc[idx]
        bull = on.shift(1).reindex(idx).fillna(True).astype(bool)        # the filter at the day's start
        c_rot = (0.55 * rot).where(bull, 0.0)
        c_book = (0.45 * book).where(bull, book)
        b1 = curve_stats(got[("B1 55-45, book in bears", s)])["ret"]
        lr = np.log1p(c_rot + c_book)
        share_r = c_rot.sum() / (c_rot.sum() + c_book.sum()) if (c_rot.sum() + c_book.sum()) != 0 else np.nan
        print("  %s  B1 %+7.0f%% | rotation %+6.0f pts, book %+6.0f pts (rotation %3.0f%% of the gain) | "
              "approx %+7.0f%% | bear days %3d" % (
                  s[:4], b1 * 100, c_rot.sum() * 100, c_book.sum() * 100, share_r * 100,
                  (np.exp(lr.sum()) - 1) * 100, int((~bull).sum())), flush=True)
        if s >= FOLDS[0][0]:
            tot_r += c_rot.sum()
            tot_b += c_book.sum()
    print("\nSix folds (2020-26), summed daily contributions: rotation %+.0f pts, book %+.0f pts; rotation %.0f%%"
          " of the total" % (tot_r * 100, tot_b * 100, tot_r / (tot_r + tot_b) * 100))


if __name__ == "__main__":
    main()
