"""H97: no momentum, two books (the user: "55% A10, the rest the book").

H96's A10 (the live book alone with the efficiency tilt cubed) at 55% of equity and the live
efficiency book alone at 45%, mixed daily, against the live bot (55% rotation, 45% book), over
the six folds and the 2018-20 holdout. Also the two books alone. No pass rule: for the user's
choice.

    python -m research.h97_two_books
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from research.folds import FOLDS
from research.h50_long_short import blend
from research.h60_swing_mft import curve_stats
from research.h95_aggressive_sleeve import LIVE, bot_curve
from research.h96_more_aggressive import MORE
from research.holdout2018 import HOLDOUT

TAGS = {"live": LIVE, "A10 sharpest efficiency book": MORE["A10 sharpest efficiency book"],
        "efficiency book alone": dict(LIVE, rotation_weight=0.0)}


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = FOLDS + HOLDOUT
    with ProcessPoolExecutor(max_workers=8) as pool:
        curves = {(t, s): c for t, s, c in pool.map(bot_curve, [(t, st, f) for t, st in TAGS.items() for f in folds])}
    rows = {}
    for start, _ in folds:
        y = start[:4]
        a10, book = curves[("A10 sharpest efficiency book", start)], curves[("efficiency book alone", start)]
        rows.setdefault("live (55% momentum, 45% book)", {})[y] = curve_stats(curves[("live", start)])
        rows.setdefault("55% A10, 45% book", {})[y] = curve_stats(blend(book, a10, 0.55))
        rows.setdefault("A10 alone", {})[y] = curve_stats(a10)
        rows.setdefault("efficiency book alone", {})[y] = curve_stats(book)
    years, hy = [s[:4] for s, _ in FOLDS], [s[:4] for s, _ in HOLDOUT]
    base = rows["live (55% momentum, 45% book)"]
    print("H97: yearly return (worst drawdown) by fold %s | six years | worst DD | mean median 14d composite | "
          "14d better | holdout %s [14d]" % (" ".join(years), " ".join(hy)))
    for n, by in rows.items():
        print("  %-32s %s | %+8.0f%% | %3.0f%% | %5.2f | %d/6 | %s [%s]" % (
            n, " ".join("%+6.0f%%(%2.0f)" % (by[y]["ret"] * 100, by[y]["mdd"] * 100) for y in years),
            (np.prod([1 + by[y]["ret"] for y in years]) - 1) * 100, max(by[y]["mdd"] for y in years) * 100,
            np.mean([by[y]["w14"] for y in years]), sum(by[y]["w14"] > base[y]["w14"] for y in years),
            " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in hy), " ".join("%.2f" % by[y]["w14"] for y in hy)),
            flush=True)


if __name__ == "__main__":
    main()
