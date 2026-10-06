"""Round 91: more return (the user: "I want more return, prioritise this").

The rotation earns most of the return and the book adds a smaller, steadier part, so the
direct lever is the split. On the live bot (K2, 3 coins), the rotation at 75% (live), 80, 85, 90
and 95% of equity (100% is not supported: the book's share divides by it), over the six folds and the 2018-20 holdout: yearly returns, worst
drawdowns and the median 14-day composite. Also hierarchical risk parity for the book (round 90's
P3, the most return of the book designs) at 75% and 85%. No pass rule: this maps return against
risk for the user's choice.

    python -m research.round91_more_return
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

BOT = "live (K2, 3 coins, 75%)"


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"%d%% rotation%s" % (w * 100, " (live)" if w == 0.75 else ""): design(BOT, rotation_weight=w)
             for w in (0.75, 0.8, 0.85, 0.9, 0.95)}
    names["75% rotation, HRP book"] = design(BOT, ls_weighting="hrp")
    names["85% rotation, HRP book"] = design(BOT, rotation_weight=0.85, ls_weighting="hrp")
    res = results_for(names)
    hold = results_for(names, HOLDOUT)
    hy = sorted(next(iter(hold.values())))
    print("Round 91: yearly return by fold %s | 6-year total | worst DD | median 14-day composite (mean) | holdout %s"
          % (" ".join(YEARS), " ".join(hy)))
    for n, by in res.items():
        total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
        h = hold[n]
        print("  %-24s %s | %+9.0f%% | %3.0f%% | %5.2f | %s (DD %s)" % (
            n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100,
            max(by[y]["mdd"] for y in YEARS) * 100, np.mean([by[y]["w14_comp"] for y in YEARS]),
            " ".join("%+5.0f%%" % (h[y]["ret"] * 100) for y in hy),
            " ".join("%.0f%%" % (h[y]["mdd"] * 100) for y in hy)), flush=True)


if __name__ == "__main__":
    main()
