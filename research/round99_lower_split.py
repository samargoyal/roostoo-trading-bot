"""Round 99: a lower split with the efficiency-ratio book (the user: "I want to lower the
rotation's share"). The live bot (K2, 3 coins, book weighted by 30-day efficiency) with the
rotation at 50, 55, 60, 65, 70 and 75% (live) of equity, over the six folds and the 2018-20
holdout: yearly returns, worst drawdowns, the share of 14-day windows won and the median 14-day
composite. No pass rule: this maps return against risk for the user's choice.

    python -m research.round99_lower_split
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

BOT = "live (K2, 3 coins, 75%)"


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"%d%% rotation%s" % (w * 100, " (live)" if w == 0.75 else ""): design(BOT, ls_er_hours=720, rotation_weight=w)
             for w in (0.75, 0.7, 0.65, 0.6, 0.55, 0.5)}
    res, hold = results_for(names), results_for(names, HOLDOUT)
    hy = sorted(next(iter(hold.values())))
    print("Round 99: yearly return by fold %s | 6-year total | worst DD | 14d won | mean 14d composite | holdout %s (DD) [14d]"
          % (" ".join(YEARS), " ".join(hy)))
    for n, by in res.items():
        h = hold[n]
        print("  %-20s %s | %+9.0f%% | %3.0f%% | %3.0f%% | %5.2f | %s (%s) [%s]" % (
            n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS),
            (np.prod([1 + by[y]["ret"] for y in YEARS]) - 1) * 100, max(by[y]["mdd"] for y in YEARS) * 100,
            np.mean([by[y]["w14_pos"] for y in YEARS]) * 100, np.mean([by[y]["w14_comp"] for y in YEARS]),
            " ".join("%+5.0f%%" % (h[y]["ret"] * 100) for y in hy), " ".join("%.0f%%" % (h[y]["mdd"] * 100) for y in hy),
            " ".join("%.2f" % h[y]["w14_comp"] for y in hy)), flush=True)


if __name__ == "__main__":
    main()
