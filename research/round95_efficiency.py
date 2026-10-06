"""Round 95: round 94's efficiency-ratio book (E1) judged on return, as the user asked (return
first). E1 on the book alone made +981% in 2020-21 against +484% and more in 4 of 6 folds, but
failed C1 (median 14-day composite better in 2 of 6). Written down before running: E1 and its
neighbours (7 and 30 days), E1 with fresh trends at 3x and with hierarchical risk parity, on the
book alone and the live bot, over the six folds and the 2018-20 holdout. A return-first pass:
yearly return better in at least 4 of 6 folds AND in both holdout years, for E1 and both
neighbours, with a worst drawdown at most 2 points deeper.

    python -m research.round95_efficiency
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

VARIANTS = {"baseline": {}, "E1 efficiency 14 days": dict(ls_er_hours=336), "E1 7 days": dict(ls_er_hours=168),
            "E1 30 days": dict(ls_er_hours=720), "E1 + fresh trends 3x": dict(ls_er_hours=336, ls_fresh_days=14.0,
                                                                              ls_fresh_boost=3.0),
            "E1 + HRP": dict(ls_er_hours=336, ls_weighting="hrp")}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the live book alone", "live (K2, 3 coins, 75%)"):
        names = {n: design(bot, **o) for n, o in VARIANTS.items()}
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["baseline"])
        b, bh = res["baseline"], hold["baseline"]
        bdd = max(b[y]["mdd"] for y in YEARS)
        print("\nRound 95 on %s: yearly return by fold %s | total | worst DD | holdout %s" % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
            dd = max(by[y]["mdd"] for y in YEARS)
            better = sum(by[y]["ret"] > b[y]["ret"] for y in YEARS)
            hb = sum(hold[n][y]["ret"] > bh[y]["ret"] for y in hy)
            print("  %-24s %s | %+8.0f%% | %3.0f%% | %s | %s" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100, dd * 100,
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                "" if n == "baseline" else "return better %d/6, holdout %d/2, DD %+.0f pts" % (better, hb, (dd - bdd) * 100)),
                flush=True)


if __name__ == "__main__":
    main()
