"""Round 98: more from the efficiency ratio (the book; return first). Round 97 found the 30-day
efficiency tilt robust on return, risk, costs and the holdout. Written down before running, on
round 95's return-first rule, on the book alone and the live bot:

  X1  keep only the cleaner half of the book's coins by efficiency, filling the gross
      (neighbours: the top third and two thirds)
  X2  a sharper tilt, efficiency squared (neighbours: powers 1.5 and 3)
  X3  the 30-day tilt with fresh trends at double weight (neighbours: 7 and 28 days)

    python -m research.round98_efficiency_more
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

ER = dict(ls_er_hours=720)
DESIGNS = {
    "X1 the cleaner half": (dict(ER, ls_er_keep=0.5), [dict(ER, ls_er_keep=1 / 3), dict(ER, ls_er_keep=2 / 3)]),
    "X2 efficiency squared": (dict(ER, ls_er_power=2.0), [dict(ER, ls_er_power=1.5), dict(ER, ls_er_power=3.0)]),
    "X3 + fresh trends 2x (14 days)": (dict(ER, ls_fresh_days=14.0),
                                       [dict(ER, ls_fresh_days=7.0), dict(ER, ls_fresh_days=28.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the live book alone", "live (K2, 3 coins, 75%)"):
        names = {"baseline": design(bot), "E1 30 days": design(bot, **ER)}
        for n, (o, nb) in DESIGNS.items():
            names[n] = design(bot, **o)
            for i, x in enumerate(nb):
                names["%s / n%d" % (n, i + 1)] = design(bot, **x)
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["baseline"])
        b, bh = res["baseline"], hold["baseline"]
        bdd = max(b[y]["mdd"] for y in YEARS)
        print("\nRound 98 on %s: yearly return by fold %s | total | worst DD | 14d better | holdout %s"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
            dd = max(by[y]["mdd"] for y in YEARS)
            print("  %-34s %s | %+8.0f%% | %3.0f%% | %d/6 | %s | %s" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100, dd * 100,
                sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                "" if n == "baseline" else "return better %d/6, holdout %d/2, DD %+.0f pts" % (
                    sum(by[y]["ret"] > b[y]["ret"] for y in YEARS),
                    sum(hold[n][y]["ret"] > bh[y]["ret"] for y in hy), (dd - bdd) * 100)), flush=True)


if __name__ == "__main__":
    main()
