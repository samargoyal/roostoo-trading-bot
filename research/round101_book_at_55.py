"""Round 101: round 98's stronger efficiency books at the live split (55% rotation, 45% book).

Written down before running, against the live bot (config/comp.json) on round 95's return-first
rule and C1-C5, on the efficiency book alone and the live bot:

  X2  the efficiency tilt squared (neighbours: powers 1.5 and 3)
  X3  the efficiency tilt with fresh trends (turned within 14 days) at double weight
      (neighbours: 7 and 28 days)

    python -m research.round101_book_at_55
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import BOTS, YEARS, design
from research.rounds import results_for

BOTS["the efficiency book alone"] = dict(BOTS["live (55%, efficiency book)"], rotation_weight=0.0)
DESIGNS = {
    "X2 efficiency squared": (dict(ls_er_power=2.0), [dict(ls_er_power=1.5), dict(ls_er_power=3.0)]),
    "X3 + fresh trends 2x (14 days)": (dict(ls_fresh_days=14.0), [dict(ls_fresh_days=7.0), dict(ls_fresh_days=28.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the efficiency book alone", "live (55%, efficiency book)"):
        names = {"live": design(bot)}
        for n, (o, nb) in DESIGNS.items():
            names[n] = design(bot, **o)
            for i, x in enumerate(nb):
                names["%s / n%d" % (n, i + 1)] = design(bot, **x)
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["live"])
        b, bh = res["live"], hold["live"]
        bdd = max(b[y]["mdd"] for y in YEARS)
        bpos = np.mean([b[y]["w14_pos"] for y in YEARS])
        print("\nRound 101 on %s: yearly return by fold %s | total | worst DD | 14d better | won | holdout %s [14d]"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
            dd = max(by[y]["mdd"] for y in YEARS)
            print("  %-36s %s | %+8.0f%% | %3.0f%% | %d/6 | %3.0f%% | %s [%s] | %s" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100, dd * 100,
                sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
                np.mean([by[y]["w14_pos"] for y in YEARS]) * 100,
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy),
                "" if n == "live" else "return better %d/6, holdout %d/2 (14d %d/2), DD %+.0f pts, won %+.0f pts" % (
                    sum(by[y]["ret"] > b[y]["ret"] for y in YEARS),
                    sum(hold[n][y]["ret"] > bh[y]["ret"] for y in hy),
                    sum(hold[n][y]["w14_comp"] > bh[y]["w14_comp"] for y in hy), (dd - bdd) * 100,
                    (np.mean([by[y]["w14_pos"] for y in YEARS]) - bpos) * 100)), flush=True)


if __name__ == "__main__":
    main()
