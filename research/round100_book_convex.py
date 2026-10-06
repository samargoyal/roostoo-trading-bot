"""Round 100: convex optimisation on the efficiency-ratio book (the user: "have you tried convex
optimisation on the new book?").

Rounds 74, 83 and 90 optimised the old book, and the optimisers replace the book's weights, so
with the efficiency tilt on they discard it. These two keep the efficiency ratio as an input.
Written down before running, against the live bot (the efficiency-ratio book) on round 95's
return-first rule and C1-C5, on the book alone and the live bot:

  C1  risk budgeting: each coin's risk contribution in proportion to its 30-day efficiency
      ratio (Spinu's convex problem), from 30 days' covariance of the positions' signed returns
      (neighbours: 14 and 60 days)
  C2  mean-variance: the efficiency ratios' normal scores as expected returns, risk aversion 1,
      10% cap (neighbours: risk aversion 0.5 and 2)

    python -m research.round100_book_convex
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

ER = dict(ls_er_hours=720)
RB = dict(ER, ls_weighting="risk_budget_er")
MV = dict(ER, ls_weighting="mv_er")
DESIGNS = {
    "C1 risk budgets by efficiency": (RB, [dict(RB, ls_cov_hours=336), dict(RB, ls_cov_hours=1440)]),
    "C2 mean-variance on efficiency": (MV, [dict(MV, ls_mv_risk_aversion=0.5), dict(MV, ls_mv_risk_aversion=2.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the live book alone", "live (K2, 3 coins, 75%)"):
        names = {"live now (efficiency book)": design(bot, **ER), "before (inverse volatility)": design(bot)}
        for n, (o, nb) in DESIGNS.items():
            names[n] = design(bot, **o)
            for i, x in enumerate(nb):
                names["%s / n%d" % (n, i + 1)] = design(bot, **x)
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(next(iter(hold.values())))
        b, bh = res["live now (efficiency book)"], hold["live now (efficiency book)"]
        bdd = max(b[y]["mdd"] for y in YEARS)
        print("\nRound 100 on %s: yearly return by fold %s | total | worst DD | 14d better | holdout %s [14d]"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
            dd = max(by[y]["mdd"] for y in YEARS)
            print("  %-36s %s | %+8.0f%% | %3.0f%% | %d/6 | %s [%s] | %s" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100, dd * 100,
                sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy),
                "" if n.startswith("live now") else "return better %d/6, holdout %d/2, DD %+.0f pts" % (
                    sum(by[y]["ret"] > b[y]["ret"] for y in YEARS),
                    sum(hold[n][y]["ret"] > bh[y]["ret"] for y in hy), (dd - bdd) * 100)), flush=True)


if __name__ == "__main__":
    main()
