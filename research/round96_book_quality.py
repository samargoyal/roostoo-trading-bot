"""Round 96: trend quality and the regime for the book (return first; after round 94, whose
efficiency-ratio weights added the most return of any book design).

Written down before running, judged on round 95's return-first rule (yearly return better in at
least 4 of 6 folds and both holdout years, for the design and both neighbours, worst drawdown at
most 2 points deeper) and C1-C5, on the book alone and the live bot:

  Q2  weights times the R^2 of a straight line through the coin's log price over 14 days (a
      smooth trend), re-scaled to the gross, capped at 3x (neighbours: 7 and 30 days)
  T1  a soft regime tilt: the side against the rotation's BTC filter (shorts while BTC trends
      up, longs while it trends down) at half weight, the rest in cash (neighbours: 0.25, 0.75)

    python -m research.round96_book_quality
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

DESIGNS = {
    "Q2 R^2 weights (14 days)": (dict(ls_r2_hours=336), [dict(ls_r2_hours=168), dict(ls_r2_hours=720)]),
    "T1 regime tilt 0.5": (dict(ls_regime_tilt=0.5), [dict(ls_regime_tilt=0.25), dict(ls_regime_tilt=0.75)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the live book alone", "live (K2, 3 coins, 75%)"):
        names = {"baseline": design(bot)}
        for n, (o, nb) in DESIGNS.items():
            names[n] = design(bot, **o)
            for i, x in enumerate(nb):
                names["%s / n%d" % (n, i + 1)] = design(bot, **x)
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["baseline"])
        b, bh = res["baseline"], hold["baseline"]
        bdd = max(b[y]["mdd"] for y in YEARS)
        print("\nRound 96 on %s: yearly return by fold %s | total | worst DD | 14d better | holdout %s"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
            dd = max(by[y]["mdd"] for y in YEARS)
            better = sum(by[y]["ret"] > b[y]["ret"] for y in YEARS)
            b14 = sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS)
            hb = sum(hold[n][y]["ret"] > bh[y]["ret"] for y in hy)
            print("  %-26s %s | %+8.0f%% | %3.0f%% | %d/6 | %s | %s" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100, dd * 100, b14,
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                "" if n == "baseline" else "return better %d/6, holdout %d/2, DD %+.0f pts" % (better, hb, (dd - bdd) * 100)),
                flush=True)


if __name__ == "__main__":
    main()
