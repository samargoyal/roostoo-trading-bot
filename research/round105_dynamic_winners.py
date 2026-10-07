"""Round 105: winners and losers chosen dynamically (the user: "don't hard-code a value like 5,
make it dynamic").

Round 104 held a fixed number of coins a side and re-ranked every hour; it traded about five
times the book's value a day and its fees ruined it. Here the count follows the market and the
membership has hysteresis: each hour (or once a day) the rotation's ranking score (7-, 14- and
21-day returns, 2/2/1) is standardised across the coins (z); a coin becomes a winner above
+entry and stays one until its z falls to 0, a loser below -entry until it rises to 0, so the
number of each changes with the spread between coins without flipping on noise. The book
replaced, filling its share, no short where funding is negative. Written down before running
(after round 104's churn was found), on the book alone and the live bot (55% rotation):

  DZ1 winners above z +1 long, losers below -1 short, dollar neutral, re-ranked hourly
      (neighbours: entry 0.75 and 1.25)
  DZ1d the same re-ranked once a day
  DZ2 every coin long or short by the sign of z, sized by |z| (and inverse volatility),
      re-ranked once a day (neighbour: dollar neutral)
  DZ3 dual momentum: winners above +0.5 with a positive 14-day return, losers below -0.5 in
      their own downtrend, hourly (neighbours: entry 0.25 and 1.0)

    python -m research.round105_dynamic_winners
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import BOTS, YEARS, design
from research.rounds import results_for

BOTS["the efficiency book alone"] = dict(BOTS["live (55%, efficiency book)"], rotation_weight=0.0)
FILL = dict(ls_rel_fill=True)
DESIGNS = {
    "DZ1 beyond 1 sd, neutral": dict(FILL, ls_xs_mode="z", ls_xs_z=1.0, ls_neutral="scale"),
    "DZ1 / 0.75 sd": dict(FILL, ls_xs_mode="z", ls_xs_z=0.75, ls_neutral="scale"),
    "DZ1 / 1.25 sd": dict(FILL, ls_xs_mode="z", ls_xs_z=1.25, ls_neutral="scale"),
    "DZ1d re-ranked daily": dict(FILL, ls_xs_mode="z", ls_xs_z=1.0, ls_neutral="scale", ls_xs_daily=True),
    "DZ2 every coin by |z|, daily": dict(FILL, ls_xs_mode="weighted", ls_xs_daily=True),
    "DZ2 / dollar neutral": dict(FILL, ls_xs_mode="weighted", ls_xs_daily=True, ls_neutral="scale"),
    "DZ3 dual momentum, 0.5 sd": dict(FILL, ls_xs_mode="dual", ls_xs_z=0.5),
    "DZ3 / 0.25 sd": dict(FILL, ls_xs_mode="dual", ls_xs_z=0.25),
    "DZ3 / 1.0 sd": dict(FILL, ls_xs_mode="dual", ls_xs_z=1.0),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the efficiency book alone", "live (55%, efficiency book)"):
        names = {"live": design(bot)}
        names.update({n: design(bot, **o) for n, o in DESIGNS.items()})
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["live"])
        b = res["live"]
        print("\nRound 105 on %s: yearly return by fold %s | total | worst DD | 14d better | holdout %s [14d]"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            print("  %-34s %s | %+8.0f%% | %3.0f%% | %d/6 | %s [%s]" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS),
                (np.prod([1 + by[y]["ret"] for y in YEARS]) - 1) * 100, max(by[y]["mdd"] for y in YEARS) * 100,
                sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy)), flush=True)


if __name__ == "__main__":
    main()
