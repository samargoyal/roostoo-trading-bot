"""Round 103: a dollar-neutral book (the user: "what if we make our book dollar neutral?").

The book is long the coins in uptrends and short those in downtrends, so it is mostly long in
bull markets and mostly short in bear ones. Dollar neutral, it would earn the spread between
strong and weak coins whatever the market does, leaving the market's direction to the rotation.
Written down before running, on the efficiency book alone and the live bot (55% rotation),
reported on return, drawdown, the 14-day yardstick and the holdout:

  DN1 the book's longs and shorts each scaled to half its gross (a side with no positions:
      its half in cash)
  DN2 rank: long the stronger half of the coins by trend (EMA gap over volatility), short the
      weaker half, each half the gross; no short where funding is negative

    python -m research.round103_dollar_neutral
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import BOTS, YEARS, design
from research.rounds import results_for

BOTS["the efficiency book alone"] = dict(BOTS["live (55%, efficiency book)"], rotation_weight=0.0)
DESIGNS = {"DN1 each side half the gross": dict(ls_neutral="scale"),
           "DN2 long the stronger half, short the weaker": dict(ls_neutral="rank")}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the efficiency book alone", "live (55%, efficiency book)"):
        names = {"live": design(bot)}
        names.update({n: design(bot, **o) for n, o in DESIGNS.items()})
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["live"])
        b = res["live"]
        print("\nRound 103 on %s: yearly return by fold %s | total | worst DD | 14d better | holdout %s [14d]"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            print("  %-44s %s | %+8.0f%% | %3.0f%% | %d/6 | %s [%s]" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS),
                (np.prod([1 + by[y]["ret"] for y in YEARS]) - 1) * 100, max(by[y]["mdd"] for y in YEARS) * 100,
                sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy)), flush=True)


if __name__ == "__main__":
    main()
