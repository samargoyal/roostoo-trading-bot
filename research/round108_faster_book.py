"""Round 108: a book that turns short sooner (the user: in the book alone only 2 coins are short
while the market falls).

The book is long a coin while its 240h EMA is above its 960h EMA and short while below; after a
rally, a few days of falling prices do not cross them. Written down before running, on the
efficiency book alone (the competition account now) and on 55/45, reported on return, drawdown,
the crash year, the 14-day yardstick and the holdout:

  F1  shorts on a faster trend, 168h/672h, longs on 240h/960h, flat while they disagree
      (round 64's option; neighbours: 120h/480h and 200h/800h for the shorts)
  F2  the whole book on a faster trend, 120h/480h (neighbours: 168h/672h and 72h/288h)

    python -m research.round108_faster_book
"""
import json
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import BOTS, YEARS, design
from research.rounds import results_for

BOTS["book alone (live)"] = json.load(open("config/comp.json"))["strategy"]
BOTS["55/45"] = json.load(open("config/comp_er_55.json"))["strategy"]
DESIGNS = {
    "F1 shorts on 168h/672h": dict(ls_short_trend=[168, 672]),
    "F1 / shorts 120h/480h": dict(ls_short_trend=[120, 480]),
    "F1 / shorts 200h/800h": dict(ls_short_trend=[200, 800]),
    "F2 whole book 120h/480h": dict(ls_trend=[120, 480]),
    "F2 / 168h/672h": dict(ls_trend=[168, 672]),
    "F2 / 72h/288h": dict(ls_trend=[72, 288]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("book alone (live)", "55/45"):
        names = {"as now": design(bot)}
        names.update({n: design(bot, **o) for n, o in DESIGNS.items()})
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["as now"])
        b = res["as now"]
        print("\nRound 108 on %s: yearly return by fold %s | total | worst DD | 14d better | holdout %s [14d]"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            print("  %-26s %s | %+8.0f%% | %3.0f%% | %d/6 | %s [%s]" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS),
                (np.prod([1 + by[y]["ret"] for y in YEARS]) - 1) * 100, max(by[y]["mdd"] for y in YEARS) * 100,
                sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy)), flush=True)


if __name__ == "__main__":
    main()
