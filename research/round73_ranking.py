"""Round 73: the multi-horizon ranking, adopted for the competition account.

Round 71's two near misses (a rank buffer, and ranking on 7-, 14- and 21-day returns together)
each broke on one neighbour. The user judged those misses unimportant, so this round ran what was
left, after the fact and so a weaker test than the rule's:

  1  both near misses, and the two together, in the untouched holdout (2018-20)
  2  the user asked to re-pick more often than daily: the multi-horizon ranking re-picked every
     24, 12, 8, 6 and 4 hours
  3  the user asked about weights: the three horizons weighted 2/1/1, 3/2/1 or 1/1/2, and the
     two picks weighted by inverse volatility or equal risk instead of equally

Adopted, by the user's choice (5 October 2026): the multi-horizon ranking re-picked daily with
the 7-, 14- and 21-day returns weighted 2/1/1 (config/comp.json; the previous settings are
config/comp_r54b.json).

    python -m research.round73_ranking
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round66_attention import design
from research.round68_profits import plain
from research.round71_entries_exits import DESIGNS
from research.rounds import results_for

BUFFER = DESIGNS["R71d rank buffer (top 3)"][0]
MULTI = DESIGNS["R71i multi-horizon ranking"][0]


def table(names, title):
    for label, folds in (("2020-26", FOLDS), ("2018-20, the holdout", HOLDOUT)):
        res = results_for(names, folds)
        ys = sorted(res["live bot (R54b)"])
        base = res["live bot (R54b)"]
        print("\n%s, %s" % (title, label))
        for n, by in res.items():
            print("  %-36s %s | total %+.0f%% | worst DD %.0f%% | score better %d/%d | 14-day score better %d/%d" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in ys), (np.prod([1 + by[y]["ret"] for y in ys]) - 1) * 100,
                max(by[y]["mdd"] for y in ys) * 100, sum(by[y]["comp"] > base[y]["comp"] for y in ys), len(ys),
                sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in ys), len(ys)), flush=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    live = {"live bot (R54b)": design()}
    table(dict(live, **{"rank buffer (top 3)": plain(**BUFFER), "multi-horizon (7/14/21 days)": plain(**MULTI),
                        "both together": plain(**BUFFER, **MULTI)}), "1. round 71's near misses")
    table(dict(live, **{"multi-horizon, re-picked every %dh" % h: plain(rotation_rebalance_hours=h, **MULTI)
                        for h in (24, 12, 8, 6, 4)}), "2. how often to re-pick")
    table(dict(live, **{"multi-horizon, weights %s" % "/".join("%g" % x for x in w): plain(rotation_horizon_weights=w, **MULTI)
                        for w in ([1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 2.0, 1.0], [1.0, 1.0, 2.0])},
               **{"multi-horizon, picks %s" % m: plain(rotation_weighting=m, **MULTI) for m in ("inverse_vol", "erc")}),
          "3. weights")


if __name__ == "__main__":
    main()
