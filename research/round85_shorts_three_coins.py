"""Round 85: shorting in the 3-coin rotation (the user keeps K2 with 3 coins, uncommitted).

Baseline: K2 with the rotation holding its top 3 ("K2, 3 coins" in research/queue.py). The
long-only rotation carries the market's falls; round 84 found bear-market shorts squeezed and an
always-on short leg costly in bull years. Written down before running, judged on C1-C5:

  X1  a short leg: 20% of the sleeve always shorts the 2 weakest coins by 2-week return
      (neighbours 10%, 30%)
  X2  the same, the weakest by the multi-horizon score (neighbours 10%, 30%)
  X3  the same, only coins in their own downtrend (168h EMA below 672h), so a short follows a
      falling coin rather than a merely lagging one (neighbours 10%, 30%)
  X4  a bear-market short basket with the 3 coins (neighbours: caps of 10%, 30% a coin)
  X5  X3 with the rotation at 80% of the account, the short leg hedging part of it (neighbours
      75%, 85%)
  X6  X3 with the top 4 (neighbours: the top 5; the top 4 with a 30% leg)

and, for information, the 3-coin bot's split between the rotation and the book from 50% to 90%.

    python -m research.round85_shorts_three_coins
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.queue import design, judge
from research.rounds import results_for


def leg(share, **extra):
    return dict(dict(rotation_short_share=share), **extra)


TREND = dict(rotation_short_trend=True)
DESIGNS = {
    "X1 short leg 20%, weakest by return": (leg(0.2), [leg(0.1), leg(0.3)]),
    "X2 short leg 20%, weakest by K2 score": (leg(0.2, rotation_short_by="multi"),
                                              [leg(0.1, rotation_short_by="multi"), leg(0.3, rotation_short_by="multi")]),
    "X3 short leg 20%, downtrends only": (leg(0.2, **TREND), [leg(0.1, **TREND), leg(0.3, **TREND)]),
    "X4 bear-market short basket": (dict(rotation_shorts=1, rotation_short_ranking="trend_basket"),
                                    [dict(rotation_shorts=1, rotation_short_ranking="trend_basket", rotation_short_cap=0.1),
                                     dict(rotation_shorts=1, rotation_short_ranking="trend_basket", rotation_short_cap=0.3)]),
    "X5 X3 with the rotation at 80%": (leg(0.2, rotation_weight=0.8, **TREND),
                                       [leg(0.2, rotation_weight=0.75, **TREND), leg(0.2, rotation_weight=0.85, **TREND)]),
    "X6 X3 with the top 4": (leg(0.2, rotation_top=4, **TREND),
                             [leg(0.2, rotation_top=5, **TREND), leg(0.3, rotation_top=4, **TREND)]),
}


def splits() -> None:
    names = {"%d/%d" % (w * 100, 100 - w * 100): design("K2, 3 coins", rotation_weight=w)
             for w in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)}
    res = results_for(names)
    ys = [f[0][:4] for f in FOLDS]
    print("\nfor information, the 3-coin bot's split between the rotation and the book:")
    for n, by in res.items():
        print("  %-6s %s | total %+.0f%% | worst DD %.0f%% | 14d windows won %.0f%%" % (
            n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in ys), (np.prod([1 + by[y]["ret"] for y in ys]) - 1) * 100,
            max(by[y]["mdd"] for y in ys) * 100, np.mean([by[y]["w14_pos"] for y in ys]) * 100), flush=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 85, shorting in the 3-coin rotation", bots=["K2, 3 coins"])
    splits()


if __name__ == "__main__":
    main()
