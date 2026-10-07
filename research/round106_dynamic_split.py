"""Round 106: the split set by a fast regime (the user: "make the 55/45 shares dynamic with
the regime, and the regime very reactive").

The live bot holds the rotation at 55% and the efficiency book at 45% whatever the market. Here
the rotation's share is set every hour by a fast read of the market: 75% in a bull market, 55%
in a neutral one and 35% in a bear one (the book takes the rest), with the rotation re-picked
whenever its share changes. Written down before running, on the live bot, reported on return,
drawdown, the 14-day yardstick and the holdout:

  S1  BTC alone: a bull market while its close is above its 50-hour EMA and that above its
      200-hour EMA (about 2 and 8 days), a bear market while below both with the 50h below
  S2  breadth: a bull market while over 60% of the coins are up over 72 hours, a bear market
      while under 40% are
  S3  both: a bull or bear market only when S1 and S2 agree

Neighbours for each: the shares 70/55/40 and 80/55/30.

    python -m research.round106_dynamic_split
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

BOT = "live (55%, efficiency book)"
DESIGNS = {}
for key, label in (("btc", "S1 BTC 50h/200h"), ("breadth", "S2 breadth 72h"), ("both", "S3 both agree")):
    DESIGNS[label + " 75/55/35"] = dict(split_regime=key)
    DESIGNS[label + " 70/55/40"] = dict(split_regime=key, split_shares=[0.70, 0.55, 0.40])
    DESIGNS[label + " 80/55/30"] = dict(split_regime=key, split_shares=[0.80, 0.55, 0.30])


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"live (fixed 55/45)": design(BOT)}
    names.update({n: design(BOT, **o) for n, o in DESIGNS.items()})
    res, hold = results_for(names), results_for(names, HOLDOUT)
    hy = sorted(hold["live (fixed 55/45)"])
    b = res["live (fixed 55/45)"]
    print("Round 106: yearly return by fold %s | total | worst DD | 14d better | won | holdout %s [14d]"
          % (" ".join(YEARS), " ".join(hy)))
    for n, by in res.items():
        print("  %-30s %s | %+8.0f%% | %3.0f%% | %d/6 | %3.0f%% | %s [%s]" % (
            n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS),
            (np.prod([1 + by[y]["ret"] for y in YEARS]) - 1) * 100, max(by[y]["mdd"] for y in YEARS) * 100,
            sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
            np.mean([by[y]["w14_pos"] for y in YEARS]) * 100,
            " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
            " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy)), flush=True)


if __name__ == "__main__":
    main()
