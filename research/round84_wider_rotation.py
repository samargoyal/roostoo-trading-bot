"""Round 84: the rotation with more coins, or with shorts (the user's question), on K2.

The rotation holds 2 coins, long only. Written down before running, judged on C1-C5
(research/queue.py) against K2:

  M1  more coins: the top 3 by K2's ranking, equally weighted (neighbours: top 4, top 5); round
      5 found more coins diluted the leaders on the bot of that time
  S1  bear-market shorts: while BTC's filter is off, the sleeve shorts the 2 weakest coins by
      2-week return instead of sitting in PAXG or cash (neighbours: 1, 3); round 19 lost
  S2  a bear-market short basket: every coin in its own downtrend, by inverse volatility, at
      most 20% each (neighbours: 10%, 30%); round 50's screen lost
  S3  an always-on short leg: while the filter is on, 30% of the sleeve shorts the 2 weakest
      coins (negative 2-week return, shorts not crowded) and the picks take 70% (neighbours:
      20%, 40%)

    python -m research.round84_wider_rotation
"""
import warnings

from research.queue import judge

DESIGNS = {
    "M1 the top 3 coins": (dict(rotation_top=3), [dict(rotation_top=4), dict(rotation_top=5)]),
    "S1 bear-market shorts, weakest 2": (dict(rotation_shorts=2), [dict(rotation_shorts=1), dict(rotation_shorts=3)]),
    "S2 bear-market short basket": (dict(rotation_shorts=1, rotation_short_ranking="trend_basket"),
                                    [dict(rotation_shorts=1, rotation_short_ranking="trend_basket", rotation_short_cap=0.1),
                                     dict(rotation_shorts=1, rotation_short_ranking="trend_basket", rotation_short_cap=0.3)]),
    "S3 always-on short leg, 30%": (dict(rotation_short_share=0.3),
                                    [dict(rotation_short_share=0.2), dict(rotation_short_share=0.4)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 84, a wider rotation", bots=["live (K2)"])


if __name__ == "__main__":
    main()
