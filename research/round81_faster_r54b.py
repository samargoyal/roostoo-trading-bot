"""Round 81: the live bot (R54b) made faster (the user's request).

Round 80 made the multi-horizon ranking faster and lost. Here each of the live bot's three
speeds is turned up, written down before running and judged on C1-C5 (research/queue.py)
against the live bot only:

  S1  the rotation ranks on the 10-day return instead of the 14-day (neighbours 7, 12 days);
      round 9 found 14 days a peak on the bot of that time
  S2  BTC's trend filter on its 5-day against 20-day EMAs instead of 7 against 28, so the
      rotation enters and leaves sooner (neighbours 4/16 and 6/24 days)
  S3  the long-short book on each coin's 7-day against 28-day EMAs instead of 10 against 40,
      flipping sides sooner (neighbours 5/20 and 8/33 days)
  S4  all three together (neighbours: S4 with 7-day and with 12-day rankings)

    python -m research.round81_faster_r54b
"""
import warnings

from research.queue import judge

FAST = dict(rotation_lookback=240, rotation_trend_fast=120, rotation_trend_slow=480, ls_trend=[168, 672])
DESIGNS = {
    "S1 rotation on 10-day returns": (dict(rotation_lookback=240),
                                      [dict(rotation_lookback=168), dict(rotation_lookback=288)]),
    "S2 BTC filter 5/20 days": (dict(rotation_trend_fast=120, rotation_trend_slow=480),
                                [dict(rotation_trend_fast=96, rotation_trend_slow=384),
                                 dict(rotation_trend_fast=144, rotation_trend_slow=576)]),
    "S3 book on 7/28-day trends": (dict(ls_trend=[168, 672]),
                                   [dict(ls_trend=[120, 480]), dict(ls_trend=[200, 800])]),
    "S4 all three faster": (FAST, [dict(FAST, rotation_lookback=168), dict(FAST, rotation_lookback=288)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 81, the live bot faster", bots=["live (R54b)"])


if __name__ == "__main__":
    main()
