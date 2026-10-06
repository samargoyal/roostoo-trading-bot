"""Round 79: re-picking the rotation as each session opens (the user's question).

Both bots re-pick the rotation once a day at 00:00 UTC. The user asked whether to re-pick as
each trading session opens instead: Asia at 00:00, Europe at 08:00 and the US at 13:00 UTC, so
the picks follow whichever region is driving the market. Round 73 re-picked the multi-horizon
ranking every 4 to 12 hours and found it worse (coins swapped on hourly noise), but at fixed
intervals and on that bot only. Written down before running, on both bots, judged on C1-C5
(research/queue.py):

  R79  re-pick at 00:00, 08:00 and 13:00 UTC (neighbours: 00:00 and 13:00, the Asian and US
       opens only; 00:00, 08:00 and 16:00, evenly spaced)

    python -m research.round79_session_repicks
"""
import warnings

from research.queue import judge

DESIGNS = {
    "R79 re-pick at each session open": (dict(rotation_rebalance_at=[0, 8, 13]),
                                         [dict(rotation_rebalance_at=[0, 13]),
                                          dict(rotation_rebalance_at=[0, 8, 16])]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 79, re-picking at each session open")


if __name__ == "__main__":
    main()
