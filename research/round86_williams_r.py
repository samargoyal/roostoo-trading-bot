"""Round 86: Williams %R on the rotation (the user's request; H88's WR5, the 14-day %R, was the
one %R design with an edge on its own, on longs).

Written down before running, on the live bot (K2, 3 coins at 75%) and R54b, judged on C1-C5
(research/queue.py):

  W1  a new pick needs its 14-day %R at or above -20, near its two-week high (neighbours: -30;
      a 21-day %R)
  W2  a held pick leaves when its 14-day %R falls below -50, the lower half of its range, and
      cools down for a day (neighbours: -40, -60)
  W3  both, H88's WR5 inside the rotation (neighbours: 10- and 21-day %R)

    python -m research.round86_williams_r
"""
import warnings

from research.queue import judge


def wr(hours=336, entry=-100.0, exit_=-100.0):
    return dict(rotation_wr_hours=hours, rotation_wr_entry=entry, rotation_wr_exit=exit_)


DESIGNS = {
    "W1 picks near their 14-day high (%R >= -20)": (wr(entry=-20), [wr(entry=-30), wr(hours=504, entry=-20)]),
    "W2 picks leave below %R -50": (wr(exit_=-50), [wr(exit_=-40), wr(exit_=-60)]),
    "W3 both (WR5)": (wr(entry=-20, exit_=-50), [wr(hours=240, entry=-20, exit_=-50), wr(hours=504, entry=-20, exit_=-50)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 86, Williams %R on the rotation", bots=["live (K2, 3 coins, 75%)", "live (R54b)"])


if __name__ == "__main__":
    main()
