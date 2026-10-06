"""Round 94: the book in choppy markets (return first).

The book's weak years (2022-23 +7%, 2024-25 +27% on its own) are chop: trends flip back and
forth and each flip pays costs and a loss. Written down before running, judged on C1-C5 on the
book alone and the live bot:

  H1  hysteresis: a coin keeps its side until its 240h/960h EMA gap is 1% past zero the other
      way (neighbours: 0.5%, 2%)
  E1  efficiency: weights times Kaufman's efficiency ratio over 14 days (net move over path
      length), re-scaled to the book's gross, capped at 3x (neighbours: 7 and 30 days)
  HF  H1 with round 93's fresh trends at 3x (neighbours: H1's 0.5% and 2%)

    python -m research.round94_book_chop
"""
import warnings

from research.queue import judge

FRESH = dict(ls_fresh_days=14.0, ls_fresh_boost=3.0)
DESIGNS = {
    "H1 hysteresis 1%": (dict(ls_hysteresis=0.01), [dict(ls_hysteresis=0.005), dict(ls_hysteresis=0.02)]),
    "E1 efficiency-ratio weights (14 days)": (dict(ls_er_hours=336), [dict(ls_er_hours=168), dict(ls_er_hours=720)]),
    "HF hysteresis 1% + fresh trends 3x": (dict(FRESH, ls_hysteresis=0.01),
                                           [dict(FRESH, ls_hysteresis=0.005), dict(FRESH, ls_hysteresis=0.02)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 94, the book in chop", bots=["the live book alone", "live (K2, 3 coins, 75%)"])


if __name__ == "__main__":
    main()
