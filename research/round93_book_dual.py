"""Round 93: dual momentum and stronger fresh trends for the book (return first).

Round 92's fresh-trend boost beat the book's yearly return in 4 of 6 folds (the 14-day yardstick
2/6). Written down before running, judged on C1-C5 on the book alone and the live bot:

  D1  dual momentum: a long also needs the coin to have beaten BTC over the last 30 days, a short
      to have lagged it; the rest in cash (neighbours: 14 and 60 days)
  D2  D1 with the held positions filling the book's whole gross (neighbours: 14 and 60 days)
  F1  fresh trends (turned within 14 days) at three times their weight (neighbours: 7 and 28 days)

    python -m research.round93_book_dual
"""
import warnings

from research.queue import judge

DESIGNS = {
    "D1 dual momentum (30 days), rest in cash": (dict(ls_rel_hours=720),
                                                 [dict(ls_rel_hours=336), dict(ls_rel_hours=1440)]),
    "D2 dual momentum (30 days), full gross": (dict(ls_rel_hours=720, ls_rel_fill=True),
                                               [dict(ls_rel_hours=336, ls_rel_fill=True),
                                                dict(ls_rel_hours=1440, ls_rel_fill=True)]),
    "F1 fresh trends (14 days) at 3x": (dict(ls_fresh_days=14.0, ls_fresh_boost=3.0),
                                        [dict(ls_fresh_days=7.0, ls_fresh_boost=3.0),
                                         dict(ls_fresh_days=28.0, ls_fresh_boost=3.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 93, dual momentum and fresh trends", bots=["the live book alone", "live (K2, 3 coins, 75%)"])


if __name__ == "__main__":
    main()
