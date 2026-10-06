"""Round 92: a book that makes more (the user: "I want a better book hypothesis, keep
experimenting"; return first).

The book holds all ~45 coins, long or short by the 240h/960h EMA trend, by inverse volatility.
Spread that thin it earns little in most years. Written down before running, judged on C1-C5
(research/queue.py) on the book alone and the live bot:

  N1  concentration: only the 10 coins with the strongest trends (the EMA gap over volatility),
      long or short, at the book's full gross (neighbours: 6, 15)
  N2  longs on a faster trend (168h/672h EMAs; crypto rallies start fast), shorts on 240h/960h,
      flat where they disagree (neighbours: longs 120h/480h and 200h/800h)
  N3  fresh trends: a coin whose trend turned in the last 14 days at twice its weight, the book
      re-scaled to its gross (neighbours: 7 and 28 days)
  N4  N1 and N2 together (neighbours: N1's 6 and 15)

    python -m research.round92_book_return
"""
import warnings

from research.queue import judge

FAST = dict(ls_trend=[168, 672], ls_short_trend=[240, 960])
DESIGNS = {
    "N1 the 10 strongest trends": (dict(ls_top_n=10), [dict(ls_top_n=6), dict(ls_top_n=15)]),
    "N2 longs on 168h/672h, shorts on 240h/960h": (FAST, [dict(FAST, ls_trend=[120, 480]),
                                                          dict(FAST, ls_trend=[200, 800])]),
    "N3 fresh trends at double weight (14 days)": (dict(ls_fresh_days=14.0),
                                                   [dict(ls_fresh_days=7.0), dict(ls_fresh_days=28.0)]),
    "N4 N1 + N2": (dict(FAST, ls_top_n=10), [dict(FAST, ls_top_n=6), dict(FAST, ls_top_n=15)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 92, a book that makes more", bots=["the live book alone", "live (K2, 3 coins, 75%)"])


if __name__ == "__main__":
    main()
