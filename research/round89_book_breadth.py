"""Round 89: the book aligned with market breadth (from H94's learned controller).

H94's controller, a model choosing each day between the whole book, its longs, its shorts or
nothing, beat the book's yearly return in 4-5 of 6 folds under every neighbouring setting. What
it learned (research/h94_q1_robust.py, and the 7-day returns by state): while most coins are in
uptrends the book's shorts lose, while most are in downtrends its longs earn nothing, and shorts
lose after BTC has fallen more than 15% in 30 days (the bounce). As plain rules, written down
before running, but read off 2020-26, so the 2018-20 holdout (C5) is the real test:

  G1  longs only while at least half the book's coins are in uptrends (240h EMA above 960h),
      shorts only while fewer are (neighbours: 45%, 55%)
  G2  G1, and no shorts while BTC's 30-day return is below -15% (neighbours: -10%, -20%)

Judged on C1-C5 (research/queue.py), on the book alone and on the live bot.

    python -m research.round89_book_breadth
"""
import warnings

from research.queue import judge

DESIGNS = {
    "G1 book aligned with breadth (50%)": (dict(ls_breadth_align=0.5),
                                           [dict(ls_breadth_align=0.45), dict(ls_breadth_align=0.55)]),
    "G2 G1 + no shorts after a BTC crash (-15%)": (
        dict(ls_breadth_align=0.5, short_btc_crash=0.15),
        [dict(ls_breadth_align=0.5, short_btc_crash=0.10), dict(ls_breadth_align=0.5, short_btc_crash=0.20)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 89, the book aligned with breadth", bots=["the live book alone", "live (K2, 3 coins, 75%)"])


if __name__ == "__main__":
    main()
