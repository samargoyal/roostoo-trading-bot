"""Round 90: correlation on the book alone (the user: "have you tried it purely on the book?").

Rounds 74 and 83 weighted the book by its coins' correlations (equal risk contribution, minimum
variance, hierarchical risk parity) inside the live bot, where the book is 25%. Here on the book
alone, with a new use of correlation as a danger signal (in sell-offs crypto's correlations jump
towards one). Written down before running, judged on C1-C5 (research/queue.py), on the book
alone and the live bot:

  P1  equal risk contribution over 30 days' covariance (round 74a)
  P2  minimum variance, 10% cap (round 74b)
  P3  hierarchical risk parity (round 83's O3)
  K1  the book halved while the 10 most traded coins' mean 72-hour correlation is above 0.7
      (neighbours: 0.6, 0.8)
  K2c the book's shorts dropped instead, at the same threshold (neighbours: 0.6, 0.8)

P1-P3 have no neighbour settings of their own; they take round 74's covariance windows (14 and 60
days).

    python -m research.round90_book_correlation
"""
import warnings

from research.queue import judge

DESIGNS = {
    "P1 equal risk contribution": (dict(ls_weighting="erc"),
                                   [dict(ls_weighting="erc", ls_cov_hours=336), dict(ls_weighting="erc", ls_cov_hours=1440)]),
    "P2 minimum variance (10% cap)": (dict(ls_weighting="min_variance"),
                                      [dict(ls_weighting="min_variance", ls_cov_hours=336),
                                       dict(ls_weighting="min_variance", ls_cov_hours=1440)]),
    "P3 hierarchical risk parity": (dict(ls_weighting="hrp"),
                                    [dict(ls_weighting="hrp", ls_cov_hours=336), dict(ls_weighting="hrp", ls_cov_hours=1440)]),
    "K1 book halved at correlation > 0.7": (dict(ls_corr_cut=0.7),
                                            [dict(ls_corr_cut=0.6), dict(ls_corr_cut=0.8)]),
    "K2c no shorts at correlation > 0.7": (dict(ls_corr_cut=0.7, ls_corr_mode="shorts"),
                                           [dict(ls_corr_cut=0.6, ls_corr_mode="shorts"),
                                            dict(ls_corr_cut=0.8, ls_corr_mode="shorts")]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 90, correlation on the book", bots=["the live book alone", "live (K2, 3 coins, 75%)"])


if __name__ == "__main__":
    main()
