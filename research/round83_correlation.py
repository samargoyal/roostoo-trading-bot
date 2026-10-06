"""Round 83: correlation and convex optimisation on the live bot (K2), the user's request.

Rounds 5, 6, 73 and 74 weighted the picks or the book by optimisers that ignore which coin is
strongest (equal risk, minimum variance), and each leaned on the calmer coin, the weaker mover in
a momentum book. These three use correlation without that flaw. Written down before running,
judged on C1-C5 (research/queue.py) against K2:

  O1  a diversified pair: the second pick is the one among the top 5 that maximises its normal
      score less half its correlation with the first (14-day hourly returns), so the two picks
      are less one bet (neighbours: a quarter, a whole)
  O2  mean-variance with momentum: the top 4 weighted by maximising the ranking's normal scores
      (expected returns) less risk aversion 1 times the variance (covariance of 14 days, scaled
      to a mean variance of 1), each at most 60%; a concave problem (neighbours: risk aversion
      0.5 and 2). O2b, information: the same over the top 2
  O3  hierarchical risk parity for the long-short book: its positions clustered by the
      correlation of their side-adjusted returns (30 days), the budget split down the cluster
      tree by inverse variance, no matrix inverted (neighbours: 14 and 60 days)

    python -m research.round83_correlation
"""
import warnings

from research.queue import judge


def mv(top, risk):
    return dict(rotation_top=top, rotation_weighting="mv", rotation_mv_risk_aversion=risk, rotation_max_weight=0.6)


DESIGNS = {
    "O1 diversified second pick": (dict(rotation_corr_lambda=0.5),
                                   [dict(rotation_corr_lambda=0.25), dict(rotation_corr_lambda=1.0)]),
    "O2 mean-variance over the top 4": (mv(4, 1.0), [mv(4, 0.5), mv(4, 2.0)]),
    "O2b mean-variance over the top 2": (mv(2, 1.0), [mv(2, 0.5), mv(2, 2.0)]),
    "O3 long-short book by HRP": (dict(ls_weighting="hrp", ls_cov_hours=720),
                                  [dict(ls_weighting="hrp", ls_cov_hours=336), dict(ls_weighting="hrp", ls_cov_hours=1440)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 83, correlation and convex optimisation", bots=["live (K2)"])


if __name__ == "__main__":
    main()
