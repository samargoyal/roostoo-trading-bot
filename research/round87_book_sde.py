"""Round 87: the long-short book from stochastic calculus (the user: "make the book hypothesis
much stronger, use stochastic calculus").

The book holds every coin long or short by its 240h/960h EMA trend, each in inverse proportion
to its volatility. Model each coin as a diffusion, dS/S = mu dt + sigma dW (plus jumps), and
these follow. Written down before running, on the live bot (K2, 3 coins at 75%) and on its book
alone, judged on C1-C5 (research/queue.py):

  S1  Merton sizing: the growth-optimal weight of a geometric Brownian motion is mu / sigma^2,
      not 1 / sigma. The drift is read from the EMA gap (an N-hour EMA lags a steady trend by
      (N - 1) / 2 hours); same gross, a coin capped at 3x its inverse-volatility weight
      (neighbours: caps 2 and 5)
  S2  Kalman drift: size by the t-statistic of a local-linear-trend Kalman filter's slope, full
      from |t| = 2, nothing where it disagrees with the EMAs (neighbours: 1 and 3)
  S3  persistence: trade only coins whose variance ratio (Lo-MacKinlay, 24 hours, last 30 days)
      is at least 1, prices that trend rather than revert (neighbours: 0.9, 1.1)
  S4  jumps: no short while jumps (1 - bipower over realised variance, last week) make up more
      than 30% of the coin's variance, where squeezes come from (neighbours: 20%, 40%)
  S5  volatility management: the book scaled down while BTC's EWMA volatility forecast is above
      its typical level (neighbour: the HAR forecast)
  S6  all of the diffusion-model book: S1, S3 and S4 together (neighbours: S1's caps 2 and 5)

    python -m research.round87_book_sde
"""
import warnings

from research.queue import judge

S6 = dict(ls_sizing="merton", ls_min_variance_ratio=1.0, short_max_jump_share=0.3)
DESIGNS = {
    "S1 Merton sizing mu / sigma^2": (dict(ls_sizing="merton"),
                                      [dict(ls_sizing="merton", ls_sizing_cap=2.0),
                                       dict(ls_sizing="merton", ls_sizing_cap=5.0)]),
    "S2 Kalman drift t-statistic": (dict(ls_sizing="kalman"),
                                    [dict(ls_sizing="kalman", ls_kalman_t=1.0), dict(ls_sizing="kalman", ls_kalman_t=3.0)]),
    "S3 variance ratio at least 1": (dict(ls_min_variance_ratio=1.0),
                                     [dict(ls_min_variance_ratio=0.9), dict(ls_min_variance_ratio=1.1)]),
    "S4 no shorts into jumps (30%)": (dict(short_max_jump_share=0.3),
                                      [dict(short_max_jump_share=0.2), dict(short_max_jump_share=0.4)]),
    "S5 volatility-managed book": (dict(ls_vol_manage="ewma"), [dict(ls_vol_manage="har")]),
    "S6 S1 + S3 + S4": (S6, [dict(S6, ls_sizing_cap=2.0), dict(S6, ls_sizing_cap=5.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 87, the book from stochastic calculus", bots=["the live book alone", "live (K2, 3 coins, 75%)"])


if __name__ == "__main__":
    main()
