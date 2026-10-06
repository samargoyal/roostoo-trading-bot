"""Round 82: the live bot and the multi-horizon bot combined (the user's question).

Written down before running, judged on C1-C5 (research/queue.py) against both:

  K1  half and half: the rotation in two equal sub-sleeves, one picking its top 2 by the 14-day
      return (R54b), the other by the multi-horizon score (7, 14 and 21 days weighted 2/1/1); a
      coin both pick gets both shares; less concentrated, since it may hold up to 4 coins
      (neighbours: the multi half weighted equally; two thirds R54b's)
  K2  one ranking: each candidate's normal scores summed over both bots' rankings, which is the
      7, 14 and 21-day returns weighted 2/2/1 (neighbours 1/2/1, 3/3/1)

    python -m research.round82_combined
"""
import warnings

from research.queue import judge

R54B, MULTI = "ret:336", "multi:168,168,336,504"


def ranking(weights):
    return dict(rotation_ranking="multi", rotation_horizons=[168, 336, 504], rotation_horizon_weights=weights)


DESIGNS = {
    "K1 half R54b picks, half multi picks": (dict(rotation_ranking="return", rotation_ensemble=[R54B, MULTI]),
                                             [dict(rotation_ranking="return", rotation_ensemble=[R54B, "multi:168,336,504"]),
                                              dict(rotation_ranking="return", rotation_ensemble=[R54B, R54B, MULTI])]),
    "K2 one combined ranking (2/2/1)": (ranking([2.0, 2.0, 1.0]),
                                        [ranking([1.0, 2.0, 1.0]), ranking([3.0, 3.0, 1.0])]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 82, the two bots combined")


if __name__ == "__main__":
    main()
