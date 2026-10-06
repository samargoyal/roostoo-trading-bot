"""Round 80: a faster multi-horizon ranking (the user: "I want it to be more fast").

The multi-horizon ranking (config/comp_multi.json: 7, 14 and 21 days weighted 2/1/1) switches
coins about 22 times a year. Re-picking more often than daily was worse (rounds 73 and 79), so
these make the ranking itself react faster, leaning on more recent returns. Written down before
running, on both bots, judged on C1-C5 (research/queue.py):

  F1  7, 14 and 21 days weighted 4/1/1 (neighbours 3/1/1, 6/1/1)
  F2  3, 7 and 14 days weighted 2/1/1 (neighbours: weighted 1/1/1; 2, 7 and 14 days 2/1/1)
  F3  3 and 7 days, equally (neighbours: 2 and 7 days; 3 and 10 days)

    python -m research.round80_faster_ranking
"""
import warnings

from research.queue import judge


def multi(horizons_days, weights):
    return dict(rotation_ranking="multi", rotation_horizons=[d * 24 for d in horizons_days],
                rotation_horizon_weights=[float(w) for w in weights])


DESIGNS = {
    "F1 7/14/21 days weighted 4/1/1": (multi([7, 14, 21], [4, 1, 1]),
                                       [multi([7, 14, 21], [3, 1, 1]), multi([7, 14, 21], [6, 1, 1])]),
    "F2 3/7/14 days weighted 2/1/1": (multi([3, 7, 14], [2, 1, 1]),
                                      [multi([3, 7, 14], [1, 1, 1]), multi([2, 7, 14], [2, 1, 1])]),
    "F3 3 and 7 days": (multi([3, 7], [1, 1]), [multi([2, 7], [1, 1]), multi([3, 10], [1, 1])]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 80, a faster ranking")


if __name__ == "__main__":
    main()
