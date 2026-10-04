"""Choosing the rotation sleeve's share of the account, in steps of 5%.

The competition scores one 14-day window and ranks by return first. Rules fixed before the new
grid points were computed (the user asked for an optimised share in steps of 5%):

  Grid       30%, 35%, ..., 95% of the account in the rotation (100% leaves no book)
  Objective  in each of the six folds, rank the shares by the median 14-day composite (the
             competition's yardstick); sum the ranks over the folds, so every year counts the
             same and 2020-21's huge values cannot dominate
  Robustness smooth the objective over each share and its neighbours (+-5%) and take the best
             smoothed value: a broad plateau, not a lucky spike
  Confirm    the chosen share must beat 40% on the median 14-day composite in both untouched
             holdout years (October 2018 - October 2020)
Drawdowns and yearly composites are shown for the cost.

    python -m research.rotation_weight
"""
import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.rounds import results_for

GRID = list(range(30, 100, 5))


def design(pct: int) -> dict:
    return {} if pct == 40 else {"strategy": {"rotation_weight": pct / 100.0}}


def main() -> None:
    res = results_for({"%d%%" % p: design(p) for p in GRID})
    years = [f[0][:4] for f in FOLDS]
    names = ["%d%%" % p for p in GRID]
    w14 = pd.DataFrame({n: [res[n][y]["w14_comp"] for y in years] for n in names}, index=years)
    ranks = w14.rank(axis=1)                    # higher median 14-day composite -> higher rank
    score = ranks.sum()
    smooth = score.rolling(3, center=True, min_periods=2).mean()
    rows = []
    for n in names:
        by = res[n]
        rows.append({"rotation": n, "rank sum (14d)": score[n], "smoothed": round(smooth[n], 2),
                     "6y return": "%+.0f%%" % ((np.prod([1 + by[y]["ret"] for y in years]) - 1) * 100),
                     "worst yearly DD": "%.0f%%" % (max(by[y]["mdd"] for y in years) * 100),
                     "median yearly composite": round(float(np.median([by[y]["comp"] for y in years])), 2),
                     "2021-22": "%+.0f%%" % (by["2021"]["ret"] * 100)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    best = smooth.idxmax()
    print("\nchosen by the smoothed 14-day rank objective: %s" % best)
    h = results_for({"40%": design(40), best: design(int(best[:-1]))}, HOLDOUT)
    hy = sorted(h["40%"])
    wins = sum(h[best][y]["w14_comp"] > h["40%"][y]["w14_comp"] for y in hy)
    print("holdout median 14-day composite %s: 40%% %s, %s %s -> %s" % (
        "/".join(hy), " ".join("%.2f" % h["40%"][y]["w14_comp"] for y in hy), best,
        " ".join("%.2f" % h[best][y]["w14_comp"] for y in hy),
        "CONFIRMED" if wins == len(hy) else "not confirmed"))


if __name__ == "__main__":
    main()
