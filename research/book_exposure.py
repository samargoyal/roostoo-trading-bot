"""Investing the defensive book's idle cash: risk_on_exposure from 75% to 100%.

With the 70/30 split the book holds at most 75% of its share when risk-on, which left $7,500 of
the $100,000 in cash on the first live cycle. The user asked to trade it. Rules fixed before
running:

  Grid       risk_on_exposure 75% (current), 80%, 85%, 90%, 95%, 100%; nothing else changes
  Compare    each against 75% on the six folds: yearly return, worst drawdown, yearly composite
             and median 14-day composite (the competition's yardstick)
  Adopt      the highest share that beats 75% on the median 14-day composite in at least 4 of
             the 6 folds with a worst yearly drawdown no more than 2 points deeper, and beats
             it in both untouched holdout years (October 2018 - October 2020)

    python -m research.book_exposure
"""
import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.rounds import results_for

GRID = [75, 80, 85, 90, 95, 100]


def design(pct: int) -> dict:
    return {} if pct == 75 else {"strategy": {"risk_on_exposure": pct / 100.0}}


def table(res, years):
    rows = []
    for p in GRID:
        by, base = res["%d%%" % p], res["75%"]
        rows.append({"book exposure": "%d%%" % p,
                     "6y return" if len(years) > 2 else "2y return":
                         "%+.0f%%" % ((np.prod([1 + by[y]["ret"] for y in years]) - 1) * 100),
                     "worst yearly DD": "%.1f%%" % (max(by[y]["mdd"] for y in years) * 100),
                     "median yearly comp": round(float(np.median([by[y]["comp"] for y in years])), 2),
                     "median 14d comp": round(float(np.median([by[y]["w14_comp"] for y in years])), 3),
                     "14d better than 75%": "%d/%d" % (sum(by[y]["w14_comp"] > base[y]["w14_comp"]
                                                           for y in years), len(years)),
                     "returns by year": " ".join("%+.0f%%" % (by[y]["ret"] * 100) for y in years)})
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 250)
    names = {"%d%%" % p: design(p) for p in GRID}
    res = results_for(names)
    years = [f[0][:4] for f in FOLDS]
    print("six folds, October 2020 - October 2026 (%s)" % "/".join(years))
    print(table(res, years).to_string(index=False))
    base_dd = max(res["75%"][y]["mdd"] for y in years)
    passed = [p for p in GRID[1:]
              if sum(res["%d%%" % p][y]["w14_comp"] > res["75%"][y]["w14_comp"] for y in years) >= 4
              and max(res["%d%%" % p][y]["mdd"] for y in years) <= base_dd + 0.02]
    h = results_for(names, HOLDOUT)
    hy = sorted(h["75%"])
    print("\nholdout, October 2018 - October 2020 (%s)" % "/".join(hy))
    print(table(h, hy).to_string(index=False))
    confirmed = [p for p in passed if all(h["%d%%" % p][y]["w14_comp"] > h["75%"][y]["w14_comp"] for y in hy)]
    print("\npass the six folds: %s; confirmed on the holdout: %s" % (
        ", ".join("%d%%" % p for p in passed) or "none", ", ".join("%d%%" % p for p in confirmed) or "none"))
    print("adopt: %s" % ("%d%%" % max(confirmed) if confirmed else "keep 75%"))


if __name__ == "__main__":
    main()
