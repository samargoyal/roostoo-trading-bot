"""Round 97: stress tests for the efficiency-ratio book at 30 days (round 95's best: on the book
alone +6,666% against +2,794% over six years, worst drawdown 31% against 43%, both holdout years
better). Written down before running, on the book alone: closer neighbours (21 and 45 days), the
cap on a coin's weight at 2x and 5x instead of 3x, and double fees (against the book at double
fees). A robust result keeps more six-year return and a lower drawdown than the book in all of
them, and wins both holdout years in most.

    python -m research.round97_efficiency_stress
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

BOT = "the live book alone"
FEES2 = {"taker_fee": 0.002}


def with_fees(d):
    return dict(d, backtest=FEES2)


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {
        "baseline": design(BOT),
        "E1 30 days (cap 3x)": design(BOT, ls_er_hours=720),
        "E1 21 days": design(BOT, ls_er_hours=504),
        "E1 45 days": design(BOT, ls_er_hours=1080),
        "E1 30 days, cap 2x": design(BOT, ls_er_hours=720, ls_sizing_cap=2.0),
        "E1 30 days, cap 5x": design(BOT, ls_er_hours=720, ls_sizing_cap=5.0),
        "baseline, double fees": with_fees(design(BOT)),
        "E1 30 days, double fees": with_fees(design(BOT, ls_er_hours=720)),
    }
    res, hold = results_for(names), results_for(names, HOLDOUT)
    hy = sorted(hold["baseline"])
    print("Round 97 (book alone): yearly return by fold %s | total | worst DD | 14d better | won | holdout %s"
          % (" ".join(YEARS), " ".join(hy)))
    for n, by in res.items():
        base = res["baseline, double fees"] if "double" in n else res["baseline"]
        total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
        b14 = sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in YEARS)
        print("  %-26s %s | %+8.0f%% | %3.0f%% | %d/6 | %3.0f%% | %s" % (
            n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100,
            max(by[y]["mdd"] for y in YEARS) * 100, b14, np.mean([by[y]["w14_pos"] for y in YEARS]) * 100,
            " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy)), flush=True)


if __name__ == "__main__":
    main()
