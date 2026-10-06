"""Round 102: the book's idle cash (the user: about $12.5k of the account sits in USD).

The book keeps a slot for every coin; a downtrending coin it may not short (crowded funding, a
stop's cooldown) leaves its slot in cash. Written down before running, on the live bot (55%
rotation, 45% efficiency book), round 95's return-first rule and C1-C5:

  I1  the book's held positions fill its whole share (the idle slots spread over them)
  I2  the idle share in PAXG while gold's 14-day return is positive (round 60's option;
      neighbours: 7 and 21 days)
  I3  the idle share to the rotation's picks (the rotation's weight raised by the book's idle
      share each hour; approximated here by the rotation at 60% and 65%, the book's idle share
      in the backtests being about 5-10% of the account)

    python -m research.round102_idle_cash
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import YEARS, design
from research.rounds import results_for

BOT = "live (55%, efficiency book)"
DESIGNS = {
    "I1 held positions fill the book": dict(ls_rel_fill=True),
    "I2 idle share in PAXG (14 days)": dict(ls_idle_horizon=336),
    "I2 / 7 days": dict(ls_idle_horizon=168),
    "I2 / 21 days": dict(ls_idle_horizon=504),
    "I3 rotation 60%": dict(rotation_weight=0.60),
    "I3 rotation 65%": dict(rotation_weight=0.65),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"live": design(BOT)}
    names.update({n: design(BOT, **o) for n, o in DESIGNS.items()})
    res, hold = results_for(names), results_for(names, HOLDOUT)
    hy = sorted(hold["live"])
    b, bh = res["live"], hold["live"]
    bdd = max(b[y]["mdd"] for y in YEARS)
    print("Round 102: yearly return by fold %s | total | worst DD | 14d better | won | holdout %s [14d]"
          % (" ".join(YEARS), " ".join(hy)))
    for n, by in res.items():
        total = np.prod([1 + by[y]["ret"] for y in YEARS]) - 1
        dd = max(by[y]["mdd"] for y in YEARS)
        print("  %-34s %s | %+8.0f%% | %3.0f%% | %d/6 | %3.0f%% | %s [%s] | %s" % (
            n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), total * 100, dd * 100,
            sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
            np.mean([by[y]["w14_pos"] for y in YEARS]) * 100,
            " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
            " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy),
            "" if n == "live" else "return better %d/6, holdout %d/2 (14d %d/2), DD %+.0f pts" % (
                sum(by[y]["ret"] > b[y]["ret"] for y in YEARS), sum(hold[n][y]["ret"] > bh[y]["ret"] for y in hy),
                sum(hold[n][y]["w14_comp"] > bh[y]["w14_comp"] for y in hy), (dd - bdd) * 100)), flush=True)


if __name__ == "__main__":
    main()
