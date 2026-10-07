"""Round 104: long the winners, short the losers, re-ranked every hour (the user: "there must
be a way to long the winners and short the losers, dynamic, changing the winners and losers").

The book replaced by a cross-sectional long-short on the rotation's own ranking (7-, 14- and
21-day returns, 2/2/1), filling the book's share, no short where funding is negative. Written
down before running, on the book alone and the live bot (55% rotation), reported on return,
drawdown, the 14-day yardstick and the holdout:

  XS1 long the best 5, short the worst 5, dollar neutral (neighbours: 3 and 8 a side)
  XS2 long the best 5; short the worst 5 only while the rotation's BTC filter is off
  XS3 long the best 5, short the worst 5 only if each is in its own downtrend, dollar neutral

    python -m research.round104_winners_losers
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import BOTS, YEARS, design
from research.rounds import results_for

BOTS["the efficiency book alone"] = dict(BOTS["live (55%, efficiency book)"], rotation_weight=0.0)
FILL = dict(ls_rel_fill=True)
DESIGNS = {
    "XS1 best 5 / worst 5, neutral": dict(FILL, ls_xs_n=5, ls_neutral="scale"),
    "XS1 / 3 a side": dict(FILL, ls_xs_n=3, ls_neutral="scale"),
    "XS1 / 8 a side": dict(FILL, ls_xs_n=8, ls_neutral="scale"),
    "XS2 best 5, worst 5 shorted in bears": dict(FILL, ls_xs_n=5, ls_xs_short="bear"),
    "XS3 best 5, worst 5 if in downtrends": dict(FILL, ls_xs_n=5, ls_xs_short="trend", ls_neutral="scale"),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    for bot in ("the efficiency book alone", "live (55%, efficiency book)"):
        names = {"live": design(bot)}
        names.update({n: design(bot, **o) for n, o in DESIGNS.items()})
        res, hold = results_for(names), results_for(names, HOLDOUT)
        hy = sorted(hold["live"])
        b = res["live"]
        print("\nRound 104 on %s: yearly return by fold %s | total | worst DD | 14d better | holdout %s [14d]"
              % (bot, " ".join(YEARS), " ".join(hy)))
        for n, by in res.items():
            print("  %-40s %s | %+8.0f%% | %3.0f%% | %d/6 | %s [%s]" % (
                n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS),
                (np.prod([1 + by[y]["ret"] for y in YEARS]) - 1) * 100, max(by[y]["mdd"] for y in YEARS) * 100,
                sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
                " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
                " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy)), flush=True)


if __name__ == "__main__":
    main()
