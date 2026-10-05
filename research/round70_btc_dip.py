"""Round 70: a smaller rotation while the market's short-term trend points down.

Following round 69, "exit when the trend is gone" tested on the market rather than on each coin
(research/h69_trend_exit.py, last part: the rotation's next 24 hours on its 1,166 invested days
of 2020-26). A vote of nine market warnings ranked the rotation's next day well
(0-1 warnings +2.45%, 6 or more +0.26%), but even the worst bucket was positive, so gates on
the vote lost (2-3 of 6 folds better). One warning stood out: while BTC's 3-day return was
negative the rotation made +0.33% a day against +1.52%, lower in all six folds, and halving
the sleeve then raised its Sharpe ratio in all six (long-only momentum carries the market's
falls; Liu and Tsyvinski 2021 find crypto's time-series momentum strongest at short horizons).
Written down before running, each on top of the live bot (R54b) and judged as rounds 65-69 were:

  R70a  BTC dip halving: while BTC's return over the 72 hours to the last 00:00 UTC close is
        below zero, the rotation's picks hold half their weight, the rest in cash; the size
        changes only at the daily close (neighbours: 48 and 120 hours)
  R70b  BTC dip exit: the same, out of the picks entirely, the user's "exit" (neighbours: 48
        and 120 hours)

    python -m research.round70_btc_dip
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round66_attention import design
from research.round68_profits import plain
from research.rounds import results_for, verdict

DESIGNS = {
    "R70a BTC dip halving": (dict(rotation_btc_dip_hours=72),
                             [dict(rotation_btc_dip_hours=48), dict(rotation_btc_dip_hours=120)]),
    "R70b BTC dip exit": (dict(rotation_btc_dip_hours=72, rotation_btc_dip_share=0.0),
                          [dict(rotation_btc_dip_hours=48, rotation_btc_dip_share=0.0),
                           dict(rotation_btc_dip_hours=120, rotation_btc_dip_share=0.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"live bot (R54b)": design()}
    names.update({n: plain(**o) for n, (o, _) in DESIGNS.items()})
    res = results_for(names)
    years = [f[0][:4] for f in FOLDS]
    base = res["live bot (R54b)"]
    for name, by in res.items():
        ok, med, better, worst = verdict(by, base, paired=True)
        six = np.prod([1 + by[y]["ret"] for y in years]) - 1
        w14 = sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in years)
        print("%-24s %s | 6y %+.0f%% DD %.0f%% | yearly better %d/6, 14-day better %d/6 | comp %s | 14d %s%s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100, better, w14,
            " ".join("%.2f" % by[y]["comp"] for y in years), " ".join("%.2f" % by[y]["w14_comp"] for y in years),
            "" if name.startswith("live") else ("  -> candidate" if ok else "  -> fail")), flush=True)
    for name, (opts, neighbours) in DESIGNS.items():
        ok, _, _, _ = verdict(res[name], base, paired=True)
        if not ok:
            continue
        nres = results_for({"%s / neighbour %d" % (name, i + 1): plain(**o) for i, o in enumerate(neighbours)})
        fine = []
        for n, by in nres.items():
            nok, med, better, worst = verdict(by, base, strict=False, paired=True)
            fine.append(nok)
            print("    %-40s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
        if not all(fine):
            print("  %s: not robust" % name)
            continue
        h = results_for({"live": design(), name: plain(**opts)}, HOLDOUT)
        hy = sorted(h["live"])
        wins = sum(h[name][y]["comp"] > h["live"][y]["comp"] for y in hy)
        print("  holdout: live %s, design %s -> %s" % (
            " ".join("%+.0f%%" % (h["live"][y]["ret"] * 100) for y in hy),
            " ".join("%+.0f%%" % (h[name][y]["ret"] * 100) for y in hy), "CONFIRMED" if wins == len(hy) else "not confirmed"))


if __name__ == "__main__":
    main()
