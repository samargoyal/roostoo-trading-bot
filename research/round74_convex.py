"""Round 74: convex optimisation on the live bot (the multi-horizon ranking of round 73).

Rounds 5 and 6 tried optimisers on the bot of that time; the user asked for convex optimisation
on the new bot. The long-short book weights its 40-odd positions by inverse volatility, which
ignores how they move together, though crypto moves largely as one; an optimiser over the
positions' side-adjusted returns (a short counts as minus the coin) can use the correlations,
and a short then offsets the longs it hedges. Written down before running, each against the
live bot and judged as rounds 65-73 were:

  R74a the book by equal risk contribution (Spinu's convex problem), covariance of 30 days of
       hourly returns, re-solved daily (neighbours: 14 and 60 days)
  R74b the book at minimum variance, at most 10% a coin (neighbours: 5%, 20%)
  R74c the rotation's two picks by equal risk contribution instead of equally (round 5 and round
       73 found the calmer leader the weaker mover; tested again on the live ranking)
  R74d the split between the rotation and the book: 60%, 65%, 75% and 80% against the 70% live,
       for information (the user chose 70%)

    python -m research.round74_convex
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round68_profits import plain
from research.rounds import results_for, verdict

LIVE = dict(rotation_ranking="multi", rotation_horizons=[168, 336, 504], rotation_horizon_weights=[2.0, 1.0, 1.0])
DESIGNS = {
    "R74a book by equal risk contribution": (dict(ls_weighting="erc", ls_cov_hours=720),
                                             [dict(ls_weighting="erc", ls_cov_hours=336),
                                              dict(ls_weighting="erc", ls_cov_hours=1440)]),
    "R74b book at minimum variance (10% cap)": (dict(ls_weighting="min_variance", ls_max_weight=0.10),
                                                [dict(ls_weighting="min_variance", ls_max_weight=0.05),
                                                 dict(ls_weighting="min_variance", ls_max_weight=0.20)]),
    "R74c picks by equal risk contribution": (dict(rotation_weighting="erc"),
                                              [dict(rotation_weighting="inverse_vol"), dict(rotation_weighting="min_variance")]),
}
SPLITS = {"R74d rotation %d%%" % (w * 100): dict(rotation_weight=w) for w in (0.6, 0.65, 0.75, 0.8)}


def live(**opts):
    return plain(**dict(LIVE, **opts))


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"live bot (multi-horizon)": live()}
    names.update({n: live(**o) for n, (o, _) in DESIGNS.items()})
    names.update({n: live(**o) for n, o in SPLITS.items()})
    res = results_for(names)
    years = [f[0][:4] for f in FOLDS]
    base = res["live bot (multi-horizon)"]
    for name, by in res.items():
        ok, med, better, worst = verdict(by, base, paired=True)
        six = np.prod([1 + by[y]["ret"] for y in years]) - 1
        w14 = sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in years)
        print("%-40s %s | 6y %+.0f%% DD %.0f%% | yearly better %d/6, 14-day better %d/6%s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100, better, w14,
            "" if name.startswith(("live", "R74d")) else ("  -> candidate" if ok else "  -> fail")), flush=True)
    for name, (opts, neighbours) in DESIGNS.items():
        ok, _, _, _ = verdict(res[name], base, paired=True)
        if not ok:
            continue
        nres = results_for({"%s / neighbour %d" % (name, i + 1): live(**o) for i, o in enumerate(neighbours)})
        fine = []
        for n, by in nres.items():
            nok, med, better, worst = verdict(by, base, strict=False, paired=True)
            fine.append(nok)
            print("    %-52s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
        if not all(fine):
            print("  %s: not robust" % name)
            continue
        h = results_for({"live": live(), name: live(**opts)}, HOLDOUT)
        hy = sorted(h["live"])
        wins = sum(h[name][y]["comp"] > h["live"][y]["comp"] for y in hy)
        print("  holdout: live %s, design %s -> %s" % (
            " ".join("%+.0f%%" % (h["live"][y]["ret"] * 100) for y in hy),
            " ".join("%+.0f%%" % (h[name][y]["ret"] * 100) for y in hy), "CONFIRMED" if wins == len(hy) else "not confirmed"))


if __name__ == "__main__":
    main()
