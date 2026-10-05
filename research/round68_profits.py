"""Round 68: securing profits inside the live bot (the 14-day lock-in rule is in h68_lock_in.py).

Earlier tries cut winners too early: always-on trailing stops on the picks (round 20), a brake
on the sleeve (round 23), halving in euphoria (round 28), blow-off exits (round 66). Written down
before running, each on top of the live bot (R54b) and judged as rounds 65-67 were:

  R68b  partial take-profit: once a pick is 40% above its entry, keep half until it leaves the
        picks; parabolic moves often reverse (neighbours: 30%, 60%)
  R68c  profit-activated trailing stop: no stop until a pick is up 25%, then a 15% trailing stop
        from its high, so only large gains are protected (neighbours: 15%/10%, 40%/20%)
  R68d  drawdown-scaled sleeve: full size until the account is 5% below its 30-day high, then in
        proportion down to a quarter at 20% below; it re-risks as the account recovers
        (neighbours: 3%-15%, 8%-25%)

    python -m research.round68_profits
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round66_attention import design
from research.rounds import results_for, verdict

DESIGNS = {
    "R68b partial take-profit at +40%": (dict(rotation_take_profit=0.40),
                                         [dict(rotation_take_profit=0.30), dict(rotation_take_profit=0.60)]),
    "R68c trailing stop after +25%": (dict(rotation_profit_trail_after=0.25, rotation_profit_trail=0.15),
                                      [dict(rotation_profit_trail_after=0.15, rotation_profit_trail=0.10),
                                       dict(rotation_profit_trail_after=0.40, rotation_profit_trail=0.20)]),
    "R68d drawdown-scaled sleeve": (dict(rotation_dd_scale=True),
                                    [dict(rotation_dd_scale=True, rotation_dd_start=0.03, rotation_dd_full=0.15),
                                     dict(rotation_dd_scale=True, rotation_dd_start=0.08, rotation_dd_full=0.25)]),
}


def plain(**opts):
    d = design()
    d["strategy"].update(opts)
    return d


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
        print("%-34s %s | 6y %+.0f%% DD %.0f%% | yearly better %d/6, 14-day better %d/6 | 14d %s%s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100, better, w14,
            " ".join("%.1f" % by[y]["w14_comp"] for y in years),
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
            print("    %-48s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
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
