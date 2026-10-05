"""Round 69: leaving a coin when its trend shows exhaustion (step 2 of research/h69_trend_exit.py).

The user's idea for securing profits: exit when most signals say a trend is over or turning
choppy. Step 1 found that a vote of many warnings does not predict the next days (most of them
fire on ordinary pullbacks inside strong trends), nor do gradient-boosted trees (AUC about 0.53);
two warnings did in all six folds, both signs of a tiring advance rather than a dip: weak highs
(near the 7-day high with RSI(14) below 60) and volume divergence (up over 3 days on less than
70% of the previous 3 days' volume). Warned coins still rose on average, so the rule replaces
rather than sells: at the daily re-pick a warned candidate is skipped and the next healthy one
takes its slot. Written down before running, each on top of the live bot (R54b) and judged as
rounds 65-68 were:

  R69a  exhaustion swap: either warning (neighbours: 2% / RSI 55 / 60%, 5% / RSI 65 / 80%)
  R69b  volume divergence only (neighbours: 60%, 80%), for information
  R69c  weak highs only (neighbours: 2% / RSI 55, 5% / RSI 65), for information

    python -m research.round69_exhaustion
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round66_attention import design
from research.round68_profits import plain
from research.rounds import results_for, verdict

ON = dict(rotation_exhaustion=True)
DESIGNS = {
    "R69a exhaustion swap": (dict(ON), [dict(ON, exhaustion_near_high=0.98, exhaustion_rsi=55.0, exhaustion_volume=0.6),
                                        dict(ON, exhaustion_near_high=0.95, exhaustion_rsi=65.0, exhaustion_volume=0.8)]),
    "R69b volume divergence only": (dict(ON, exhaustion_rsi=0.0),
                                    [dict(ON, exhaustion_rsi=0.0, exhaustion_volume=0.6),
                                     dict(ON, exhaustion_rsi=0.0, exhaustion_volume=0.8)]),
    "R69c weak highs only": (dict(ON, exhaustion_volume=0.0),
                             [dict(ON, exhaustion_volume=0.0, exhaustion_near_high=0.98, exhaustion_rsi=55.0),
                              dict(ON, exhaustion_volume=0.0, exhaustion_near_high=0.95, exhaustion_rsi=65.0)]),
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
        print("%-30s %s | 6y %+.0f%% DD %.0f%% | yearly better %d/6, 14-day better %d/6%s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100, better, w14,
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
            print("    %-44s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
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
