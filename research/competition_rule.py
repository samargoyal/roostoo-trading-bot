"""Every design of rounds 20+ judged on the competition's own yardstick: 14-day windows.

The competition scores one 14-day window, but rounds 1-37 chose on yearly composites, with the
14-day figures printed only for information. This applies, to every design in research/rounds.py,
a rule fixed after round 37 (so after seeing those figures, which is why the untouched holdout
decides):

  C1  median 14-day composite higher than the incumbent's in at least 5 of the 6 folds
  C2  share of positive 14-day windows, averaged over the folds, no lower than the incumbent's
  C3  worst yearly drawdown at most 2 points worse
  C4  each neighbour has the higher median 14-day composite in at least 4 of 6 folds
  C5  the median 14-day composite higher in both holdout years (October 2018 - October 2020)

    python -m research.competition_rule
"""
import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.rounds import ROUNDS, results_for

YEARS = [f[0][:4] for f in FOLDS]


def better_14d(by: dict, base: dict, years=YEARS) -> int:
    return sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in years)


def main() -> None:
    designs = {"incumbent": {}}
    neighbours = {}
    for number in sorted(ROUNDS):
        for name, (overrides, nbrs) in ROUNDS[number].items():
            designs[name] = overrides
            neighbours[name] = nbrs
    res = results_for(designs)
    base = res["incumbent"]
    bpos = np.mean([base[y]["w14_pos"] for y in YEARS])
    bworst = max(base[y]["mdd"] for y in YEARS)
    print("Competition rule (14-day windows): incumbent median 14-day composite by fold %s, positive share %.0f%%"
          % (" ".join("%.2f" % base[y]["w14_comp"] for y in YEARS), bpos * 100))
    survivors = []
    for name in designs:
        if name == "incumbent":
            continue
        by = res[name]
        b14 = better_14d(by, base)
        pos = np.mean([by[y]["w14_pos"] for y in YEARS])
        worst = max(by[y]["mdd"] for y in YEARS)
        ok = b14 >= 5 and pos >= bpos and worst <= bworst + 0.02
        if ok or b14 >= 5:
            print("  %-52s 14d better %d/6, positive %.0f%%, worst DD %.0f%% -> %s" % (
                name, b14, pos * 100, worst * 100, "C1-C3 pass" if ok else "fails C2/C3"))
        if ok:
            survivors.append(name)
    print("\n%d of %d designs pass C1-C3" % (len(survivors), len(designs) - 1))
    confirmed = []
    for name in survivors:
        nb = {"%s / neighbour %d" % (name, i + 1): o for i, o in enumerate(neighbours[name])}
        nres = results_for(nb)
        robust = all(better_14d(nres[n], base) >= 4 for n in nb)
        print("  %-52s neighbours 14d better: %s -> %s" % (
            name, ", ".join("%d/6" % better_14d(nres[n], base) for n in nb), "robust" if robust else "breaks"))
        if not robust:
            continue
        hres = results_for({"incumbent": {}, name: designs[name]}, HOLDOUT)
        hy = [f[0][:4] for f in HOLDOUT]
        wins = better_14d(hres[name], hres["incumbent"], hy)
        print("    holdout median 14-day composite %s: incumbent %s, design %s -> %s" % (
            "/".join(hy), " ".join("%.2f" % hres["incumbent"][y]["w14_comp"] for y in hy),
            " ".join("%.2f" % hres[name][y]["w14_comp"] for y in hy),
            "CONFIRMED" if wins == len(hy) else "not confirmed"))
        if wins == len(hy):
            confirmed.append(name)
    print("\nconfirmed better on the competition's yardstick: %s" % (", ".join(confirmed) or "none"))


if __name__ == "__main__":
    main()
