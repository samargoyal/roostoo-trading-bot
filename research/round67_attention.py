"""Round 67: round 66's attention-share filter, centred on its stronger neighbour.

Round 66's share filter (R66b: skip a coin whose share of all crypto attention is below its
norm) beat the live bot in 5 of 6 folds, but its neighbour at +0.1 broke, while the one at -0.1
beat it in all six. Centring the design on -0.1 was a choice made after seeing the folds, so
written down before running: the design is judged by the usual rule, its neighbours (-0.2 and
0.0) must hold in 4 of 6, and the untouched holdout decides: it must win both years.

    python -m research.round67_attention
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round66_attention import design
from research.rounds import results_for, verdict


def share(floor):
    return design(rotation_attention_filter=True, rotation_attention_key="SHR", rotation_attention_floor=floor)


def main():
    warnings.filterwarnings("ignore")
    res = results_for({"live": design(), "R67 floor -0.10": share(-0.1), "neighbour -0.20": share(-0.2),
                       "neighbour 0.00": share(0.0)})
    years = [f[0][:4] for f in FOLDS]
    base = res["live"]
    for name, by in res.items():
        strict = name.startswith("R67")
        ok, med, better, worst = verdict(by, base, strict=strict, paired=True)
        six = np.prod([1 + by[y]["ret"] for y in years]) - 1
        print("%-16s %s | 6y %+.0f%% DD %.0f%% | comp %s | better %d/6 %s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100,
            " ".join("%.2f" % by[y]["comp"] for y in years), better,
            "" if name == "live" else ("PASS" if ok else "FAIL") if strict else ("holds" if ok else "breaks")))
    h = results_for({"live": design(), "R67": share(-0.1)}, HOLDOUT)
    for y in sorted(h["live"]):
        print("holdout %s: live %+.0f%% c%.2f DD %.0f%% | R67 %+.0f%% c%.2f DD %.0f%%" % (
            y, h["live"][y]["ret"] * 100, h["live"][y]["comp"], h["live"][y]["mdd"] * 100,
            h["R67"][y]["ret"] * 100, h["R67"][y]["comp"], h["R67"][y]["mdd"] * 100))


if __name__ == "__main__":
    main()
