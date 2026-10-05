"""Round 66: attention (Wikipedia page views) inside the live bot, after round 65's near miss.

Round 65's hard filter (only coins with normal or rising attention) beat the live bot in 4 of
6 folds: it helped in 2023-26 but dropped strong quiet coins in 2020-21. Written down before
running, each on top of the live bot (R54b) and judged as round 65 was (paired gains in 5 of 6
folds, worst drawdown at most 2 points deeper, neighbours in 4 of 6, then both holdout years):

  R66a  soft ranking: candidates ranked by the normal score of their 14-day return plus half
        that of their attention (Hou, Peng and Xiong 2009); a quiet coin can still win
        (neighbours: a quarter, one)
  R66b  attention share: round 65's filter on the coin's share of all crypto attention, since
        absolute views rise for every coin in a bull market (neighbours: floors -0.1, +0.1)
  R66c  attention collapse: a held pick leaves when its attention falls 40% below normal,
        retail having left (neighbours: 50%, 30%)
  R66d  blow-off: a held pick leaves after a day of 5x its usual views (Barber and Odean 2008)
        (neighbours: 4x, 7x)
  R66e  retail gate: the rotation is in the market only while attention to crypto as a whole
        is no more than 20% below normal (neighbours: 30%, 10%)
  R66f  long-short: the long-short book does not short a coin whose attention is above
        normal: shorting into retail hype invites a squeeze (neighbours: -0.15, +0.15)

    python -m research.round66_attention
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round65_attention import LIVE
from research.rounds import results_for, verdict


def design(**opts):
    strategy = dict(LIVE)
    research = {"funding_hours": 72}
    if opts:
        strategy.update(opts)
        research["attention"] = True
    return {"strategy": strategy, "research": research}


DESIGNS = {
    "R66a soft attention ranking": (dict(rotation_attention_rank=0.5),
                                    [dict(rotation_attention_rank=0.25), dict(rotation_attention_rank=1.0)]),
    "R66b attention share filter": (dict(rotation_attention_filter=True, rotation_attention_key="SHR", rotation_attention_floor=0.0),
                                    [dict(rotation_attention_filter=True, rotation_attention_key="SHR", rotation_attention_floor=-0.1),
                                     dict(rotation_attention_filter=True, rotation_attention_key="SHR", rotation_attention_floor=0.1)]),
    "R66c attention-collapse exit": (dict(rotation_exit_attention=True, rotation_exit_attention_floor=float(np.log(0.6))),
                                     [dict(rotation_exit_attention=True, rotation_exit_attention_floor=float(np.log(0.5))),
                                      dict(rotation_exit_attention=True, rotation_exit_attention_floor=float(np.log(0.7)))]),
    "R66d blow-off exit": (dict(rotation_exit_spike=5.0), [dict(rotation_exit_spike=4.0), dict(rotation_exit_spike=7.0)]),
    "R66e retail gate": (dict(rotation_market_attention=True, rotation_market_attention_floor=float(np.log(0.8))),
                         [dict(rotation_market_attention=True, rotation_market_attention_floor=float(np.log(0.7))),
                          dict(rotation_market_attention=True, rotation_market_attention_floor=float(np.log(0.9)))]),
    "R66f no shorts into rising attention": (dict(short_attention_max=0.0),
                                             [dict(short_attention_max=-0.15), dict(short_attention_max=0.15)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"live bot (R54b)": design()}
    names.update({n: design(**o) for n, (o, _) in DESIGNS.items()})
    res = results_for(names)
    years = [f[0][:4] for f in FOLDS]
    base = res["live bot (R54b)"]
    for name, by in res.items():
        ok, med, better, worst = verdict(by, base, paired=True)
        six = np.prod([1 + by[y]["ret"] for y in years]) - 1
        print("%-38s %s | 6y %+.0f%% DD %.0f%% | better %d/6%s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100, better,
            "" if name.startswith("live") else ("  -> candidate" if ok else "  -> fail")), flush=True)
    for name, (opts, neighbours) in DESIGNS.items():
        ok, _, _, _ = verdict(res[name], base, paired=True)
        if not ok:
            continue
        nres = results_for({"%s / neighbour %d" % (name, i + 1): design(**o) for i, o in enumerate(neighbours)})
        fine = []
        for n, by in nres.items():
            nok, med, better, worst = verdict(by, base, strict=False, paired=True)
            fine.append(nok)
            print("    %-48s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
        if not all(fine):
            print("  %s: not robust" % name)
            continue
        h = results_for({"live": design(), name: design(**opts)}, HOLDOUT)
        hy = sorted(h["live"])
        wins = sum(h[name][y]["comp"] > h["live"][y]["comp"] for y in hy)
        print("  holdout %s: live %s, design %s -> %s" % (
            "/".join(hy), " ".join("%+.0f%% c%.2f" % (h["live"][y]["ret"] * 100, h["live"][y]["comp"]) for y in hy),
            " ".join("%+.0f%% c%.2f" % (h[name][y]["ret"] * 100, h[name][y]["comp"]) for y in hy),
            "CONFIRMED" if wins == len(hy) else "not confirmed"))


if __name__ == "__main__":
    main()
