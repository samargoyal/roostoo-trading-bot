"""Round 65: Wikipedia attention as a filter on the live bot's rotation (after H62).

H62 found retail attention predictive on its own (A01, A11) but redundant beside the bot, since
coins with rising attention are mostly the momentum leaders the rotation already holds. Written
down before running: use it to choose among the rotation's candidates instead, skipping a coin
whose attention is fading (research/attention.py; views up to two days before the daily
rebalance; coins without an English article are never skipped). Judged against the live bot
(R54b, config/comp.json) with the rule of rounds 31 on: paired gains in at least 5 of 6 folds,
worst drawdown at most 2 points deeper, both neighbours holding in 4 of 6, then both holdout
years.

  R65a  skip a coin whose 7-day views are 30% or more below their 90-day median
        (neighbours: 40% and 20%)
  R65b  rotate only into coins whose attention is at or above normal (neighbours: -10%, +10%)

    python -m research.round65_attention
"""
import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.rounds import results_for, verdict

LIVE = {"book_mode": "long_short", "ls_trend": [240, 960], "short_exclude_external": True, "short_stop_atr": 10.0}


def design(floor=None):
    strategy = dict(LIVE)
    research = {"funding_hours": 72}
    if floor is not None:
        strategy.update(rotation_attention_filter=True, rotation_attention_floor=float(floor))
        research["attention"] = True
    return {"strategy": strategy, "research": research}


DESIGNS = {
    "R65a skip coins whose attention is fading": (np.log(0.7), [np.log(0.6), np.log(0.8)]),
    "R65b only coins with normal or rising attention": (0.0, [-0.1, 0.1]),
}


def main() -> None:
    names = {"live bot (R54b)": design()}
    names.update({n: design(f) for n, (f, _) in DESIGNS.items()})
    res = results_for(names)
    years = [f[0][:4] for f in FOLDS]
    base = res["live bot (R54b)"]
    for name, by in res.items():
        ok, med, better, worst = verdict(by, base, paired=True)
        six = np.prod([1 + by[y]["ret"] for y in years]) - 1
        print("%-48s %s | 6y %+.0f%% DD %.0f%% | comp %s | better %d/6%s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100,
            " ".join("%.2f" % by[y]["comp"] for y in years), better,
            "" if name.startswith("live") else ("  -> candidate" if ok else "  -> fail")))
    for name, (floor, neighbours) in DESIGNS.items():
        ok, _, _, _ = verdict(res[name], base, paired=True)
        if not ok:
            continue
        nres = results_for({"%s / %.2f" % (name, f): design(f) for f in neighbours})
        fine = []
        for n, by in nres.items():
            nok, med, better, worst = verdict(by, base, strict=False, paired=True)
            fine.append(nok)
            print("    neighbour %-40s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
        if not all(fine):
            print("  %s: not robust" % name)
            continue
        h = results_for({"live": design(), name: design(floor)}, HOLDOUT)
        hy = sorted(h["live"])
        wins = sum(h[name][y]["comp"] > h["live"][y]["comp"] for y in hy)
        print("  holdout %s: live %s, design %s -> %s" % (
            "/".join(hy), " ".join("%.2f" % h["live"][y]["comp"] for y in hy),
            " ".join("%.2f" % h[name][y]["comp"] for y in hy), "CONFIRMED" if wins == len(hy) else "not confirmed"))


if __name__ == "__main__":
    main()
