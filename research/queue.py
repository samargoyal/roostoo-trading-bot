"""Shared harness for the research queue (RESEARCH_QUEUE.md): each design is run on top of both
bots, R54b (config/comp_r54b.json, live until 6 October 2026) and the multi-horizon one
(config/comp_multi.json), and judged on the competition rule of research/competition_rule.py
against that bot:

  C1  median 14-day composite higher in at least 5 of the 6 folds
  C2  share of positive 14-day windows, averaged over the folds, no lower
  C3  worst yearly drawdown at most 2 points worse
  C4  each neighbour has the higher median 14-day composite in at least 4 of 6 folds
  C5  the median 14-day composite higher in both holdout years (2018-2020)

Neighbours run only for designs that pass C1-C3, and the holdout only for those that also pass C4.
"""
import json

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.rounds import results_for

YEARS = [f[0][:4] for f in FOLDS]
BOTS = {name: json.load(open(path))["strategy"] for name, path in
        (("multi-horizon", "config/comp_multi.json"), ("live (R54b)", "config/comp_r54b.json"),
         ("live (K2)", "config/comp_k2.json"))}
DEFAULT_BOTS = ["multi-horizon", "live (R54b)"]       # rounds 75-82 ran on these two
BOTS["K2, 3 coins"] = dict(BOTS["live (K2)"], rotation_top=3)      # the user's choice, round 85


def design(bot: str, **opts) -> dict:
    """The bot with `opts` on top; a "research" option adds data tables (research/folds.py)."""
    extra = opts.pop("research", {})
    return {"strategy": dict(BOTS[bot], **opts), "research": dict({"funding_hours": 72}, **extra)}


def better_14d(by, base, years=YEARS):
    return sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in years)


def judge(designs: dict, title: str, bots=None) -> dict:
    """designs: {name: (options, [neighbour options])}. Prints each bot's table (all of BOTS
    unless `bots` names some); returns {bot: {name: "confirmed" | "fails C.."}}."""
    verdicts = {}
    for bot in bots or DEFAULT_BOTS:
        names = {"baseline": design(bot)}
        names.update({n: design(bot, **dict(o)) for n, (o, _) in designs.items()})
        res = results_for(names)
        base = res["baseline"]
        bpos = np.mean([base[y]["w14_pos"] for y in YEARS])
        bworst = max(base[y]["mdd"] for y in YEARS)
        print("\n%s, on the %s bot" % (title, bot))
        print("  %-44s %s | worst DD %.0f%% | 14-day windows won %.0f%%" % (
            "baseline", " ".join("%+6.0f%%" % (base[y]["ret"] * 100) for y in YEARS), bworst * 100, bpos * 100))
        verdicts[bot] = {}
        for name, (opts, neighbours) in designs.items():
            by = res[name]
            b14 = better_14d(by, base)
            pos = np.mean([by[y]["w14_pos"] for y in YEARS])
            worst = max(by[y]["mdd"] for y in YEARS)
            gave = np.mean([by[y]["ret"] - base[y]["ret"] for y in YEARS])
            fails = [c for c, ok in (("C1", b14 >= 5), ("C2", pos >= bpos), ("C3", worst <= bworst + 0.02)) if not ok]
            verdict = "fails " + ",".join(fails) if fails else None
            if verdict is None:
                nres = results_for({"%s / n%d" % (name, i): design(bot, **dict(o)) for i, o in enumerate(neighbours)})
                nb = [better_14d(nres[n], base) for n in nres]
                if not all(x >= 4 for x in nb):
                    verdict = "fails C4 (neighbours %s)" % "/".join("%d/6" % x for x in nb)
                else:
                    h = results_for({"baseline": design(bot), name: design(bot, **dict(opts))}, HOLDOUT)
                    hy = sorted(h["baseline"])
                    wins = better_14d(h[name], h["baseline"], hy)
                    verdict = "CONFIRMED (C1-C5)" if wins == len(hy) else "fails C5 (holdout %d/2)" % wins
            verdicts[bot][name] = verdict
            print("  %-44s %s | worst DD %.0f%% | won %.0f%% | 14d better %d/6 | mean yearly return %+.0f pts | %s" % (
                name, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS), worst * 100, pos * 100, b14,
                gave * 100, verdict), flush=True)
    return verdicts
