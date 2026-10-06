"""D2: the deflated Sharpe ratio of the live bot and the previous one (RESEARCH_QUEUE.md part D).

Bailey and Lopez de Prado (2014): with N designs tried, the best of N unskilled ones reaches a
Sharpe ratio of about SR0 = sqrt(V) x ((1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e))) by luck alone
(V the variance of the tried designs' Sharpe ratios, g Euler's constant, Z the normal quantile).
The deflated Sharpe ratio is the probability that the chosen design's Sharpe ratio beats SR0,
given its skewness and kurtosis and the number of days:

    DSR = Z^-1[ (SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1) / 4 SR^2) ]

on daily returns, per fold. N counts every design and neighbour defined in research/rounds.py
(rounds 20-64) and the round scripts since (65-77); V is estimated per fold from a random
sample of 40 of them, re-run here (the fold cache did not keep Sharpe ratios).

    python -m research.d2_deflated_sharpe
"""
import copy
import importlib
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from statistics import NormalDist

import numpy as np

from research.folds import FOLDS, run

OUT = os.path.join("runs", "research", "d2")
R54B = {"book_mode": "long_short", "ls_trend": [240, 960], "short_stop_atr": 10.0, "short_exclude_external": True}
SCRIPTS = ["round65_attention", "round66_attention", "round68_profits", "round69_exhaustion", "round70_btc_dip",
           "round71_entries_exits", "round72_book_entries_exits", "round74_convex", "round75_risk_breakers",
           "round76_sessions", "round77_signals"]


def designs():
    """Every design and neighbour tried, as fold-runner overrides."""
    from research.rounds import ROUNDS
    out = []
    for number, spec in ROUNDS.items():
        for name, (overrides, neighbours) in spec.items():
            out += [overrides] + list(neighbours)
    for module in SCRIPTS:
        spec = getattr(importlib.import_module("research." + module), "DESIGNS", {})
        for name, (opts, neighbours) in spec.items():
            for o in [opts] + list(neighbours):
                if not isinstance(o, dict):              # round 65: an attention floor
                    o = {"rotation_attention_filter": True, "rotation_attention_floor": float(o),
                         "research": {"attention": True}}
                o = dict(o)
                extra = o.pop("research", {})
                out.append({"strategy": dict(R54B, **o), "research": dict({"funding_hours": 72}, **extra)})
    return out


def sharpe_job(args):
    i, overrides, fold = args
    path = os.path.join(OUT, "trial%d_%s.json" % (i, fold[0]))
    if os.path.exists(path):
        return json.load(open(path))
    import research.folds as folds
    from bot import backtest
    captured = {}
    original = backtest.run_backtest

    def capture(*a, **k):
        result = original(*a, **k)
        captured["sharpe"] = result.stats["sharpe"]
        return result
    folds.run_backtest = capture
    run(("trial", copy.deepcopy(overrides), fold))
    out = {"i": i, "fold": fold[0], "sharpe": captured["sharpe"]}
    json.dump(out, open(path, "w"))
    return out


def live_job(args):
    """The live design's daily returns for one fold."""
    name, path_cfg, fold = args
    path = os.path.join(OUT, "live_%s_%s.json" % (name, fold[0]))
    if os.path.exists(path):
        return json.load(open(path))
    import research.folds as folds
    from bot import backtest
    captured = {}
    original = backtest.run_backtest

    def capture(*a, **k):
        result = original(*a, **k)
        captured["curve"] = result.curve
        return result
    folds.run_backtest = capture
    run(("live", {"strategy": json.load(open(path_cfg))["strategy"], "research": {"funding_hours": 72}}, fold))
    eq = np.array([v for _, v in captured["curve"]])
    daily = eq[::24]
    out = {"name": name, "fold": fold[0], "returns": list(daily[1:] / daily[:-1] - 1)}
    json.dump(out, open(path, "w"))
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tried = designs()
    n = len(tried)
    sample = random.Random(0).sample(range(n), 40)
    with ProcessPoolExecutor(max_workers=12) as pool:
        trials = list(pool.map(sharpe_job, [(i, tried[i], f) for i in sample for f in FOLDS]))
        lives = list(pool.map(live_job, [(name, p, f) for name, p in (("multi", "config/comp_multi.json"),
                                                                     ("r54b", "config/comp_r54b.json"))
                                         for f in FOLDS]))
    z, g = NormalDist().inv_cdf, 0.5772156649
    print("designs tried (N): %d; Sharpe variance estimated from %d of them, re-run" % (n, len(sample)))
    for name in ("multi", "r54b"):
        print("\n%s bot" % ("multi-horizon (config/comp_multi.json)" if name == "multi" else "live (R54b)"))
        for start, _ in FOLDS:
            sr_trials = np.array([t["sharpe"] for t in trials if t["fold"] == start]) / math.sqrt(365)
            v = sr_trials.var(ddof=1)
            sr0 = math.sqrt(v) * ((1 - g) * z(1 - 1 / n) + g * z(1 - 1 / (n * math.e)))
            r = np.array(next(x for x in lives if x["name"] == name and x["fold"] == start)["returns"])
            sr = r.mean() / r.std(ddof=1)
            skew = ((r - r.mean()) ** 3).mean() / r.std() ** 3
            kurt = ((r - r.mean()) ** 4).mean() / r.std() ** 4
            t = len(r)
            dsr = NormalDist().cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
            print("  %s: Sharpe %.2f (annualised), luck's best of N %.2f, deflated Sharpe ratio %.2f" % (
                start[:4], sr * math.sqrt(365), sr0 * math.sqrt(365), dsr))


if __name__ == "__main__":
    main()
