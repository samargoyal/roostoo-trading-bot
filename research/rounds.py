"""Rounds 20 and on: hypotheses aimed at what the first 19 rounds found, under a stricter rule.

Rounds 1-19 tried about 50 designs. A design with no real edge beats the incumbent in at least
4 of 6 folds with a higher median about one time in five, so from round 20 the rule is:

  1  a higher median composite than the incumbent over the six folds
  2  a higher composite in at least 5 of the 6 folds
  3  a worst-fold drawdown at most 2 points worse
  4  robustness: each of the design's nearest neighbours (its setting nudged both ways)
     passes the old rule (higher median, at least 4 of 6)

Each round is fixed in ROUNDS before it runs, in the light of the rounds before it. Results of
each (design, fold) are cached (the incumbent is the same in every round while the defaults
are unchanged; bump VERSION if they change).

    python -m research.rounds 20
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from research.folds import FOLDS, run

VERSION = "defaults-2026-10-03"
CACHE = os.path.join("runs", "research", "fold_cache")
WORKERS = int(os.environ.get("FOLD_WORKERS", "12"))

# name -> (overrides, [neighbour overrides])
ROUNDS = {
    # The rotation has no stop: a pick is held until the daily rebalance or the BTC filter turns,
    # and the worst losses came from tops and the 2021-22 crash. A trailing stop per pick, its
    # slot left in cash until the next rebalance and the coin barred from it for 24 hours.
    20: {
        "R20a rotation picks: trailing stop 8 ATR": ({"strategy": {"rotation_stop_atr": 8.0}},
                                                     [{"strategy": {"rotation_stop_atr": 6.0}},
                                                      {"strategy": {"rotation_stop_atr": 10.0}}]),
        "R20b rotation picks: trailing stop 5 ATR": ({"strategy": {"rotation_stop_atr": 5.0}},
                                                     [{"strategy": {"rotation_stop_atr": 4.0}},
                                                      {"strategy": {"rotation_stop_atr": 6.0}}]),
        "R20c rotation picks: stop 15% below the high": ({"strategy": {"rotation_stop_pct": 0.15}},
                                                         [{"strategy": {"rotation_stop_pct": 0.10}},
                                                          {"strategy": {"rotation_stop_pct": 0.20}}]),
    },
    # Round 20: stops fire on the picks' ordinary swings and cut winners; the daily rebalance is
    # the better exit. So improve what is picked. With 20 coins the bot lagged in BTC-led
    # rallies; demand that a pick beat BTC, and average horizons so no lucky lookback decides.
    21: {
        "R21a picks must beat BTC, else hold BTC": (
            {"strategy": {"rotation_vs_btc": "btc"}},
            [{"strategy": {"rotation_vs_btc": "btc", "rotation_btc_margin": -0.05}},
             {"strategy": {"rotation_vs_btc": "btc", "rotation_btc_margin": 0.05}}]),
        "R21b picks must beat BTC, else PAXG or cash": (
            {"strategy": {"rotation_vs_btc": "cash"}},
            [{"strategy": {"rotation_vs_btc": "cash", "rotation_btc_margin": -0.05}},
             {"strategy": {"rotation_vs_btc": "cash", "rotation_btc_margin": 0.05}}]),
        "R21c ranked by 1-, 2- and 3-week momentum together": (
            {"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504]}},
            [{"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336]}},
             {"strategy": {"rotation_ranking": "multi", "rotation_horizons": [336, 504, 720]}}]),
    },
    # Round 21: multi-horizon ranking was better in 5 of 6 folds but its median fell (2024-25), a
    # near miss; demanding that picks beat BTC changed little. The rotation re-picks daily, so
    # small rank changes swap coins and pay fees: let winners stay.
    22: {
        "R22a keep a pick while it ranks in the top 4": (
            {"strategy": {"rotation_buffer": 4}},
            [{"strategy": {"rotation_buffer": 3}}, {"strategy": {"rotation_buffer": 5}}]),
        "R22b keep a pick at least 3 days": (
            {"strategy": {"rotation_min_hold_hours": 72}},
            [{"strategy": {"rotation_min_hold_hours": 48}}, {"strategy": {"rotation_min_hold_hours": 96}}]),
    },
    # Round 22: letting winners stay cut drawdowns and lifted the 14-day composite, but lost in
    # 2023-25 (4/6). The losses that remain are crash years; nothing yet acts on the account's
    # own drawdown at the rotation level.
    23: {
        "R23a halve the rotation 15% below the account's peak": (
            {"strategy": {"rotation_brake_drawdown": 0.15, "rotation_brake_release": 0.075}},
            [{"strategy": {"rotation_brake_drawdown": 0.10, "rotation_brake_release": 0.05}},
             {"strategy": {"rotation_brake_drawdown": 0.20, "rotation_brake_release": 0.10}}]),
        "R23b rotation only above the account's 30-day average": (
            {"strategy": {"rotation_equity_ma_hours": 720}},
            [{"strategy": {"rotation_equity_ma_hours": 336}}, {"strategy": {"rotation_equity_ma_hours": 1440}}]),
    },
    # Round 23: account-level controls cut drawdowns but cost too much return. The regime is read
    # from BTC alone although the bot trades 45 coins: use market breadth (the share of coins
    # whose 50-hour EMA is above their 200-hour).
    24: {
        "R24a rotation needs BTC's filter and breadth > 50%": (
            {"strategy": {"rotation_breadth": 0.5}},
            [{"strategy": {"rotation_breadth": 0.4}}, {"strategy": {"rotation_breadth": 0.6}}]),
        "R24b rotation filter: breadth > 50% instead of BTC": (
            {"strategy": {"rotation_breadth": 0.5, "rotation_breadth_mode": "only"}},
            [{"strategy": {"rotation_breadth": 0.4, "rotation_breadth_mode": "only"}},
             {"strategy": {"rotation_breadth": 0.6, "rotation_breadth_mode": "only"}}]),
        "R24c book risk-on when breadth > 50% instead of BTC": (
            {"strategy": {"regime_breadth": 0.5}},
            [{"strategy": {"regime_breadth": 0.4}}, {"strategy": {"regime_breadth": 0.6}}]),
    },
    # Round 24: breadth was a worse regime than BTC; BTC leads. What is left of the crash-year
    # losses is the filter's lag: it is slow to leave and quick to whipsaw back in.
    25: {
        "R25a rotation also out below BTC's 200-hour average": (
            {"strategy": {"rotation_exit_sma": 200}},
            [{"strategy": {"rotation_exit_sma": 100}}, {"strategy": {"rotation_exit_sma": 400}}]),
        "R25b rotation re-enters only after 24h of the filter on": (
            {"strategy": {"rotation_reentry_hours": 24}},
            [{"strategy": {"rotation_reentry_hours": 12}}, {"strategy": {"rotation_reentry_hours": 48}}]),
    },
    # Round 25: a faster exit whipsaws; the slow, symmetric filter stays. The two near misses,
    # multi-horizon ranking (R21c) and letting winners stay (R22a, R22b), attack one weakness, a
    # ranking that hangs on one lucky lookback and swaps daily. Together, chosen after seeing
    # both, hence the strict rule and the neighbours.
    26: {
        "R26a multi-horizon ranking + keep picks in the top 4": (
            {"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504], "rotation_buffer": 4}},
            [{"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504], "rotation_buffer": 3}},
             {"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504], "rotation_buffer": 5}}]),
        "R26b multi-horizon ranking + keep picks 3 days": (
            {"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504],
                          "rotation_min_hold_hours": 72}},
            [{"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504],
                           "rotation_min_hold_hours": 48}},
             {"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504],
                           "rotation_min_hold_hours": 96}}]),
    },
    # Round 26: the near misses do not stack; every robustness variant loses 2024-25, a year the
    # rotation was developed on. Remaining untried mechanisms, one per round:
    # 27: concentrate when one coin leads by far (top 1 always was worse in round 9).
    27: {
        "R27 one pick takes the sleeve when it doubles the second": (
            {"strategy": {"rotation_concentrate": 2.0}},
            [{"strategy": {"rotation_concentrate": 1.5}}, {"strategy": {"rotation_concentrate": 3.0}}]),
    },
    # 28: momentum crashes follow euphoria: halve the sleeve when BTC itself is up a lot.
    28: {
        "R28 halve the rotation when BTC is up 25% in 2 weeks": (
            {"strategy": {"rotation_euphoria": 0.25}},
            [{"strategy": {"rotation_euphoria": 0.20}}, {"strategy": {"rotation_euphoria": 0.35}}]),
    },
    # 29: avoid bounces inside downtrends: a pick must be in its own uptrend.
    29: {
        "R29 picks must be in their own uptrend": (
            {"strategy": {"rotation_pick_trend": "both"}},
            [{"strategy": {"rotation_pick_trend": "close"}}, {"strategy": {"rotation_pick_trend": "ema"}}]),
    },
    # 30: more capital at work when the market is risk-on: the book fully invested in 10 coins.
    30: {
        "R30 book risk-on: 100% in up to 10 coins": (
            {"strategy": {"risk_on_exposure": 1.0, "max_positions_risk_on": 10}},
            [{"strategy": {"risk_on_exposure": 0.9, "max_positions_risk_on": 8}},
             {"strategy": {"risk_on_exposure": 1.0, "max_positions_risk_on": 8}}]),
    },
}


def key(overrides: dict, fold) -> str:
    text = json.dumps([VERSION, overrides, fold], sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()


def results_for(designs: dict) -> dict:
    """{name: {year: stats}} for every (name, overrides), computed once and cached."""
    os.makedirs(CACHE, exist_ok=True)
    todo, out = [], {}
    for name, overrides in designs.items():
        for fold in FOLDS:
            path = os.path.join(CACHE, key(overrides, fold) + ".json")
            if os.path.exists(path):
                with open(path) as f:
                    out.setdefault(name, {})[fold[0][:4]] = json.load(f)
            else:
                todo.append((name, overrides, fold))
    if todo:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for (name, overrides, fold), (_, start, r) in zip(todo, pool.map(run, todo)):
                with open(os.path.join(CACHE, key(overrides, fold) + ".json"), "w") as f:
                    json.dump(r, f)
                out.setdefault(name, {})[start[:4]] = r
    return out


def verdict(by_year: dict, base: dict, strict: bool = True):
    years = [f[0][:4] for f in FOLDS]
    comps = [by_year[y]["comp"] for y in years]
    bcomps = [base[y]["comp"] for y in years]
    better = sum(c > b for c, b in zip(comps, bcomps))
    worst = max(by_year[y]["mdd"] for y in years)
    bworst = max(base[y]["mdd"] for y in years)
    median_ok = np.median(comps) > np.median(bcomps)
    if strict:
        ok = median_ok and better >= 5 and worst <= bworst + 0.02
    else:
        ok = median_ok and better >= 4
    return ok, float(np.median(comps)), better, worst


def main() -> None:
    number = int(sys.argv[1])
    spec = ROUNDS[number]
    designs = {"incumbent": {}}
    designs.update({name: o for name, (o, _) in spec.items()})
    res = results_for(designs)
    base = res["incumbent"]
    years = [f[0][:4] for f in FOLDS]
    print("Round %d: composite per fold (from October 2020), strict rule: median up, 5/6 better, "
          "drawdown at most 2 points worse, neighbours robust" % number)
    candidates = []
    for name in designs:
        by = res[name]
        ok, med, better, worst = verdict(by, base)
        w14 = np.mean([by[y].get("w14_comp", 0.0) for y in years])
        print("  %-48s %s | median %.2f, better %d/6, worst drawdown %.0f%%, 14d %.2f%s" % (
            name, " ".join("%6.2f" % by[y]["comp"] for y in years), med, better, worst * 100, w14,
            "" if name == "incumbent" else ("  -> candidate" if ok else "  -> fail")))
        if ok and name != "incumbent":
            candidates.append(name)
    for name in candidates:
        neighbours = {"%s / neighbour %d" % (name, i + 1): o for i, o in enumerate(spec[name][1])}
        nres = results_for(neighbours)
        fine = []
        for n in neighbours:
            ok, med, better, worst = verdict(nres[n], base, strict=False)
            fine.append(ok)
            print("    %-60s median %.2f, better %d/6 -> %s" % (n, med, better, "holds" if ok else "breaks"))
        print("  %s: %s" % (name, "ADOPT" if all(fine) else "not robust, not adopted"))
    if not candidates:
        print("  no candidate: nothing adopted")


if __name__ == "__main__":
    main()
