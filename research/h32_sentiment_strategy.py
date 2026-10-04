"""H32 (round 41): Fear & Greed and stablecoin liquidity in the strategy (both passed the H31
screen against BTC's next week).

Fixed before running; the round-31 rule (at least 5 of 6 folds better, paired; worst drawdown at
most 2 points worse; neighbours at least 4 of 6; then both 2018-2020 holdout years):

  R41a  halve the rotation while the Fear & Greed index is 80 or more (extreme greed)
        neighbours: 75, 85
  R41b  halve the rotation while the stablecoin supply's 7-day growth is above the 80th
        percentile of its past year; neighbours: 70th, 90th

    python -m research.h32_sentiment_strategy
"""
import copy
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import apply_overrides, load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from research.folds import FOLDS, candidates, ms
from research.h31_sentiment import fear_greed, stable_supply
from research.holdout2018 import HOLDOUT

SCALE = {}
DESIGNS = {
    "R41a halve rotation in extreme greed (F&G >= 80)": ("fng", 80, [75, 85]),
    "R41b halve rotation when stablecoin growth is in its top fifth": ("stable", 0.8, [0.7, 0.9]),
}


def scale_table(kind: str, level: float) -> dict:
    """{00:00 bar time of each day (ms): {"__scale__": 0.5 or 1.0}} from data known by then."""
    if kind == "fng":
        s = fear_greed()
        hot = s >= level
    else:
        g = stable_supply().pct_change(7)
        hot = g > g.rolling(365, min_periods=180).quantile(level)
    days = pd.date_range("2018-03-01", "2026-10-01", freq="D", tz="UTC")
    hot = hot.reindex(days, method="ffill").fillna(False)
    # the strategy reads the day of the latest closed 00:00 bar: key each day's flag by that bar
    return {int(d.value // 10 ** 6): {"__scale__": 0.5 if h else 1.0} for d, h in hot.items()}


def init() -> None:
    for name, (kind, level, nbrs) in DESIGNS.items():
        for lv in [level] + nbrs:
            SCALE[(kind, lv)] = scale_table(kind, lv)


def run(job):
    key, (start, end) = job
    cfg = load_config()
    if key is not None:
        apply_overrides(cfg, {"strategy": {"rotation_external_scale": True}})
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {p: load_history(client, p, warm, e, cfg.backtest.data_dir) for p in slippage}
    bars = {p: b for p, b in bars.items() if b}
    r = run_backtest(copy.deepcopy(cfg), bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                     monthly_universe=True, slippage_by_pair=slippage,
                     external_scores=SCALE.get(key) if key is not None else None)
    return key, start[:4], {"comp": r.stats["composite"], "mdd": r.stats["max_drawdown"]}


def table(keys, folds):
    jobs = [(k, f) for k in keys for f in folds]
    with ProcessPoolExecutor(max_workers=12, initializer=init) as pool:
        out = {}
        for k, y, r in pool.map(run, jobs):
            out.setdefault(k, {})[y] = r
    return out


def main() -> None:
    years = [f[0][:4] for f in FOLDS]
    keys = [None] + [(kind, lv) for kind, level, nbrs in DESIGNS.values() for lv in [level] + nbrs]
    res = table(keys, FOLDS)
    base = res[None]
    bworst = max(base[y]["mdd"] for y in years)
    print("Composite per fold (incumbent %s)" % " ".join("%.2f" % base[y]["comp"] for y in years))
    for name, (kind, level, nbrs) in DESIGNS.items():
        by = res[(kind, level)]
        better = sum(by[y]["comp"] > base[y]["comp"] for y in years)
        worst = max(by[y]["mdd"] for y in years)
        ok = better >= 5 and worst <= bworst + 0.02
        print("  %-64s %s | better %d/6, worst DD %.0f%% -> %s" % (
            name, " ".join("%.2f" % by[y]["comp"] for y in years), better, worst * 100,
            "candidate" if ok else "fail"))
        if not ok:
            continue
        nb_ok = all(sum(res[(kind, lv)][y]["comp"] > base[y]["comp"] for y in years) >= 4 for lv in nbrs)
        print("    neighbours %s -> %s" % (", ".join("%d/6" % sum(res[(kind, lv)][y]["comp"] > base[y]["comp"]
                                                                for y in years) for lv in nbrs),
                                         "robust" if nb_ok else "breaks"))
        if nb_ok:
            h = table([None, (kind, level)], HOLDOUT)
            hy = [f[0][:4] for f in HOLDOUT]
            wins = sum(h[(kind, level)][y]["comp"] > h[None][y]["comp"] for y in hy)
            print("    holdout: %s" % ("CONFIRMED" if wins == 2 else "not confirmed"))


if __name__ == "__main__":
    main()
