"""Run strategy variants through the simulator on both development periods.

    python -m research.run_variants research/variants/h3.json

Each variant names a score and SimConfig overrides. Scores:
  current        0.5 x 72h + 0.5 x 168h volatility-adjusted momentum (the bot today)
  quality        mean cross-sectional rank of low volatility, liquidity, closeness to the
                 168h high and 336h momentum (fixed weights, nothing fitted)
  low_vol        lowest 168h volatility first
  ml:<model>:<h> walk-forward out-of-sample predictions from research.h2_ml
  ensemble       average rank of quality and ml:gbm:24
"""
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

from bot.config import UniverseConfig
from research.h1_signals import HOLD_OUT_START
from research.panel import cached_pairs, features, load_panel, monthly_universe
from research.sim import SimConfig, simulate, with_fees

PERIODS = {"Y0": ("2024-10-01", "2025-10-01"), "Y1": ("2025-10-01", HOLD_OUT_START)}
VOL_FORECASTS = os.path.join("runs", "research", "vol_forecasts.pkl")
_CACHE = {}


def data(universe_size: int = 20):
    if not _CACHE:
        pairs = cached_pairs()
        panel = load_panel(pairs)
        _CACHE.update(pairs=pairs, panel=panel, feats=features(panel), masks={})
        if os.path.exists(VOL_FORECASTS):
            with open(VOL_FORECASTS, "rb") as fh:
                _CACHE["har"] = pickle.load(fh)["forecasts"]["har"]
    masks = _CACHE["masks"]
    if universe_size not in masks:
        masks[universe_size] = monthly_universe(_CACHE["panel"], UniverseConfig(size=universe_size),
                                                _CACHE["pairs"])
    _CACHE["mask"] = masks[universe_size]
    return _CACHE


def score(name: str) -> pd.DataFrame:
    c = data()
    f, mask = c["feats"], c["mask"]
    rank = lambda x: x.where(mask).rank(axis=1, pct=True)
    if name == "current":
        return (0.5 * f["mom_72"] + 0.5 * f["mom_168"]).where(mask)
    if name == "quality":
        return (rank(-f["vol_168"]) + rank(f["liquidity"]) + rank(f["drawdown_168"]) + rank(f["mom_336"])) / 4
    if name == "low_vol":
        return (-f["vol_168"]).where(mask)
    if name == "low_vol_har":  # ranked by the HAR-RV volatility forecast instead
        return (-c["har"]).where(mask)
    if name == "low_vol+mom336":
        return (rank(-f["vol_168"]) + rank(f["mom_336"])) / 2
    if name == "low_vol+near_high":
        return (rank(-f["vol_168"]) + rank(f["drawdown_168"])) / 2
    if name == "low_vol+liquidity":
        return (rank(-f["vol_168"]) + rank(f["liquidity"])) / 2
    if name == "low_vol+gbm":
        return (rank(-f["vol_168"]) + rank(score("ml:gbm:24"))) / 2
    if name == "ensemble":  # quality blend and boosted-tree forecast, equally weighted by rank
        return (rank(score("quality")) + rank(score("ml:gbm:24"))) / 2
    if name.startswith("ml:"):
        _, model, horizon = name.split(":")
        with open(os.path.join("runs", "research", "predictions_%sh.pkl" % horizon), "rb") as fh:
            preds = pickle.load(fh)
        return preds[model].unstack("pair").reindex(index=mask.index, columns=mask.columns)
    raise ValueError(name)


def run(job):
    name, score_name, overrides, period, maker = job
    overrides = dict(overrides)
    c = data(overrides.pop("universe_size", 20))
    cfg = with_fees(replace(SimConfig(), **overrides), maker)
    start, end = PERIODS[period]
    stats, _ = simulate(c["panel"]["close"], c["mask"], score(score_name), cfg, start, end,
                        high=c["panel"]["high"], low=c["panel"]["low"], vol_forecast=c.get("har"))
    return name, period, maker, stats


def main(path: str) -> None:
    variants = json.load(open(path))
    jobs = [(v["name"], v["score"], v.get("config", {}), p, maker)
            for v in variants for p in PERIODS for maker in (False, True)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, jobs))
    table = {}
    for name, period, maker, st in results:
        row = table.setdefault(name, {})
        if maker:
            row[period + " maker"] = st["total_return"]
        else:
            row[period + " ret"] = st["total_return"]
            row[period + " mdd"] = st["max_drawdown"]
            row[period + " comp"] = st["composite"]
            row[period + " 14d+"] = st.get("window_positive_share", 0)
            row[period + " tr/d"] = st["trades_per_day"]
    df = pd.DataFrame(table).T
    cols = [p + s for p in PERIODS for s in (" ret", " maker", " mdd", " comp", " 14d+", " tr/d")]
    df = df[cols]
    fmt = {c: ("{:.1%}" if c.split(" ")[1] in ("ret", "maker", "mdd", "14d+") else "{:.2f}") for c in cols}
    close = data()["panel"]["close"]["BTC/USD"]
    hold = {}
    for period, (start, end) in PERIODS.items():
        x = close[(close.index >= start) & (close.index < end)]
        hold[period + " ret"] = x.iloc[-1] / x.iloc[0] - 1
        hold[period + " mdd"] = float((1 - x / x.cummax()).max())
    df.loc["(hold BTC)"] = pd.Series(hold)
    pd.set_option("display.width", 200)
    print(df.to_string(formatters={c: fmt[c].format for c in cols}, na_rep=""))


if __name__ == "__main__":
    main(sys.argv[1])
