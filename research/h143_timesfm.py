"""H143 (part 2): does TimesFM 3.0 rank coins for the all-in rotation?

Uses runs/research/h143/timesfm_daily.csv (research/h143_timesfm_forecasts.py). First the forecasts'
skill: each day, the rank correlation across pairs between the forecast and the realised three-day
(and one-day) log return (the information coefficient, IC), averaged with a t-statistic. Then the
all-in rotation on every pair (config/comp_bomber_k10.json's settings, re-picked daily) ranked by
the forecast instead of momentum (strategy rotation_ranking "external"): by ret3 (the expected
three-day return) and by rr3 (the same over its 10th-90th percentile width), against the momentum
ranking, October 2023 to October 2026 and the competition, with the three-day odds as in H142.

    python -m research.h143_timesfm
"""
import json
import math
import os
import warnings

import numpy as np
import pandas as pd

OUT = os.path.join("runs", "research", "h143")


def skill(df):
    from research.h143_timesfm_forecasts import daily_log_closes
    real = {}
    for pair in df["pair"].unique():
        d = daily_log_closes(pair)
        if d is None:
            continue
        real[pair] = pd.DataFrame({"r1": d.shift(-1) - d, "r3": d.shift(-3) - d},
                                  index=(d.index.astype("int64") // 10 ** 6))
    x = df.join(pd.concat(real, names=["pair", "day"]), on=["pair", "day"]).dropna()
    for f, r in (("ret1", "r1"), ("ret3", "r3"), ("rr3", "r3")):
        vals = {}
        for day, g in x.groupby("day"):
            if len(g) >= 10:
                vals[day] = float(np.corrcoef(g[f].rank().values, g[r].rank().values)[0, 1])
        ics = pd.Series(vals).dropna()
        for label, sel in (("Oct 2023 - Oct 2024", (ics.index < 1727740800000)),
                           ("Oct 2024 - Oct 2026", (ics.index >= 1727740800000)), ("all", ics.index == ics.index)):
            s = ics[sel]
            print("  IC %s vs realised %s, %-20s mean %+.4f  t %+.2f  (%d days)" % (
                f, r, label, s.mean(), s.mean() / s.std() * math.sqrt(len(s)), len(s)))


def run(name, extra, table, start, end):
    from bot.backtest import load_all, run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import HOUR_MS
    from bot.roostoo import RoostooClient
    from bot.universe import fetch_candidates, select_universe
    cfg = load_config("config/comp_bomber_k10.json")
    apply_overrides(cfg, {"strategy": extra})
    s, e = (int(pd.Timestamp(x, tz="UTC").value // 10 ** 6) for x in (start, end))
    pairs = sorted(fetch_candidates(RoostooClient("", "", base_url=cfg.api.base_url), cfg.universe))
    bars = load_all(cfg, pairs, s - cfg.backtest.warmup_bars * HOUR_MS, e)
    universe = select_universe(bars, s, cfg.universe, cfg.strategy.defensive_pair)
    if cfg.strategy.regime_pair not in universe:
        universe.insert(0, cfg.strategy.regime_pair)
    cfg.strategy.universe = universe
    r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker", True,
                     external_scores=table)
    return pd.Series([v for _, v in r.curve], index=pd.to_datetime([t for t, _ in r.curve], unit="ms", utc=True))


def describe(name, s, comp):
    h = s.resample("1h").last().ffill()
    r = (h.shift(-72) / h - 1).dropna()
    yrs = "/".join("%+.0f%%" % (100 * (h[(h.index >= a) & (h.index < b)].iloc[-1] / h[(h.index >= a) & (h.index < b)].iloc[0] - 1))
                   for a, b in (("2023-10-01", "2024-10-01"), ("2024-10-01", "2025-10-01"), ("2025-10-01", "2026-10-01")))
    c = comp[comp.index >= pd.Timestamp("2026-10-04 13:00", tz="UTC")]
    print("  %-32s +10%% %5.1f%% +20%% %4.1f%% -10%% %5.1f%% -20%% %4.1f%% | 3y %+7.0f%% DD %3.0f%% | %s | competition %+.1f%%" % (
        name, 100 * (r >= .10).mean(), 100 * (r >= .20).mean(), 100 * (r <= -.10).mean(), 100 * (r <= -.20).mean(),
        100 * (h.iloc[-1] / h.iloc[0] - 1), 100 * (1 - (h / h.cummax()).min()), yrs, 100 * (c.iloc[-1] / c.iloc[0] - 1)))


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pd.read_csv(os.path.join(OUT, "timesfm_daily.csv"))
    print("TimesFM 3.0 forecast skill (rank IC across pairs, daily):")
    skill(df)
    print("\nAll-in rotation, every pair, re-picked daily (taker fees):")
    for name, extra, col in (("momentum (K10, for comparison)", {}, None),
                             ("TimesFM expected 3-day return", {"rotation_ranking": "external"}, "ret3"),
                             ("TimesFM risk-adjusted 3-day", {"rotation_ranking": "external"}, "rr3")):
        table = {}
        if col:
            for day, g in df.groupby("day"):
                table[int(day)] = dict(zip(g["pair"], g[col]))
        s = run(name, extra, table, "2023-10-01", "2026-10-01")
        comp = run(name, extra, table, "2026-10-04", "2026-10-14")
        describe(name, s, comp)
    _ = json


if __name__ == "__main__":
    main()
