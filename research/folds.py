"""Six one-year folds, October 2020 to October 2026, through the bot's own backtester.

The cure for overfitting is not to tune until one more year looks good. Instead: test a small
set of whole designs, fixed in advance, on many independent years with very different markets
(the 2020-21 bull run, the 2022 crash, the 2023 recovery, the 2024-25 rally, the 2025-26
decline), and pick by consistency. The selection rule was fixed before running:

    highest median composite score across the six folds; ties go to the smaller worst-fold
    drawdown.

Each fold: market-order fees, the coin list refreshed by the universe rule on the first day of
every month, using only coins Roostoo lists today (a survivorship bias: coins delisted since are
missing, which flatters the earlier years slightly).

    python -m research.folds              # the eight designs fixed in advance
    python -m research.folds --all        # plus the BTC-slot variants added afterwards
    python -m research.folds --universe   # how many of Roostoo's coins to choose from
"""
import copy
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from bot.backtest import run_backtest
from bot.config import apply_overrides, load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history

FOLDS = [("2020-10-01", "2021-10-01"), ("2021-10-01", "2022-10-01"), ("2022-10-01", "2023-10-01"),
         ("2023-10-01", "2024-10-01"), ("2024-10-01", "2025-10-01"), ("2025-10-01", "2026-10-01")]
DESIGNS = [
    ("current: 40% rotation + low-vol book", {}),
    ("low-vol book only", {"strategy": {"rotation_weight": 0.0}}),
    ("momentum-ranked book only", {"strategy": {"rotation_weight": 0.0, "ranking": "momentum"}}),
    ("40% rotation + momentum-ranked book", {"strategy": {"ranking": "momentum"}}),
    ("20% rotation + low-vol book", {"strategy": {"rotation_weight": 0.2}}),
    ("60% rotation + low-vol book", {"strategy": {"rotation_weight": 0.6}}),
    ("current + 15% short sleeve", {"strategy": {"short_exposure": 0.15}}),
    ("rotation only", {"strategy": {"rotation_weight": 0.99}}),
]
# Added after seeing the fold results above, to address the one weak spot (BTC-led rallies):
# part of the rotation sleeve always in BTC while the trend filter is on. Run with --extra.
EXTRA = [
    ("current: 40% rotation + low-vol book", {}),
    ("rotation with a 50% BTC slot, top 1", {"strategy": {"rotation_core_share": 0.5, "rotation_top": 1}}),
    ("rotation with a 34% BTC slot, top 2", {"strategy": {"rotation_core_share": 0.34}}),
    ("current + shorts + 34% BTC slot", {"strategy": {"rotation_core_share": 0.34, "short_exposure": 0.15}}),
]
# How wide a net: the universe rule's size, asset types and spread limit. Coins with wide
# spreads pay half their spread on every trade. Fixed before running.
WIDTH = [
    ("top 20 crypto, spread <= 0.1% (current)", {}),
    ("top 30 crypto, spread <= 0.1%", {"universe": {"size": 30}}),
    ("top 45 crypto, spread <= 0.1% (all)", {"universe": {"size": 45}}),
    ("top 20 crypto + stocks", {"universe": {"asset_type": "any"}}),
    ("top 30 crypto, spread <= 0.3%", {"universe": {"size": 30, "max_spread": 0.003}}),
    ("top 60 of all 86, spread <= 0.5%", {"universe": {"size": 60, "asset_type": "any", "max_spread": 0.005}}),
]
# Round 5 (H17): how the rotation sleeve weights its momentum picks, on the 45-coin list.
# Written down before running, with the rule for adopting one over R0: a higher median
# composite, a higher composite in at least 4 of the 6 folds, and a worst-fold drawdown no
# more than 2 points worse.
# Round 6: convex optimisation in the defensive book (up to 8 coins so the optimiser, not the
# 15% cap, sets the weights), and the split between the books. Same adoption rule as round 5.
DEFENSIVE = [
    ("T0 current", {}),
    ("T1 book: 8 coins, minimum variance", {"strategy": {"max_positions_risk_on": 8, "sizing": "min_variance"}}),
    ("T2 book: 8 coins, equal risk contribution", {"strategy": {"max_positions_risk_on": 8, "sizing": "erc"}}),
    ("T3 50% rotation / 50% book", {"strategy": {"rotation_weight": 0.5}}),
    ("T4 30% rotation / 70% book", {"strategy": {"rotation_weight": 0.3}}),
]
# Round 7: ideas from VECM-ARB, applied to the rotation sleeve's choice of coins. Same rule.
VECM_IDEAS = [
    ("V0 current", {}),
    ("V1 residual momentum (BTC beta removed)", {"strategy": {"rotation_ranking": "residual"}}),
    ("V2 skip picks more than 2 sd above their 1-week mean", {"strategy": {"rotation_max_z": 2.0}}),
    ("V3 both", {"strategy": {"rotation_ranking": "residual", "rotation_max_z": 2.0}}),
]
# Rounds 7 and 8 run against the round-6 winner (8-coin ERC book), with a control that keeps
# 8 coins but the old inverse-ATR sizing, to see what the optimiser itself adds.
ROUNDS_7_8 = [
    ("incumbent: 8-coin ERC book", {}),
    ("control: 8 coins, inverse-ATR sizing", {"strategy": {"sizing": "inverse_atr"}}),
    ("V1 residual momentum", {"strategy": {"rotation_ranking": "residual"}}),
    ("V2 skip picks > 2 sd above 1-week mean", {"strategy": {"rotation_max_z": 2.0}}),
    ("V3 residual + z guard", {"strategy": {"rotation_ranking": "residual", "rotation_max_z": 2.0}}),
    ("C1 rotation CVaR cap 3%", {"strategy": {"rotation_cvar_limit": 0.03}}),
    ("C2 rotation CVaR cap 5%", {"strategy": {"rotation_cvar_limit": 0.05}}),
]
# Round 9: are the rotation engine's settings robust? Each varies one setting either side of
# the default (chosen on 20 coins and two years); the short sleeve is retried on the new design.
# Same adoption rule.
ROUND_9 = [
    ("incumbent", {}),
    ("lookback 168h (1 week)", {"strategy": {"rotation_lookback": 168}}),
    ("lookback 504h (3 weeks)", {"strategy": {"rotation_lookback": 504}}),
    ("lookback 720h (30 days)", {"strategy": {"rotation_lookback": 720}}),
    ("top 1", {"strategy": {"rotation_top": 1}}),
    ("top 3", {"strategy": {"rotation_top": 3}}),
    ("rebalance every 72h", {"strategy": {"rotation_rebalance_hours": 72}}),
    ("trend filter 72h/288h", {"strategy": {"rotation_trend_fast": 72, "rotation_trend_slow": 288}}),
    ("trend filter 336h/1344h", {"strategy": {"rotation_trend_fast": 336, "rotation_trend_slow": 1344}}),
    ("+ 15% short sleeve", {"strategy": {"short_exposure": 0.15}}),
]
# Round 13: the three signals that passed the H20 screen (estimated spread, Kalman trend, and
# Amihud illiquidity, which repeats the spread's information), used in the two rankings. Fixed
# after the screen and before any fold run; same rule.
ROUND_13 = [
    ("incumbent", {}),
    ("C1 book: low vol + narrow spread + Kalman trend", {"strategy": {"ranking": "composite"}}),
    ("K1 rotation ranked by Kalman trend strength", {"strategy": {"rotation_ranking": "kalman"}}),
    ("C1 + K1", {"strategy": {"ranking": "composite", "rotation_ranking": "kalman"}}),
]
# Round 19: the rotation's idle capital in a BTC downtrend backs shorts (the user's hypothesis),
# and volatility forecasts scale the rotation. Fixed before running; same rule.
ROUND_19 = [
    ("incumbent", {}),
    ("A1 bear market: short the 2 weakest coins", {"strategy": {"rotation_shorts": 2}}),
    ("A2 bear market: short the 2 wildest coins", {"strategy": {"rotation_shorts": 2,
                                                                 "rotation_short_ranking": "volatility"}}),
    ("B1 rotation scaled by a HAR volatility forecast", {"strategy": {"rotation_vol_forecast": "har"}}),
    ("B2 rotation scaled by an EWMA volatility forecast", {"strategy": {"rotation_vol_forecast": "ewma"}}),
]
# Round 8: tail-risk control of the rotation sleeve. Same rule.
CVAR = [
    ("C0 current", {}),
    ("C1 rotation 1-day CVaR capped at 3% of equity", {"strategy": {"rotation_cvar_limit": 0.03}}),
    ("C2 rotation 1-day CVaR capped at 5% of equity", {"strategy": {"rotation_cvar_limit": 0.05}}),
]
ROTATION_WEIGHTING = [
    ("R0 top 2, equal (current)", {}),
    ("R1 top 5, equal", {"strategy": {"rotation_top": 5}}),
    ("R2 top 5, inverse volatility", {"strategy": {"rotation_top": 5, "rotation_weighting": "inverse_vol"}}),
    ("R3 top 5, equal risk contribution", {"strategy": {"rotation_top": 5, "rotation_weighting": "erc"}}),
    ("R4 top 5, minimum variance, cap 40%", {"strategy": {"rotation_top": 5, "rotation_weighting": "min_variance",
                                                          "rotation_max_weight": 0.4}}),
]
CANDIDATES = os.path.join("research", "candidates.csv")


def candidate_table():
    """Every Roostoo pair with its asset type and bid-ask spread, frozen on 3 October 2026 so
    the study gives the same answer every time (live spreads drift)."""
    with open(CANDIDATES, newline="") as f:
        return [dict(r, spread=float(r["spread"])) for r in csv.DictReader(f)]


def candidates(cfg) -> dict:
    """Pairs passing the universe filters of `cfg`, with half their spread as slippage."""
    u = cfg.universe
    out = {}
    for row in candidate_table():
        if u.asset_type != "any" and row["asset_type"] != u.asset_type:
            continue
        if row["spread"] <= u.max_spread:
            out[row["pair"]] = max(cfg.backtest.taker_slippage, row["spread"] / 2)
    return out


def ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


_FUNDING = {}


def funding_table(hours: int) -> dict:
    """{00:00 of each day (ms): {pair: mean perpetual funding rate over the `hours` up to then}},
    from data/funding (research round 54). Negative: shorts pay longs, the shorts are crowded."""
    if hours in _FUNDING:
        return _FUNDING[hours]
    import glob
    out = {}
    for path in glob.glob(os.path.join("data", "funding", "*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        s = pd.Series(df["rate"].values, index=pd.to_datetime(df["ts"] // 1000, unit="s", utc=True))
        s = s[~s.index.duplicated()].sort_index()
        mean = s.rolling("%dh" % hours).mean()
        days = pd.date_range(s.index[0].ceil("D"), s.index[-1].floor("D"), freq="D")
        pair = os.path.basename(path)[:-4] + "/USD"
        for day, value in mean.reindex(mean.index.union(days)).ffill().reindex(days).items():
            if value == value:
                out.setdefault(int(day.timestamp() * 1000), {})[pair] = float(value)
    _FUNDING[hours] = out
    return out


def run(job):
    name, overrides, (start, end) = job
    overrides = copy.deepcopy(overrides)
    extra = overrides.pop("research", {})
    cfg = load_config()
    apply_overrides(cfg, overrides)
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {}
    for pair in slippage:
        series = load_history(client, pair, warm, e, cfg.backtest.data_dir)
        if series:
            bars[pair] = series
    if "BTC/USD" not in bars:
        bars["BTC/USD"] = load_history(client, "BTC/USD", warm, e, cfg.backtest.data_dir)
    external = funding_table(extra["funding_hours"]) if extra.get("funding_hours") else None
    if extra.get("attention"):
        from research.attention import attention_table
        merged = {day: dict(v) for day, v in (external or {}).items()}
        for day, v in attention_table().items():
            merged.setdefault(day, {}).update(v)
        external = merged
    result = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage,
                          "taker", monthly_universe=True, slippage_by_pair=slippage, external_scores=external)
    btc = [b.close for b in bars["BTC/USD"] if s <= b.ts < e]
    peak, worst = btc[0], 0.0
    for price in btc:
        peak = max(peak, price)
        worst = max(worst, 1 - price / peak)
    st = result.stats
    return name, start, {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"],
                         "w14_pos": st.get("window_positive_share", 0.0),
                         "w14_comp": st.get("window_composite_median", 0.0),
                         "btc": btc[-1] / btc[0] - 1, "btc_mdd": worst}


def main() -> None:
    if "--universe" in sys.argv:
        designs = WIDTH
    elif "--defensive" in sys.argv:
        designs = DEFENSIVE
    elif "--vecm" in sys.argv:
        designs = VECM_IDEAS
    elif "--cvar" in sys.argv:
        designs = CVAR
    elif "--rounds78" in sys.argv:
        designs = ROUNDS_7_8
    elif "--round9" in sys.argv:
        designs = ROUND_9
    elif "--round13" in sys.argv:
        designs = ROUND_13
    elif "--round19" in sys.argv:
        designs = ROUND_19
    elif "--rotation-weighting" in sys.argv:
        designs = ROTATION_WEIGHTING
    elif "--all" in sys.argv:
        designs = DESIGNS + EXTRA[1:]
    elif "--extra" in sys.argv:
        designs = EXTRA
    else:
        designs = DESIGNS
    jobs = [(n, o, fold) for n, o in designs for fold in FOLDS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, jobs))

    table = {}
    btc = {}
    for name, start, r in results:
        table.setdefault(name, {})[start[:4]] = r
        btc[start[:4]] = (r["btc"], r["btc_mdd"])
    years = [f[0][:4] for f in FOLDS]
    flag = next((a for a in sys.argv[1:] if a.startswith("--")), "--designs")[2:]
    with open(os.path.join("runs", "research", "folds_%s.json" % flag), "w") as f:
        json.dump(table, f, indent=1)
    # Composite score per fold, and how many folds each design beats the first one (the
    # incumbent) in: the adoption rule needs at least 4 of 6.
    base = designs[0][0]
    print("Composite score per fold")
    for name, by_year in table.items():
        better = sum(by_year[y]["comp"] > table[base][y]["comp"] for y in years)
        print("  %-42s %s   better than %s in %d/6" % (
            name, "  ".join("%6.2f" % by_year[y]["comp"] for y in years), base[:12], better))
    # The competition is one 14-day window: for information, the share of positive 14-day
    # windows and the median 14-day composite, each averaged over the folds.
    print("14-day windows (information only): share positive, median composite")
    for name, by_year in table.items():
        print("  %-42s %5.1f%%  %6.2f" % (name, sum(by_year[y].get("w14_pos", 0) for y in years) / 6 * 100,
                                          sum(by_year[y].get("w14_comp", 0) for y in years) / 6))
    rows = []
    for name, by_year in table.items():
        row = {"design": name}
        growth, beat = 1.0, 0
        for y in years:
            r = by_year[y]
            row[y] = "%+.0f%% (%.0f%%)" % (r["ret"] * 100, r["mdd"] * 100)
            growth *= 1 + r["ret"]
            beat += r["ret"] > btc[y][0]
        comps = [by_year[y]["comp"] for y in years]
        row["median comp"] = float(pd.Series(comps).median())
        row["worst comp"] = min(comps)
        row["worst mdd"] = max(by_year[y]["mdd"] for y in years)
        row["6y"] = growth - 1
        row["beats BTC"] = "%d/6" % beat
        rows.append(row)
    hold_growth = 1.0
    hold = {"design": "hold BTC"}
    for y in years:
        hold[y] = "%+.0f%% (%.0f%%)" % (btc[y][0] * 100, btc[y][1] * 100)
        hold_growth *= 1 + btc[y][0]
    hold["6y"] = hold_growth - 1
    df = pd.DataFrame(rows).sort_values(["median comp", "worst mdd"], ascending=[False, True])
    df = pd.concat([df, pd.DataFrame([hold])], ignore_index=True)
    pd.set_option("display.width", 260)
    print("Fold returns (max drawdown), market-order fees; folds start in October of each year")
    print(df.to_string(index=False, na_rep="", formatters={
        "median comp": lambda v: "%.2f" % v if pd.notna(v) else "",
        "worst comp": lambda v: "%.2f" % v if pd.notna(v) else "",
        "worst mdd": lambda v: "%.0f%%" % (v * 100) if pd.notna(v) else "",
        "6y": lambda v: "%+.0f%%" % (v * 100)}))


if __name__ == "__main__":
    main()
