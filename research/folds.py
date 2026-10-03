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


def run(job):
    name, overrides, (start, end) = job
    cfg = load_config()
    apply_overrides(cfg, copy.deepcopy(overrides))
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
    result = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage,
                          "taker", monthly_universe=True, slippage_by_pair=slippage)
    btc = [b.close for b in bars["BTC/USD"] if s <= b.ts < e]
    peak, worst = btc[0], 0.0
    for price in btc:
        peak = max(peak, price)
        worst = max(worst, 1 - price / peak)
    st = result.stats
    return name, start, {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"],
                         "btc": btc[-1] / btc[0] - 1, "btc_mdd": worst}


def main() -> None:
    if "--universe" in sys.argv:
        designs = WIDTH
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
