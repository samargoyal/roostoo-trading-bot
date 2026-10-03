"""H24: taker order flow in the strategy.

On complete data the H20 screen also passes the 168-hour taker flow (the share of volume bought
by takers), which an earlier run, made before every coin's candles had downloaded, had failed.
By the rule it gets a strategy test. The bot's candles carry no taker volume, so the scores are
computed here, once a day at 00:00 UTC, and fed to the backtester's "external" ranking (PAXG
keeps its first place, as under low volatility). Fixed before running; the usual rule:

  T1  the defensive book ranked by low volatility plus taker flow (sum of normal scores)
  C2  the book ranked by every signal that passed on complete data: low volatility, narrow
      spread, Kalman trend strength and taker flow (Amihud repeats the spread)

    python -m research.h24_order_flow
"""
import copy
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import UniverseConfig, apply_overrides, load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from research import fullbars
from research.folds import FOLDS, candidates, ms
from research.h20_screen import build
from research.h21_learning import gauss_rank
from research.panel import DEFENSIVE, monthly_universe

FILE = os.path.join("runs", "research", "h24_scores.csv.gz")
DESIGNS = [
    ("incumbent", None),
    ("T1 book: low vol + taker flow", "t1"),
    ("C2 book: low vol + spread + Kalman + taker flow", "c2"),
]
SCORES = {}


def make_scores() -> None:
    cfg = load_config()
    pairs = sorted(candidates(cfg))
    full = fullbars.load(pairs)
    pairs = list(full["close"].columns)
    mask = monthly_universe(full, UniverseConfig(), pairs)
    mask[DEFENSIVE] = False
    sig = build(full, mask)
    z = {k: gauss_rank(v).fillna(0.0).where(mask.loc[v.index]) for k, v in sig.items()}
    t1 = z["low_vol_168"] + z["taker_flow_168"]
    c2 = z["low_vol_168"] - z["cs_spread_168"] + z["kalman_slope"] + z["taker_flow_168"]
    out = pd.DataFrame({"t1": t1.stack(), "c2": c2.stack()})
    out.index.names = ["day", "pair"]
    out.to_csv(FILE, compression="gzip")


def load_scores() -> None:
    df = pd.read_csv(FILE, compression="gzip")
    df["day"] = pd.to_datetime(df["day"], utc=True)
    for key in ("t1", "c2"):
        part = df[["day", "pair", key]].dropna()
        SCORES[key] = {int(day.value // 10 ** 6): dict(zip(g["pair"], g[key])) for day, g in part.groupby("day")}


def run(job):
    name, key, (start, end) = job
    cfg = load_config()
    if key:
        apply_overrides(cfg, {"strategy": {"ranking": "external"}})
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {}
    for pair in slippage:
        series = load_history(client, pair, warm, e, cfg.backtest.data_dir)
        if series:
            bars[pair] = series
    result = run_backtest(copy.deepcopy(cfg), bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage,
                          "taker", monthly_universe=True, slippage_by_pair=slippage,
                          external_scores=SCORES.get(key) if key else None)
    st = result.stats
    return name, start[:4], {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"]}


def main() -> None:
    make_scores()
    jobs = [(n, k, fold) for n, k in DESIGNS for fold in FOLDS]
    with ProcessPoolExecutor(max_workers=6, initializer=load_scores) as pool:
        results = list(pool.map(run, jobs))
    table = {}
    for name, year, r in results:
        table.setdefault(name, {})[year] = r
    years = [f[0][:4] for f in FOLDS]
    base = table["incumbent"]
    rows = []
    for name, by_year in table.items():
        comps = [by_year[y]["comp"] for y in years]
        rows.append({"design": name,
                     **{y: "%+.0f%% (%.0f%%) %.2f" % (by_year[y]["ret"] * 100, by_year[y]["mdd"] * 100,
                                                     by_year[y]["comp"]) for y in years},
                     "median": round(float(np.median(comps)), 2),
                     "better": "%d/6" % sum(by_year[y]["comp"] > base[y]["comp"] for y in years),
                     "worst mdd": "%.0f%%" % (max(by_year[y]["mdd"] for y in years) * 100)})
    pd.set_option("display.width", 300)
    print("Return (max drawdown) composite per fold, market-order fees; folds start in October")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
