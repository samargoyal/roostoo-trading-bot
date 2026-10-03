"""H30: funding-rate crowding in the strategy (H29 passed the screen).

Fixed before running; scores are computed from data known at each day's 00:00 bar and fed to
the backtester's research hooks; the usual rule (higher median composite, better in at least 4
of 6 folds, worst drawdown at most 2 points worse):

  F1  the defensive book ranked by low volatility plus low funding (sum of normal scores)
  F2  the rotation skips crowded coins: those in the top fifth of the universe by 7-day funding
  F3  the rotation stays out while BTC's 7-day funding is in the top fifth of its past year

    python -m research.h30_funding_strategy
"""
import copy
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import UniverseConfig, apply_overrides, load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from research.folds import FOLDS, candidates, ms
from research.h21_learning import gauss_rank
from research.h29_funding import fetch
from research.panel import DEFENSIVE, load_panel, monthly_universe

FILE = os.path.join("runs", "research", "h30_scores.csv.gz")
DESIGNS = [
    ("incumbent", None, {}),
    ("F1 book: low vol + low funding", "f1", {"strategy": {"ranking": "external"}}),
    ("F2 rotation skips crowded coins", "f2", {"strategy": {"rotation_exclude_external": True}}),
    ("F3 rotation out while BTC funding is crowded", "f3", {"strategy": {"rotation_exclude_external": True}}),
]
SCORES = {}


def make_scores() -> None:
    cfg = load_config()
    pairs = sorted(candidates(cfg))
    panel = load_panel(pairs)
    pairs = list(panel["close"].columns)
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    mask[DEFENSIVE] = False
    close = panel["close"]
    days = close.index[close.index.hour == 0]
    f7 = {}
    for p in pairs:
        if p != DEFENSIVE:
            s = fetch(p.split("/")[0])
            if len(s):
                f7[p] = s.rolling(21, min_periods=15).mean().reindex(days + pd.Timedelta(hours=1), method="ffill").values
    f7 = pd.DataFrame(f7, index=days).reindex(columns=pairs)
    inside = mask.loc[days]
    funding = f7.where(inside)
    low_vol = (-np.log(close).diff().rolling(168, min_periods=150).std()).loc[days].where(inside)
    f1 = gauss_rank(low_vol).fillna(0) + gauss_rank(-funding).fillna(0)
    crowded = funding.rank(axis=1, pct=True) > 0.8
    f2 = pd.DataFrame(1.0, index=days, columns=pairs).where(~crowded, -1.0)
    btc = f7["BTC/USD"]
    hot = btc > btc.rolling(365, min_periods=180).quantile(0.8)
    f3 = pd.DataFrame(1.0, index=days, columns=pairs).mul(np.where(hot, -1.0, 1.0), axis=0)
    out = pd.DataFrame({"f1": f1.where(inside).stack(), "f2": f2.where(inside).stack(), "f3": f3.where(inside).stack()})
    out.index.names = ["day", "pair"]
    out.to_csv(FILE, compression="gzip")
    print("BTC funding crowded on %.0f%% of days; %.0f%% of coin-days crowded" % (
        100 * hot.mean(), 100 * crowded.where(inside).stack().mean()))


def load_scores() -> None:
    df = pd.read_csv(FILE, compression="gzip")
    df["day"] = pd.to_datetime(df["day"], utc=True)
    for key in ("f1", "f2", "f3"):
        part = df[["day", "pair", key]].dropna()
        SCORES[key] = {int(d.value // 10 ** 6): dict(zip(g["pair"], g[key])) for d, g in part.groupby("day")}


def run(job):
    name, key, overrides, (start, end) = job
    cfg = load_config()
    apply_overrides(cfg, copy.deepcopy(overrides))
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {p: load_history(client, p, warm, e, cfg.backtest.data_dir) for p in slippage}
    bars = {p: b for p, b in bars.items() if b}
    r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                     monthly_universe=True, slippage_by_pair=slippage,
                     external_scores=SCORES.get(key) if key else None)
    st = r.stats
    return name, start[:4], {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"]}


def main() -> None:
    make_scores()
    jobs = [(n, k, o, f) for n, k, o in DESIGNS for f in FOLDS]
    with ProcessPoolExecutor(max_workers=6, initializer=load_scores) as pool:
        results = list(pool.map(run, jobs))
    table = {}
    for name, year, r in results:
        table.setdefault(name, {})[year] = r
    years = [f[0][:4] for f in FOLDS]
    base = table["incumbent"]
    rows = []
    for name, by in table.items():
        comps = [by[y]["comp"] for y in years]
        rows.append({"design": name,
                     **{y: "%+.0f%% (%.0f%%) %.2f" % (by[y]["ret"] * 100, by[y]["mdd"] * 100, by[y]["comp"]) for y in years},
                     "median": round(float(np.median(comps)), 2),
                     "better": "%d/6" % sum(by[y]["comp"] > base[y]["comp"] for y in years),
                     "worst mdd": "%.0f%%" % (max(by[y]["mdd"] for y in years) * 100)})
    pd.set_option("display.width", 300)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
