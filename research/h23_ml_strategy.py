"""H23: the H21 models' rankings inside the bot's own backtester.

All four H21 models passed the IC screen, mostly by rediscovering low volatility, so by the
rule they get a strategy test. Ridge (the best IC) and the GRU (the deep-learning model) each
rank, walk-forward, (a) the defensive book's coins and (b) the rotation book's rising coins.
The scores are H21's walk-forward predictions, made each day at 00:00 UTC by a model trained
only on earlier data; the bot uses the latest day's. PAXG, outside the models' universe,
keeps the first place it has under low volatility. Fixed before running; the usual rule: a
higher median composite, better in at least 4 of 6 folds, worst drawdown at most 2 points
worse.

    python -m research.h21_learning      # first: writes runs/research/h21_scores.csv.gz
    python -m research.h23_ml_strategy
"""
import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import apply_overrides, load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from research.folds import FOLDS, candidates, ms
from research.h21_learning import SCORES as SCORES_FILE

DESIGNS = [
    ("incumbent", {}, None),
    ("ML1 book ranked by ridge", {"strategy": {"ranking": "external"}}, "ridge"),
    ("ML2 book ranked by GRU", {"strategy": {"ranking": "external"}}, "gru"),
    ("ML3 rotation ranked by ridge", {"strategy": {"rotation_ranking": "external"}}, "ridge"),
    ("ML4 rotation ranked by GRU", {"strategy": {"rotation_ranking": "external"}}, "gru"),
]
SCORES = {}


def load_scores() -> None:
    df = pd.read_csv(SCORES_FILE, compression="gzip")
    df["day"] = pd.to_datetime(df["day"], utc=True)
    for model in ("ridge", "gru"):
        part = df[["day", "pair", model]].dropna()
        SCORES[model] = {int(day.value // 10 ** 6): dict(zip(g["pair"], g[model]))
                         for day, g in part.groupby("day")}


def run(job):
    name, overrides, model, (start, end) = job
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
    result = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                          monthly_universe=True, slippage_by_pair=slippage,
                          external_scores=SCORES.get(model) if model else None)
    st = result.stats
    return name, start[:4], {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"],
                             "w14": st.get("window_composite_median", 0.0)}


def main() -> None:
    jobs = [(n, o, m, fold) for n, o, m in DESIGNS for fold in FOLDS]
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
        rows.append({
            "design": name,
            **{y: "%+.0f%% (%.0f%%) %.2f" % (by_year[y]["ret"] * 100, by_year[y]["mdd"] * 100, by_year[y]["comp"])
               for y in years},
            "median": round(float(np.median(comps)), 2),
            "better": "%d/6" % sum(by_year[y]["comp"] > base[y]["comp"] for y in years),
            "worst mdd": "%.0f%%" % (max(by_year[y]["mdd"] for y in years) * 100),
            "14d median": round(float(np.mean([by_year[y]["w14"] for y in years])), 2)})
    pd.set_option("display.width", 300)
    print("Return (max drawdown) composite per fold, market-order fees; folds start in October")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
