"""H34: the indicators that passed the H33 screen, used in the strategy in four ways.

Selection, fixed before H33 ran: in each family (momentum, trend, volatility, volume) the
indicator series that passed the 168-hour ranking screen with the largest |IC| (else the
24-hour screen's), and in each family the series that passed the BTC timing screen with the
largest |correlation|. Its sign makes "higher is better".

Uses (each a design; scores computed from data known at each day's 00:00 bar and fed to the
backtester's research hooks):
  U1  the defensive book ranked by low volatility plus the indicator (sum of normal scores)
  U2  the rotation's rising coins ranked by the indicator
  U3  the rotation skips the worst fifth of the universe by the indicator
  U4  (timing indicators) the rotation halved while BTC's indicator is in its bearish fifth of
      the past year
Rule: the round-31 rule (at least 5 of 6 folds better, paired; worst drawdown at most 2 points
worse; neighbours, the indicator's windows x0.7 and x1.4, each at least 4 of 6; then both
2018-2020 holdout years).

    python -m research.h33_indicators          # first: the screen
    python -m research.h34_indicator_strategies
"""
import copy
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import UniverseConfig, apply_overrides, load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from research.folds import FOLDS, candidates, ms
from research.h21_learning import gauss_rank
from research.h33_indicators import OUT as SCREEN, compute
from research.holdout2018 import HOLDOUT
from research.panel import DEFENSIVE, load_panel, monthly_universe

TABLES = os.path.join("runs", "research", "h34_tables.pkl")
_TABLES = {}
HOOK = {"U1": {"strategy": {"ranking": "external"}},
        "U2": {"strategy": {"rotation_ranking": "external"}},
        "U3": {"strategy": {"rotation_exclude_external": True}},
        "U4": {"strategy": {"rotation_external_scale": True}}}


def select():
    t = pd.read_csv(SCREEN)
    picks, timing = [], []
    for family, part in t.groupby("family"):
        p168 = part[part["pass 168h"] == True]
        if len(p168):
            row = p168.loc[p168["IC 168h"].abs().idxmax()]
            picks.append((row["indicator"], row["chart"], float(np.sign(row["IC 168h"]))))
        else:
            p24 = part[part["pass 24h"] == True]
            if len(p24):
                row = p24.loc[p24["IC 24h"].abs().idxmax()]
                picks.append((row["indicator"], row["chart"], float(np.sign(row["IC 24h"]))))
        pt = part[part["pass timing"] == True]
        if len(pt):
            row = pt.loc[pt["BTC timing corr"].abs().idxmax()]
            timing.append((row["indicator"], row["chart"], float(np.sign(row["BTC timing corr"]))))
    return picks, timing


def to_table(frame: pd.DataFrame) -> dict:
    out = {}
    for day, row in frame.iterrows():
        row = row.dropna()
        if len(row):
            out[int(day.value // 10 ** 6)] = row.to_dict()
    return out


def build_tables():
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pairs = sorted(candidates(cfg))
    panel = load_panel(pairs)
    close = panel["close"]
    mask = monthly_universe(panel, UniverseConfig(), list(close.columns))
    mask[DEFENSIVE] = False
    days = close.index[(close.index.hour == 0) & (close.index >= pd.Timestamp("2018-06-01", tz="UTC"))]
    inside = mask.loc[days]
    low_vol = (-np.log(close).diff().rolling(168, min_periods=150).std()).loc[days].where(inside)
    picks, timing = select()
    coins = [p for p in pairs if p != DEFENSIVE]
    tables, designs = {}, []
    for k in (1.0, 0.7, 1.4):
        names = sorted({n for n, _, _ in picks + timing})
        frames = compute(coins, k, names)
        for name, chart, sign in picks:
            f = (frames[(name, chart)].reindex(index=days, columns=close.columns) * sign).where(inside)
            tag = "%s (%s)" % (name, chart)
            tables[("U1", tag, k)] = to_table(gauss_rank(low_vol).fillna(0) + gauss_rank(f).fillna(0))
            tables[("U2", tag, k)] = to_table(f)
            worst = f.rank(axis=1, pct=True) <= 0.2
            tables[("U3", tag, k)] = to_table(pd.DataFrame(1.0, index=days, columns=close.columns).where(~worst, -1.0).where(inside))
            if k == 1.0:
                designs += [("U1", tag), ("U2", tag), ("U3", tag)]
        for name, chart, sign in timing:
            btc = frames[(name, chart)]["BTC/USD"].reindex(days) * sign     # higher is better
            bearish = btc < btc.rolling(365, min_periods=180).quantile(0.2)
            tag = "%s (%s)" % (name, chart)
            tables[("U4", tag, k)] = {int(d.value // 10 ** 6): {"__scale__": 0.5 if b else 1.0} for d, b in bearish.items()}
            if k == 1.0:
                designs.append(("U4", tag))
    with open(TABLES, "wb") as f:
        pickle.dump(tables, f)
    return designs, picks, timing


def init() -> None:
    global _TABLES
    with open(TABLES, "rb") as f:
        _TABLES = pickle.load(f)


def run(job):
    key, (start, end) = job
    cfg = load_config()
    if key is not None:
        apply_overrides(cfg, copy.deepcopy(HOOK[key[0]]))
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {p: load_history(client, p, warm, e, cfg.backtest.data_dir) for p in slippage}
    bars = {p: b for p, b in bars.items() if b}
    r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                     monthly_universe=True, slippage_by_pair=slippage,
                     external_scores=_TABLES.get(key) if key is not None else None)
    return key, start[:4], {"comp": r.stats["composite"], "mdd": r.stats["max_drawdown"]}


def results(keys, folds):
    with ProcessPoolExecutor(max_workers=12, initializer=init) as pool:
        out = {}
        for k, y, r in pool.map(run, [(k, f) for k in keys for f in folds]):
            out.setdefault(k, {})[y] = r
    return out


def main() -> None:
    designs, picks, timing = build_tables()
    print("selected for ranking uses: %s" % ", ".join("%s (%s, sign %+d)" % (n, c, s) for n, c, s in picks))
    print("selected for timing: %s" % ", ".join("%s (%s, sign %+d)" % (n, c, s) for n, c, s in timing))
    years = [f[0][:4] for f in FOLDS]
    res = results([None] + [(u, tag, 1.0) for u, tag in designs], FOLDS)
    base = res[None]
    bworst = max(base[y]["mdd"] for y in years)
    print("Composite per fold (incumbent %s)" % " ".join("%.2f" % base[y]["comp"] for y in years))
    labels = {"U1": "book: low vol +", "U2": "rotation ranked by", "U3": "rotation skips worst fifth by",
              "U4": "rotation halved when BTC is bearish on"}
    for u, tag in designs:
        by = res[(u, tag, 1.0)]
        better = sum(by[y]["comp"] > base[y]["comp"] for y in years)
        worst = max(by[y]["mdd"] for y in years)
        ok = better >= 5 and worst <= bworst + 0.02
        print("  %-70s %s | better %d/6, worst DD %.0f%% -> %s" % (
            "%s %s %s" % (u, labels[u], tag), " ".join("%.2f" % by[y]["comp"] for y in years), better,
            worst * 100, "candidate" if ok else "fail"))
        if not ok:
            continue
        nres = results([(u, tag, 0.7), (u, tag, 1.4)], FOLDS)
        counts = [sum(nres[(u, tag, k)][y]["comp"] > base[y]["comp"] for y in years) for k in (0.7, 1.4)]
        print("    neighbours (windows x0.7, x1.4): %s -> %s" % (
            ", ".join("%d/6" % c for c in counts), "robust" if min(counts) >= 4 else "breaks"))
        if min(counts) < 4:
            continue
        h = results([None, (u, tag, 1.0)], HOLDOUT)
        hy = [f[0][:4] for f in HOLDOUT]
        wins = sum(h[(u, tag, 1.0)][y]["comp"] > h[None][y]["comp"] for y in hy)
        print("    holdout %s: incumbent %s, design %s -> %s" % (
            "/".join(hy), " ".join("%.2f" % h[None][y]["comp"] for y in hy),
            " ".join("%.2f" % h[(u, tag, 1.0)][y]["comp"] for y in hy),
            "CONFIRMED BETTER" if wins == 2 else "not confirmed"))


if __name__ == "__main__":
    main()
