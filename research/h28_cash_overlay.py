"""H28: round 3 of the stock search. Park only the crypto bot's idle cash in shares.

Every sleeve tried so far (H26-H27) took money from the crypto books and so diluted their best
years. The bot holds 45-65% cash on average, though. This round leaves the crypto books
alone and puts each day's idle cash (1 - invested share at 00:00 UTC) into a defensive share
strategy from H26, on the point-in-time S&P 500 top 50. Written down before running:

  O1  idle cash in H7, the volatility-managed universe (the best round-1 composite after
      simply holding: 2.04, worst drawdown 21%)
  O2  idle cash in H4, trend per stock (the smallest round-1 drawdown, 18%)

Both are compared with the bot alone on daily closes, on the usual rule: a higher median
composite, better in at least 4 of the 6 crypto folds, worst drawdown at most 2 points worse.
Switching costs between cash and shares are already in the share strategies' returns; the
cost of resizing the overlay as the bot's cash changes is charged at 0.15% of each change.

    python -m research.h28_cash_overlay
"""
import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from bot.metrics import summarize
from research import stocks
from research.folds import FOLDS, candidates, ms
from research.h26_stock_strategies import designs, run, universe

COST = 0.0015


def bot_fold(fold):
    start, end = fold
    cfg = load_config()
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {p: load_history(client, p, warm, e, cfg.backtest.data_dir) for p in slippage}
    bars = {p: b for p, b in bars.items() if b}
    r = run_backtest(copy.deepcopy(cfg), bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage,
                     "taker", monthly_universe=True, slippage_by_pair=slippage)
    return start, r.curve, r.exposure


def daily(series_pairs):
    s = pd.Series(dict(series_pairs))
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s.resample("1D").last()


def main() -> None:
    with ProcessPoolExecutor(max_workers=6) as pool:
        bot = {start: (curve, expo) for start, curve, expo in pool.map(bot_fold, FOLDS)}
    data = stocks.load(stocks.members_since())
    close = data["close"][data["close"].index >= stocks.START]
    volume = data["volume"].reindex(close.index)
    uni = universe(close, volume, stocks.member_mask(close.index, list(close.columns)))
    base = designs(close, uni)
    sleeves = {"O1 idle cash in H7 (volatility-managed)": run(close, *base["H7 volatility-managed benchmark"]),
               "O2 idle cash in H4 (trend per stock)": run(close, *base["H4 trend per stock"])}

    rows = {"bot alone": []}
    for name in sleeves:
        rows[name] = []
    for start, end in FOLDS:
        curve, expo = bot[start]
        eq = daily(curve).dropna()
        cash = (1 - daily(expo).reindex(eq.index).ffill()).clip(0, 1)
        r_bot = eq.pct_change().fillna(eq.iloc[0] / 100000.0 - 1)
        to_ms = lambda idx: [int(t.value // 10 ** 6) for t in idx]
        rows["bot alone"].append(summarize(list(zip(to_ms(eq.index), eq.values)), 100000.0))
        for name, sleeve in sleeves.items():
            sr = sleeve.pct_change()
            sr.index = pd.to_datetime(sr.index).tz_localize("UTC") + pd.Timedelta(days=1)  # known by next 00:00
            sr = sr.reindex(eq.index).fillna(0.0)
            held = cash.shift(1).fillna(0.0)                   # idle cash at the start of each day
            r = r_bot + held * sr - COST * held.diff().abs().fillna(held.iloc[0])
            combined = 100000.0 * (1 + r).cumprod()
            rows[name].append(summarize(list(zip(to_ms(combined.index), combined.values)), 100000.0))
    base_c = [st["composite"] for st in rows["bot alone"]]
    base_d = max(st["max_drawdown"] for st in rows["bot alone"])
    print("Composite per crypto fold (daily closes), from October 2020; average idle cash %.0f%%" % (
        100 * np.mean([1 - np.mean([x for _, x in bot[s][1]]) for s, _ in FOLDS])))
    for name, sts in rows.items():
        comps = [st["composite"] for st in sts]
        better = sum(c > b for c, b in zip(comps, base_c))
        worst = max(st["max_drawdown"] for st in sts)
        ok = np.median(comps) > np.median(base_c) and better >= 4 and worst <= base_d + 0.02
        print("  %-42s %s | median %.2f, better %d/6, worst drawdown %.0f%%, 6y %+.0f%%%s" % (
            name, " ".join("%.2f" % c for c in comps), float(np.median(comps)), better, worst * 100,
            100 * (np.prod([1 + st["total_return"] for st in sts]) - 1),
            "" if name == "bot alone" else (" -> PASS" if ok else " -> fail")))


if __name__ == "__main__":
    main()
