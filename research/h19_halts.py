"""H19: what the halt handling (VECM-ARB's dynamic risk engine, adapted) is worth when coins
the bot holds stop trading.

Roostoo can stop trading a pair (exchangeInfo CanTrade false). The bot used to refuse to
start if any universe pair was halted, and mid-run it kept sending orders in the pair, which
the exchange would refuse. Now it re-reads the rules every hour and the planner drops trades
in halted pairs. The question here is whether the strategy should also be told: it then holds
a halted pair as it is, never enters it, and re-solves the rest of the book around it (equal
risk contributions with the halted weight fixed). Rule, set after schedule 1 and before
schedule 2: tell the strategy only if that beats not telling it in at least 6 of the 12 runs.

The halts are aimed at holdings: every 7 days, one coin the bot bought that week (in the run
with no halts) stops trading 24 hours after that purchase, for 72 hours. Schedule 1 takes
the week's first purchase, schedule 2 its last, to show how much of any difference is luck.
Both halted runs face the same schedule.

    python -m research.h19_halts
"""
import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from bot.planner import BUY
from research.folds import FOLDS, candidates, ms

DAY = 24 * HOUR_MS


def run(job):
    fold, which = job
    start, end = fold
    cfg = load_config()
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {}
    for pair in slippage:
        series = load_history(client, pair, warm, e, cfg.backtest.data_dir)
        if series:
            bars[pair] = series

    def backtest(halts=None, aware=True):
        c = copy.deepcopy(cfg)
        c.strategy.plan_around_halts = aware
        return run_backtest(c, bars, s, e, c.backtest.taker_fee, c.backtest.taker_slippage, "taker",
                            monthly_universe=True, slippage_by_pair=slippage, halts=halts).stats

    base = run_backtest(copy.deepcopy(cfg), bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage,
                        "taker", monthly_universe=True, slippage_by_pair=slippage)
    halts = {}
    for week in range(s, e, 7 * DAY):
        buys = [t for t in base.trades if week <= t.ts < week + 7 * DAY and t.side == BUY]
        buy = (buys[0] if which == 1 else buys[-1]) if buys else None
        if buy is not None:
            halts.setdefault(buy.pair, []).append((buy.ts + DAY, buy.ts + 4 * DAY))
    out = {"no halts": base.stats, "halts, strategy not told": backtest(halts, aware=False),
           "halts, strategy re-solves (N-1)": backtest(halts, aware=True)}
    return which, start[:4], sum(len(v) for v in halts.values()), out


def main() -> None:
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, [(fold, which) for which in (1, 2) for fold in FOLDS]))
    pd.set_option("display.width", 250)
    for which in (1, 2):
        part = [(y, n, out) for w, y, n, out in results if w == which]
        rows = []
        for name in part[0][2]:
            row = {"run": name}
            for year, n, out in part:
                st = out[name]
                row[year] = "%+.0f%% (%.0f%%) %.2f" % (st["total_return"] * 100, st["max_drawdown"] * 100,
                                                       st["composite"])
            comps = [out[name]["composite"] for _, _, out in part]
            row["median comp"] = "%.2f" % float(np.median(comps))
            simple = [out["halts, strategy not told"]["composite"] for _, _, out in part]
            row["better than not told"] = "%d/6" % sum(a > b for a, b in zip(comps, simple))
            rows.append(row)
        print("Schedule %d. Halts per fold: %s" % (which, ", ".join("%s: %d" % (y, n) for y, n, _ in part)))
        print("Return (max drawdown) composite per fold; folds start in October")
        print(pd.DataFrame(rows).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
