"""D1: a permutation test of the live bot and the previous one (RESEARCH_QUEUE.md part D).

The live settings were chosen after some 75 rounds, and selection across that many tries
inflates the chosen design's backtest even with folds and a holdout. Here each fold's price
history (its warm-up included) is shuffled in blocks of 24 hours, in the same order for every
coin, so the market's drift, volatility, fat tails and cross-coin correlation survive while the
ordering a trend follower lives on is destroyed. Each bar keeps its open, high, low and close
relative to the previous close, and its volume. The full strategy then runs on each shuffled
history through the bot's own backtester.

Statistic: the median 14-day composite of the fold. The p-value is the share of shuffled
histories that match or beat the real one. Coins without a history over the fold and its
warm-up (99% of its hours, the few missing ones filled with flat bars) are left out of both
the real and the shuffled runs, so the comparison is like for
like (the real run here is therefore not exactly the bot's fold result).

    python -m research.d1_permutation [permutations per fold, default 100]
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT

OUT = os.path.join("runs", "research", "d1")
CONFIGS = {"multi-horizon": "config/comp_multi.json", "live (R54b)": "config/comp_r54b.json"}


def shuffled(bars_by_pair, seed):
    """Bars with their content permuted in 24-hour blocks (same order for every coin)."""
    from bot.market_data import Bar
    pairs = list(bars_by_pair)
    n = len(bars_by_pair[pairs[0]])
    blocks = n // 24
    order = np.random.default_rng(seed).permutation(blocks)
    index = np.concatenate([np.arange(b * 24, b * 24 + 24) for b in order] + [np.arange(blocks * 24, n)])
    out = {}
    for pair in pairs:
        bars = bars_by_pair[pair]
        prev = [bars[0].open] + [b.close for b in bars[:-1]]
        rel = [(b.open / p, b.high / p, b.low / p, b.close / p, b.volume) for b, p in zip(bars, prev)]
        price, new = bars[0].open, []
        for t, i in enumerate(index):
            o, h, l, c, v = rel[i]
            new.append(Bar(bars[t].ts, price * o, price * h, price * l, price * c, v))
            price *= c
        out[pair] = new
    return out


def job(args):
    bot, (start, end), seed = args
    path = os.path.join(OUT, "%s_%s_%s.json" % (bot.split()[0], start, seed))
    if os.path.exists(path):
        return json.load(open(path))
    from bot.backtest import run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import HOUR_MS, BinanceClient, load_history
    from research.folds import candidates, funding_table, ms
    cfg = load_config()
    apply_overrides(cfg, {"strategy": json.load(open(CONFIGS[bot]))["strategy"]})
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    slip = candidates(cfg)
    client = BinanceClient()
    from bot.market_data import Bar
    expected = (e - warm) // HOUR_MS
    bars = {}
    for p in slip:
        series = {b.ts: b for b in load_history(client, p, warm, e, cfg.backtest.data_dir) if warm <= b.ts < e}
        if len(series) < 0.99 * expected or warm not in series:
            continue
        full, last = [], series[warm]
        for ts in range(warm, e, HOUR_MS):              # isolated missing hours: flat bars
            last = series.get(ts) or Bar(ts, last.close, last.close, last.close, last.close, 0.0)
            full.append(last)
        bars[p] = full
    if seed:
        bars = shuffled(bars, seed)
    r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                     monthly_universe=True, slippage_by_pair=slip, external_scores=funding_table(72))
    out = {"bot": bot, "fold": start, "seed": seed, "coins": len(bars),
           "w14": r.stats.get("window_composite_median", 0.0), "ret": r.stats.get("total_return", 0.0)}
    json.dump(out, open(path, "w"))
    return out


def main() -> None:
    permutations = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    os.makedirs(OUT, exist_ok=True)
    folds = list(FOLDS) + list(HOLDOUT)
    jobs = [(bot, f, seed) for bot in CONFIGS for f in folds for seed in range(permutations + 1)]
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(job, jobs, chunksize=4))
    for bot in CONFIGS:
        print("\n%s bot: median 14-day composite, real against %d shuffled histories" % (bot, permutations))
        for start, _ in folds:
            rows = [r for r in results if r["bot"] == bot and r["fold"] == start]
            real = next(r for r in rows if r["seed"] == 0)
            fake = np.array([r["w14"] for r in rows if r["seed"] != 0])
            p = (np.sum(fake >= real["w14"]) + 1) / (len(fake) + 1)
            print("  %s (%d coins): real %.2f, shuffled median %.2f (p90 %.2f) -> p = %.3f" % (
                start[:4], real["coins"], real["w14"], np.median(fake), np.quantile(fake, 0.9), p))


if __name__ == "__main__":
    main()
