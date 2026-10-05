"""H70: securing profits within the competition's window, without a fixed profit target.

The user asked for a rule that secures profits and keeps the Sharpe ratio high, but not a fixed
target such as "once up 10%, halve" (h68_lock_in.py's best rule, which raised the median window
composite in all six folds and both holdout years, but cut the mean window return from +5.7% to
+3.7%). The competition first takes the top 20 by return, then ranks on 0.4 Sortino + 0.3
Sharpe + 0.3 Calmar over the window, so a late drawdown costs more than a late gain earns.

Written down before running, on every 14-day window (each starting at a UTC day's close, after
each fold's first 30 days) of the live bot's hourly equity (R54b, the six folds and the 2018-2020
holdout years). Each rule halves the exposure once, for the rest of the window, paying 0.1% on
the half sold; sigma is the bot's daily volatility over the 30 days before the window:

  S1  risk-scaled lock-in: once the window's gain exceeds k sigma times the square root of the
      days left, a gain a k-sigma loss over the remaining time could not erase; the bar adapts to
      volatility and falls as the end nears. k = 1 (neighbours 0.75 and 1.5)
  S2  profit ratchet: once the window has been up at least one sigma, halve when it gives back
      half of its peak gain (neighbours: a third, two thirds)
  S3  profit and the market's trend: while the window is in profit, halve when BTC's 3-day
      return at a UTC close turns negative, the strongest market warning of h69 (neighbours: 2
      and 5 days)

A rule is adopted if it raises the median window composite in at least 5 of the 6 folds while
cutting the mean window return by no more than a tenth, its neighbours raise it in at least 4,
and it raises it in both holdout years. h68's fixed 10% lock is shown for comparison.

Step 2 (--bot) runs the rule as the bot implements it (StrategyConfig.secure_k; the volatility
there comes from the positions' own hourly returns) through the bot's backtester, the window
restarting every 14 days from each fold's start, and scores those windows against the live bot's.

Step 3 (--modes): the user asked to keep the capital invested rather than halve it. Written down
before running, the same trigger (k = 1), and once secured the rotation's coins move:

  book    into the long-short book, so the whole account runs it (about 40 coins, long and short)
  btc     into BTC, the market's leader, with about half an altcoin's volatility
  spread  into the rotation's top 5 coins instead of its top 2
  gold    into PAXG, invested but out of crypto's swings (for comparison)

and, added after those results (only leaving crypto protected the score): half_gold, every
position halved as in step 2 but the freed half put into PAXG instead of cash; then the user's
design, refresh: once secured, the rotation's coins are sold into PAXG and barred for the rest
of the window, so its next daily pick puts the money into other coins (k = 1, and 2).

judged as step 2 (the median window score and Sharpe ratio, six folds and the holdout), with
the mean window return, which should fall less than with halving.

    python -m research.h70_secure_profits          # the window study
    python -m research.h70_secure_profits --bot    # step 2, in the bot's backtester
    python -m research.h70_secure_profits --modes  # step 3, keeping the capital invested
"""
import copy
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.market_data import HOUR_MS, _read_bars
from bot.metrics import calmar, composite, max_drawdown, sharpe, sortino
from research.folds import FOLDS
from research.holdout2018 import HOLDOUT

DAY = 24
COST = 0.001
RULES = {
    "no rule (live bot)": ("none", {}),
    "h68: lock at +10%, half": ("lock", {"lock": 0.10}),
    "S1 risk-scaled lock-in": ("risk", {"k": 1.0}),
    "S2 profit ratchet": ("ratchet", {"keep": 0.5}),
    "S3 profit and BTC's trend": ("trend", {"days": 3}),
}
NEIGHBOURS = {
    "S1 risk-scaled lock-in": [("risk", {"k": 0.75}), ("risk", {"k": 1.5})],
    "S2 profit ratchet": [("ratchet", {"keep": 1 / 3}), ("ratchet", {"keep": 2 / 3})],
    "S3 profit and BTC's trend": [("trend", {"days": 2}), ("trend", {"days": 5})],
}


def load(start):
    df = pd.read_csv(os.path.join("runs", "research", "h60", "r54b_%s.csv" % start))
    return df["ts"].values.astype(np.int64), df["equity"].values


def btc_closes():
    bars = _read_bars(os.path.join("data", "binance", "BTCUSDT_1h.csv"))
    return pd.Series({b.ts + HOUR_MS: b.close for b in bars}).sort_index()


def btc_falling(ts, closes, days):
    """At each curve time: was BTC's return over `days` days, to the last 00:00 UTC close, negative?"""
    midnight = (ts // (DAY * HOUR_MS)) * DAY * HOUR_MS
    now = closes.reindex(midnight).values
    then = closes.reindex(midnight - days * DAY * HOUR_MS).values
    return now < then


def window(r, sigma, falling, kind, p):
    """Hourly returns of one window -> (return, max drawdown, composite, Sharpe) under the rule."""
    a, s, peak, path, n = 1.0, 1.0, 1.0, [1.0], len(r)
    for i in range(n):
        a *= 1 + s * r[i]
        peak = max(peak, a)
        if s == 1.0 and kind != "none":
            gain, left = a - 1, (n - 1 - i) / DAY
            hit = ((kind == "lock" and gain >= p["lock"])
                   or (kind == "risk" and gain >= p["k"] * sigma * math.sqrt(left))
                   or (kind == "ratchet" and peak - 1 >= sigma and gain <= p["keep"] * (peak - 1))
                   or (kind == "trend" and gain > 0 and falling[i]))
            if hit:
                a *= 1 - COST * 0.5
                s = 0.5
        path.append(a)
    daily = path[::DAY]
    d = [b / x - 1 for x, b in zip(daily, daily[1:])]
    total, mdd = path[-1] - 1, max_drawdown(path)
    return total, mdd, composite(sortino(d), sharpe(d), calmar(total, 14, mdd)), sharpe(d)


def fold_windows(ts, eq, closes, kind, p):
    hourly = eq[1:] / eq[:-1] - 1
    daily_eq = eq[::DAY]
    falling = btc_falling(ts[1:], closes, p.get("days", 3))
    out = []
    for start in range(30 * DAY, len(hourly) - 14 * DAY, DAY):
        past = daily_eq[start // DAY - 30:start // DAY + 1]
        sigma = float(np.std(past[1:] / past[:-1] - 1, ddof=1))
        out.append(window(hourly[start:start + 14 * DAY], sigma, falling[start:start + 14 * DAY], kind, p))
    return np.array(out)


def judge(folds, rules, closes):
    """{name: {year: (median composite, mean return, median Sharpe, worst drawdown, share won)}}"""
    curves = {s: load(s) for s, _ in folds}
    out = {}
    for name, (kind, p) in rules.items():
        out[name] = {}
        for start, _ in folds:
            w = fold_windows(*curves[start], closes, kind, p)
            out[name][start[:4]] = (float(np.median(w[:, 2])), float(w[:, 0].mean()), float(np.median(w[:, 3])),
                                    float(w[:, 1].max()), float((w[:, 0] > 0).mean()))
    return out


LIVE = {"strategy": {"book_mode": "long_short", "ls_trend": [240, 960], "short_exclude_external": True,
                     "short_stop_atr": 10.0}}
OUT = os.path.join("runs", "research", "h70")


def bot_curve(job):
    """The bot's hourly equity over one fold with the profits rule at secure_k = k (0: the live
    bot) and secure_mode = mode."""
    k, mode, (start, end) = job
    from bot.backtest import run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import load_history, BinanceClient
    from research.folds import candidates, funding_table, ms
    path = os.path.join(OUT, "k%s%s_%s.csv" % (k, "" if mode == "half" else "-" + mode, start))
    if not os.path.exists(path):
        overrides = copy.deepcopy(LIVE)
        if k > 0:
            overrides["strategy"].update(secure_k=k, secure_mode=mode, window_start_ms=ms(start), window_days=14)
        cfg = load_config()
        apply_overrides(cfg, overrides)
        s, e = ms(start), ms(end)
        slip = candidates(cfg)
        client = BinanceClient()
        bars = {p: load_history(client, p, s - cfg.backtest.warmup_bars * HOUR_MS, e, cfg.backtest.data_dir) for p in slip}
        r = run_backtest(cfg, {p: b for p, b in bars.items() if b}, s, e, cfg.backtest.taker_fee,
                         cfg.backtest.taker_slippage, "taker", monthly_universe=True, slippage_by_pair=slip,
                         external_scores=funding_table(72))
        pd.DataFrame(r.curve, columns=["ts", "equity"]).to_csv(path, index=False)
    return path


def scored_windows(path, start):
    """(return, max drawdown, composite, Sharpe) of each 14-day window from the fold's start."""
    from research.folds import ms
    df = pd.read_csv(path)
    ts, eq = df["ts"].values.astype(np.int64), df["equity"].values
    out, first, day = [], ms(start), DAY * HOUR_MS
    while first + 14 * day <= ts[-1]:
        # Daily values by time (the data has a few missing hours): the last value at each close.
        at = np.maximum(np.searchsorted(ts, [first + i * day for i in range(15)], side="right") - 1, 0)
        daily = eq[at]
        path_ = list(eq[at[0]:at[-1] + 1] / daily[0])
        d = list(daily[1:] / daily[:-1] - 1)
        total, mdd = daily[-1] / daily[0] - 1, max_drawdown(path_)
        out.append((total, mdd, composite(sortino(d), sharpe(d), calmar(total, 14, mdd)), sharpe(d)))
        first += 14 * day
    return np.array(out)


def in_the_bot(variants) -> None:
    os.makedirs(OUT, exist_ok=True)
    folds = list(FOLDS) + list(HOLDOUT)
    jobs = [(k, mode, f) for _, k, mode in variants for f in folds]
    with ProcessPoolExecutor(max_workers=12) as pool:
        paths = dict(zip([(k, mode, f[0]) for k, mode, f in jobs], pool.map(bot_curve, jobs)))
    for label, group in (("six folds", FOLDS), ("holdout", HOLDOUT)):
        print("\n%s: the bot's 14-day windows (26 a year) | median composite, median Sharpe, mean return per year" % label)
        base = {f[0]: scored_windows(paths[(0, "half", f[0])], f[0]) for f in group}
        for name, k, mode in variants:
            rows = {f[0]: scored_windows(paths[(k, mode, f[0])], f[0]) for f in group}
            med = [float(np.median(rows[f[0]][:, 2])) for f in group]
            shp = [float(np.median(rows[f[0]][:, 3])) for f in group]
            better = sum(np.median(rows[f[0]][:, 2]) > np.median(base[f[0]][:, 2]) for f in group)
            sbetter = sum(np.median(rows[f[0]][:, 3]) > np.median(base[f[0]][:, 3]) for f in group)
            year = [float(np.prod(1 + rows[f[0]][:, 0]) - 1) for f in group]
            print("  %-14s comp %s | Sharpe %s | mean %+.1f%% | year %s | composite better %d/%d, Sharpe better %d/%d" % (
                name, " ".join("%6.2f" % x for x in med), " ".join("%5.2f" % x for x in shp),
                np.mean([rows[f[0]][:, 0].mean() for f in group]) * 100, " ".join("%+5.0f%%" % (y * 100) for y in year),
                better, len(group), sbetter, len(group)))


def main() -> None:
    if "--bot" in sys.argv:
        in_the_bot([("live bot", 0, "half")] + [("k = %.1f" % k, k, "half") for k in (1.0, 1.5, 2.0)])
        return
    if "--modes" in sys.argv:
        in_the_bot([("live bot", 0, "half"), ("halve", 1.0, "half")]
                   + [("into " + m if m != "spread" else "spread to 5", 1.0, m) for m in ("book", "btc", "spread", "gold")]
                   + [("half to gold", 1.0, "half_gold"), ("refresh, k = 1", 1.0, "refresh"),
                      ("refresh, k = 2", 2.0, "refresh")])
        return
    closes = btc_closes()
    years = [s[:4] for s, _ in FOLDS]
    res = judge(FOLDS, RULES, closes)
    base = res["no rule (live bot)"]
    base_mean = np.mean([base[y][1] for y in years])
    print("14-day windows of the live bot, six folds (median composite | mean return | median Sharpe per fold):")
    passed = []
    for name, by in res.items():
        better = sum(by[y][0] > base[y][0] for y in years)
        mean = np.mean([by[y][1] for y in years])
        ok = better >= 5 and mean >= 0.9 * base_mean
        print("  %-28s comp %s | mean %+.1f%% | Sharpe %s | worst DD %.0f%% | composite better %d/6%s" % (
            name, " ".join("%5.2f" % by[y][0] for y in years), mean * 100, " ".join("%5.2f" % by[y][2] for y in years),
            max(by[y][3] for y in years) * 100, better,
            "" if name.startswith("no rule") else ("  -> candidate" if ok else "  -> fail")))
        if ok and name in NEIGHBOURS:
            passed.append(name)
    for name in passed:
        nres = judge(FOLDS, {"%s / neighbour %d" % (name, i + 1): r for i, r in enumerate(NEIGHBOURS[name])}, closes)
        fine = True
        for n, by in nres.items():
            better = sum(by[y][0] > base[y][0] for y in years)
            fine &= better >= 4
            print("    %-40s composite better %d/6, mean %+.1f%% -> %s" % (
                n, better, np.mean([by[y][1] for y in years]) * 100, "holds" if better >= 4 else "breaks"))
        if not fine:
            print("  %s: not robust" % name)
            continue
        h = judge(HOLDOUT, {"live": RULES["no rule (live bot)"], name: RULES[name]}, closes)
        hy = sorted(h["live"])
        wins = sum(h[name][y][0] > h["live"][y][0] for y in hy)
        print("  holdout: composite live %s, rule %s; mean return live %s, rule %s -> %s" % (
            " ".join("%.2f" % h["live"][y][0] for y in hy), " ".join("%.2f" % h[name][y][0] for y in hy),
            " ".join("%+.1f%%" % (h["live"][y][1] * 100) for y in hy), " ".join("%+.1f%%" % (h[name][y][1] * 100) for y in hy),
            "CONFIRMED" if wins == len(hy) else "not confirmed"))


if __name__ == "__main__":
    main()
