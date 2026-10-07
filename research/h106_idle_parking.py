"""H106: parking the book's idle cash in something calm (the user: "keep the left money, but test
investing it in less volatile stocks").

The live book (alone, 24/72-hour EMAs, 0.4% band) leaves part of the account in cash: coins
inside the band, crowded and stopped shorts. Written down before running, over every BTC crash
since 2018 and every year:

  P1  the idle share in PAXG while gold's 14-day return is positive (the bot's option
      ls_idle_horizon; neighbour: 7 days)
  P2  the idle share in a low-volatility stock basket: each month the 5 of Roostoo's tokenized
      stocks (bar COIN, MSTR and CRCL, which follow crypto) with the lowest 60-day volatility,
      equal weights, from the underlying shares' daily closes; the bot cannot do this yet, so it
      is simulated: each day the book's idle share (1 - its gross exposure at the day's start)
      earns the basket's return, paying 0.1% on each change in that share. The list is today's
      tokens, so it carries hindsight (round 16).

    python -m research.h106_idle_parking
"""
import copy
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research.folds import FOLDS, funding_table
from research.h60_swing_mft import curve_stats
from research.h61_swing import load
from research.h95_aggressive_sleeve import OUT, bot_curve
from research.h98_crash_validation import chained, crashes
from research.holdout2018 import HOLDOUT

LIVE = json.load(open("config/comp.json"))["strategy"]
STOCKS = ["MSFT", "GOOGL", "META", "QCOM", "INTC", "GLW", "AMD", "NVDA", "TSLA", "PLTR", "MU", "WDC", "LITE",
          "SNDK", "NBIS", "CBRS"]


def with_exposure(fold):
    """The live book's hourly equity and gross exposure for one fold (cached)."""
    from bot.backtest import run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import HOUR_MS, BinanceClient, load_history
    from research.folds import candidates, ms
    start, end = fold
    path = os.path.join(OUT, "live_book_exposure_%s.csv" % start)
    if not os.path.exists(path):
        cfg = load_config()
        apply_overrides(cfg, copy.deepcopy({"strategy": LIVE}))
        s, e = ms(start), ms(end)
        client = BinanceClient()
        slip = candidates(cfg)
        bars = {p: load_history(client, p, s - cfg.backtest.warmup_bars * HOUR_MS, e, cfg.backtest.data_dir) for p in slip}
        bars = {p: b for p, b in bars.items() if b}
        r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                         monthly_universe=True, slippage_by_pair=slip, external_scores=funding_table(72))
        eq = pd.DataFrame(r.curve, columns=["ts", "equity"]).set_index("ts")
        ex = pd.DataFrame(r.exposure, columns=["ts", "exposure"]).set_index("ts")
        eq.join(ex, how="left").reset_index().to_csv(path, index=False)
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return start, pd.Series(df["equity"].values / df["equity"].values[0], index=idx), \
        pd.Series(df["exposure"].ffill().fillna(0.0).values, index=idx)


def basket():
    closes = {}
    for t in STOCKS:
        path = os.path.join("data", "stocks", t + ".csv")
        if os.path.exists(path):
            d = pd.read_csv(path)
            closes[t] = pd.Series(d["close"].values, index=pd.to_datetime(d["Date"], utc=True))
    px = pd.DataFrame(closes).sort_index()
    r = px.pct_change()
    vol = r.rolling(60, min_periods=60).std()
    month = r.index.to_period("M")
    out = pd.Series(0.0, index=r.index)
    for m in sorted(set(month)):
        days = r.index[month == m]
        first = days[0]
        prior = vol[vol.index < first]
        if prior.empty:
            continue
        v = prior.iloc[-1].dropna()
        if len(v) < 5:
            continue
        pick = v.nsmallest(5).index
        out.loc[days] = r.loc[days, pick].mean(axis=1).fillna(0.0)
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    designs = {"P1 idle in PAXG (14 days)": dict(LIVE, ls_idle_horizon=336),
               "P1 / 7 days": dict(LIVE, ls_idle_horizon=168)}
    with ProcessPoolExecutor(max_workers=8) as pool:
        live = {s: (eq, ex) for s, eq, ex in pool.map(with_exposure, folds)}
        got = {(n, s): c for n, s, c in pool.map(bot_curve, [(n, st, f) for n, st in designs.items() for f in folds])}
    stock = basket()
    curves = {"live book (idle in cash)": {}, "P2 idle in low-vol stocks (simulated)": {}}
    for s, _ in folds:
        eq, ex = live[s]
        curves["live book (idle in cash)"][s] = eq
        day_eq = eq.resample("1D").last().dropna()
        idle = (1.0 - ex.resample("1D").first()).clip(lower=0.0, upper=1.0).reindex(day_eq.index).fillna(0.0)
        r_book = day_eq.pct_change().fillna(0.0)
        r_stock = stock.reindex(day_eq.index.normalize()).fillna(0.0).values
        fee = 0.001 * idle.diff().abs().fillna(idle.iloc[0]).values
        r = r_book.values + idle.shift(1).fillna(0.0).values * r_stock - fee
        curves["P2 idle in low-vol stocks (simulated)"][s] = pd.Series(np.cumprod(1 + r), index=day_eq.index)
    for n in designs:
        curves[n] = {s: got[(n, s)] for s, _ in folds}
    btc = load().c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    mean_idle = np.mean([1 - live[s][1].mean() for s, _ in FOLDS])
    print("The live book's mean idle share 2020-26: %.0f%%" % (mean_idle * 100))
    print("Return over each of the %d crashes | mean | yearly (%s) | worst DD | six years | mean median 14d"
          % (len(events), " ".join(s[:4] for s, _ in folds)))
    for n, by in curves.items():
        eq = chained([by[s] for s, _ in folds])
        crash = []
        for a, b, _ in events:
            seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
            crash.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
        stats = [curve_stats(by[s]) for s, _ in folds]
        six = [x for (s, _), x in zip(folds, stats) if s >= FOLDS[0][0]]
        print("  %-40s %s | %+5.1f%% | %s | %3.0f%% | %+7.0f%% | %5.2f" % (
            n, " ".join("%+5.0f%%" % (x * 100) for x in crash), np.nanmean(crash) * 100,
            " ".join("%+6.0f%%" % (x["ret"] * 100) for x in stats), max(x["mdd"] for x in six) * 100,
            (np.prod([1 + x["ret"] for x in six]) - 1) * 100, np.mean([x["w14"] for x in six])), flush=True)


if __name__ == "__main__":
    main()
