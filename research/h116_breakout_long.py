"""H116: support/resistance breakouts, long only (the user: "only breakout then", after H115 found
the breakout's longs made money and its shorts lost at every lookback).

A coin goes long on a close above its previous N-hour high and flat on a close below its N-hour
low; every coin in a position equally weighted within the sleeve; costs paid. Written down before
running: N = 72, 168, 336 and 720 hours alone, and the 336-hour sleeve beside the live book
(book alone, 24/72-hour EMAs, HAR weights, stops) at 30% and 50%, mixed daily, against the live
book; each year, the worst drawdown, the six-year total, the 14-day yardstick and every crash.

    python -m research.h116_breakout_long
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h50_long_short import blend
from research.h60_swing_mft import costs, curve_stats
from research.h61_swing import load
from research.h95_aggressive_sleeve import bot_curve
from research.h98_crash_validation import chained, crashes
from research.holdout2018 import HOLDOUT

LIVE = json.load(open("config/comp.json"))["strategy"]


def sleeve(D, n, cost):
    res = D.h.shift(1).rolling(n, min_periods=n).max()
    sup = D.l.shift(1).rolling(n, min_periods=n).min()
    sig = pd.DataFrame(np.nan, index=D.c.index, columns=D.c.columns)
    sig[D.c > res] = 1.0
    sig[D.c < sup] = 0.0
    pos = sig.ffill().fillna(0.0).where(D.ok, 0.0)
    w = pos.div(pos.sum(axis=1).clip(lower=1.0), axis=0)          # the sleeve split over its longs
    hold = w.shift(1).fillna(0.0)
    r = (D.c / D.c.shift(1) - 1).fillna(0.0)
    fees = (hold.diff().abs().fillna(0.0) * cost.reindex(w.columns).fillna(0.002)).sum(axis=1)
    net = (hold * r).sum(axis=1) - fees
    net.index = net.index + pd.Timedelta(hours=1)
    return net


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = sorted(HOLDOUT + FOLDS)
    with ProcessPoolExecutor(max_workers=8) as pool:
        book = {s: c for _, s, c in pool.map(bot_curve, [("live book 4ae7681", LIVE, f) for f in folds])}
    D = load()
    cost = costs(list(D.c.columns))
    btc = D.c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    rows = {"live book": {s: book[s] for s, _ in folds}}
    nets = {n: sleeve(D, n, cost) for n in (72, 168, 336, 720)}
    for n, net in nets.items():
        rows["breakout longs %dh" % n] = {}
        for s, e in folds:
            sl = (net.index > pd.Timestamp(s, tz="UTC")) & (net.index <= pd.Timestamp(e, tz="UTC"))
            rows["breakout longs %dh" % n][s] = (1 + net[sl]).cumprod()
    for share in (0.3, 0.5):
        name = "%.0f%% breakout 336h + %.0f%% book" % (share * 100, (1 - share) * 100)
        rows[name] = {}
        for s, _ in folds:
            b = book[s]
            sv = rows["breakout longs 336h"][s].reindex(b.index).ffill().fillna(1.0)
            rows[name][s] = blend(b, sv / sv.iloc[0], share)
    print("Crashes %d | yearly (%s) | worst DD | six years | mean median 14d | crash returns, mean"
          % (len(events), " ".join(s[:4] for s, _ in folds)))
    for name, by in rows.items():
        stats = [curve_stats(by[s]) for s, _ in folds]
        six = [x for (s, _), x in zip(folds, stats) if s >= FOLDS[0][0]]
        eq = chained([by[s] for s, _ in folds])
        crash = []
        for a, b, _ in events:
            seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
            crash.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
        print("  %-34s %s | %3.0f%% | %+8.0f%% | %5.2f | %s, %+5.1f%%" % (
            name, " ".join("%+6.0f%%" % (x["ret"] * 100) for x in stats), max(x["mdd"] for x in six) * 100,
            (np.prod([1 + x["ret"] for x in six]) - 1) * 100, np.mean([x["w14"] for x in six]),
            " ".join("%+4.0f%%" % (x * 100) for x in crash), np.nanmean(crash) * 100), flush=True)


if __name__ == "__main__":
    main()
