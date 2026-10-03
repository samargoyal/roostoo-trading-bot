"""H27: round 2 of the stock-sleeve search.

Round 1 (H26): no classic anomaly beat simply holding the point-in-time S&P 500 top 50 over
2011-2026, and even holding them, as a 20% sleeve, helped the crypto bot in only 2 of 6 folds.
A sleeve's job is to improve the combined account rather than to beat stocks, so round 2,
written down before running:

  2a  the four round-1 designs with the smallest drawdowns (H7 volatility-managed, H4 trend
      per stock, H3 low volatility, H8 momentum and low volatility), chosen after seeing
      round 1, go straight to gate 3
  2b  three timing hypotheses, gates 1 and 3 as in H26:
      T1  turn of the month (Ariel, 1987; Lakonishok and Smidt, 1988): the universe equally
          weighted from the close two trading days before month end to the close of the
          third trading day of the next month, cash otherwise
      T2  time-series momentum on the index (Moskowitz, Ooi and Pedersen): the universe
          equally weighted while its own 12-month return is positive, checked monthly
      T3  trend per stock scaled to 15% annual volatility (H4 with H7's scaling)
  Gate 3 for a sleeve also needs a positive median composite on its own (the S&P universe).

    python -m research.h27_stock_round2
"""
import json
import os

import numpy as np
import pandas as pd

from bot.metrics import summarize
from research import stocks
from research.folds import FOLDS
from research.h18_vecm import blend
from research.h26_stock_strategies import SLEEVE, designs, run, to_curve, universe, yearly


def gate3(eq: pd.Series):
    rows = []
    for s, e in FOLDS:
        with open(os.path.join("runs", "research", "h18_incumbent_%s.json" % s[:4])) as f:
            inc = [tuple(x) for x in json.load(f)]
        stamps = pd.Series(dict(to_curve(eq, s, e))).sort_index()
        hourly = [(ts, float(stamps[stamps.index <= ts].iloc[-1]) if (stamps.index <= ts).any() else 100000.0)
                  for ts, _ in inc]
        a, m = summarize(inc, 100000.0), summarize(blend(inc, hourly, SLEEVE), 100000.0)
        rows.append((s[:4], a["composite"], m["composite"], a["max_drawdown"], m["max_drawdown"]))
    better = sum(m > a for _, a, m, _, _ in rows)
    med_a, med_m = (float(np.median([r[i] for r in rows])) for i in (1, 2))
    worst_a, worst_m = max(r[3] for r in rows), max(r[4] for r in rows)
    ok = med_m > med_a and better >= 4 and worst_m <= worst_a + 0.02
    return ok, "%s | median %.2f vs %.2f, better %d/6, worst drawdown %.0f%% vs %.0f%%" % (
        " ".join("%s %.2f->%.2f" % (y, a, m) for y, a, m, _, _ in rows), med_m, med_a, better,
        worst_m * 100, worst_a * 100)


def main() -> None:
    data = stocks.load(stocks.members_since())
    close = data["close"][data["close"].index >= stocks.START]
    volume = data["volume"].reindex(close.index)
    member = stocks.member_mask(close.index, list(close.columns))
    uni = universe(close, volume, member)
    base = designs(close, uni)
    rets = close.pct_change()
    n_uni = uni.sum(axis=1).replace(0, np.nan)
    ew = uni.astype(float).div(n_uni, axis=0).fillna(0.0)
    month = close.index.to_period("M")
    pos = pd.Series(np.arange(len(close)), index=close.index)
    first = pos.groupby(month).transform("min").values
    last = pos.groupby(month).transform("max").values
    i = np.arange(len(close))
    # In the market from the close of day (last - 2) to the close of the 3rd day of the next month.
    in_tom = (i >= last - 2) | (i - first <= 1)
    enter = in_tom & ~np.r_[False, in_tom[:-1]]
    leave = ~in_tom & np.r_[False, in_tom[:-1]]
    t1 = ew.mul(in_tom.astype(float), axis=0)
    index = (rets.where(uni).mean(axis=1).fillna(0.0) + 1).cumprod()
    monthly = np.r_[True, month[1:] != month[:-1]]
    t2 = ew.mul((index / index.shift(252) - 1 > 0).astype(float), axis=0)
    trend, weekly = base["H4 trend per stock"]
    port = (trend.shift(1) * rets.fillna(0.0)).sum(axis=1)
    scale = (0.15 / (port.rolling(21).std() * np.sqrt(252))).clip(upper=1.0).fillna(0.0)
    t3 = trend.mul(scale, axis=0)
    new = {"T1 turn of the month": (t1, enter | leave | monthly),
           "T2 index time-series momentum": (t2, monthly),
           "T3 trend per stock, volatility-scaled": (t3, weekly)}

    bench = yearly(run(close, *base["B0 benchmark: universe equal weight"]))
    pd.set_option("display.width", 250)
    print("2a: round-1 designs with the smallest drawdowns, straight to gate 3")
    for name in ("H7 volatility-managed benchmark", "H4 trend per stock", "H3 low volatility",
                 "H8 momentum + low volatility"):
        eq = run(close, *base[name])
        alone = float(np.median([c for _, _, c in yearly(eq)]))
        ok, text = gate3(eq)
        print("  %-38s alone %.2f | %s -> %s" % (name, alone, text, "PASS" if ok and alone > 0 else "fail"))
    print("\n2b: new timing hypotheses (gate 1 against the equal-weight benchmark, then gate 3)")
    for name, (t, reb) in new.items():
        eq = run(close, t, reb)
        y = yearly(eq)
        comps = [c for _, _, c in y]
        better = sum(c > b for (_, _, c), (_, _, b) in zip(y, bench))
        worst = max(d for _, d, _ in y)
        g1 = (np.median(comps) > np.median([b for _, _, b in bench]) and better >= 9
              and worst <= max(d for _, d, _ in bench) + 0.05)
        ok, text = gate3(eq)
        print("  %-38s median %.2f, better than B0 %d/15, worst drawdown %.0f%%, gate 1 %s" % (
            name, float(np.median(comps)), better, worst * 100, "PASS" if g1 else "fail"))
        print("  %-38s gate 3: %s -> %s" % ("", text, "PASS" if ok else "fail"))


if __name__ == "__main__":
    main()
