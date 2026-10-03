"""H26: a separate strategy for the tokenized stocks, from hypotheses to a final strategy.

The crypto rotation fails on shares (README round 16): over two weeks large shares tend to
reverse, where coins keep going. So the shares get their own search. To avoid choosing with
hindsight, every design is first tested on the S&P 500 as it was at each date: each month,
the 50 members with the highest dollar volume over the previous 3 months (history from
github.com/fja05680/sp500; companies long delisted are partly missing from Yahoo, a small
survivorship bias that coverage reports).

Hypotheses, each a whole design with the textbook settings, written down before running
(top = the best fifth of the universe, at least 2; 0.15% per unit of turnover for the fee,
spread and token tracking; decided and traded at the close):
  B0  benchmark: the universe equally weighted, rebalanced monthly
  H1  momentum 12-1 (Jegadeesh and Titman, 1993): top by the return from 12 months to 1 month
      ago, monthly
  H2  H1 only while the equal-weight universe is above its 200-day average, else cash
      (dual momentum, Antonacci)
  H3  low volatility (Baker, Bradley and Wurgler; Frazzini and Pedersen): the calmest by
      6-month daily volatility, monthly
  H4  trend per stock (Faber; Moskowitz, Ooi and Pedersen): each member at 1/N while it
      closes above its 200-day average, weekly
  H5  short-term reversal (Jegadeesh, 1990; Lehmann, 1990): the week's biggest losers, weekly
  H6  52-week-high momentum (George and Hwang, 2004): top by close over the 52-week high, monthly
  H7  volatility-managed (Moreira and Muir, 2017): the benchmark scaled to 15% annual
      volatility from the last month's (at most fully invested), adjusted when the scale moves
      by a tenth
  H8  momentum and low volatility together: top by the sum of their normal scores, monthly

Gates, fixed before running:
  1  on the point-in-time S&P universe, over 15 October-to-October years from 2011: a higher
     median composite than B0, better in at least 9 of 15 years, and a worst drawdown at most
     5 points worse
  2  (information) the same rules on the tokenized shares, a list chosen with hindsight
  3  the gate-1 returns as a 20% sleeve beside the crypto bot: the usual rule on the six
     crypto folds (higher median, better in 4 of 6, worst drawdown at most 2 points worse)

    python -m research.stocks              # first: membership and prices
    python -m research.h26_stock_strategies
"""
import json
import os
from typing import Callable, Dict

import numpy as np
import pandas as pd

from bot.metrics import summarize
from research import stocks
from research.folds import FOLDS, candidate_table
from research.h18_vecm import blend
from research.h25_stock_proxy import daily_stock, underlying

COST = 0.0015
SLEEVE = 0.2
YEARS = [("%d-10-01" % y, "%d-10-01" % (y + 1)) for y in range(2011, 2026)]


def run(close: pd.DataFrame, targets: pd.DataFrame, rebalance: np.ndarray) -> pd.Series:
    """Equity (starting at 1) of target weights set at the close on rebalance days."""
    R = close.pct_change().fillna(0.0).values
    T = targets.reindex(columns=close.columns).fillna(0.0).values
    valid = close.notna().values
    w = np.zeros(close.shape[1])
    equity, out = 1.0, np.empty(len(close))
    for t in range(len(close)):
        gross = float(w @ R[t])
        equity *= 1 + gross
        if 1 + gross > 0:
            w = w * (1 + R[t]) / (1 + gross)
        w[~valid[t]] = 0.0
        if rebalance[t]:
            equity *= 1 - COST * float(np.abs(T[t] - w).sum())
            w = T[t].copy()
        out[t] = equity
    return pd.Series(out, index=close.index)


def universe(close: pd.DataFrame, volume: pd.DataFrame, member: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    """Each month: the `top` members by 63-day dollar volume, with a year of history."""
    dollar = (close * volume).rolling(63, min_periods=40).mean().shift(1)
    listed = (close.notna().cumsum() >= 252).shift(1, fill_value=False)
    month = close.index.to_period("M")
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    for m in month.unique():
        rows = month == m
        first = np.argmax(rows)
        eligible = member.iloc[first] & listed.iloc[first] & dollar.iloc[first].notna()
        chosen = dollar.iloc[first][eligible].nlargest(top).index
        mask.loc[rows, chosen] = True
    return mask


def designs(close: pd.DataFrame, uni: pd.DataFrame) -> Dict[str, tuple]:
    """name -> (targets, rebalance days)."""
    rets = close.pct_change()
    n_uni = uni.sum(axis=1).replace(0, np.nan)
    k = np.maximum(2, (0.2 * n_uni).round()).fillna(2).astype(int)
    month = close.index.to_period("M")
    monthly = np.r_[True, month[1:] != month[:-1]]
    weekly = np.arange(len(close)) % 5 == 0
    ew = uni.astype(float).div(n_uni, axis=0)
    index = (rets.where(uni).mean(axis=1).fillna(0.0) + 1).cumprod()
    market_on = index > index.rolling(200).mean()

    def top_by(score: pd.DataFrame) -> pd.DataFrame:
        s = score.where(uni)
        ranks = s.rank(axis=1, ascending=False)
        pick = ranks.le(k, axis=0) & s.notna()
        return pick.astype(float).div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    mom = close.shift(21) / close.shift(252) - 1
    vol = rets.rolling(126, min_periods=100).std()
    hi = close / close.rolling(252, min_periods=200).max()
    rev = close / close.shift(5) - 1
    sma = close.rolling(200, min_periods=180).mean()

    def z(frame: pd.DataFrame) -> pd.DataFrame:
        r = frame.where(uni).rank(axis=1, pct=True)
        return r - 0.5

    trend = (uni & (close > sma)).astype(float).div(n_uni, axis=0).fillna(0.0)
    realised = rets.where(uni).mean(axis=1).rolling(21).std() * np.sqrt(252)
    scale = (0.15 / realised).clip(upper=1.0).fillna(0.0)
    adjust = np.zeros(len(close), dtype=bool)
    last = None
    for i, sc in enumerate(scale.values):
        if monthly[i] or last is None or abs(sc - last) > 0.1 * max(last, 1e-9):
            adjust[i], last = True, sc
    return {
        "B0 benchmark: universe equal weight": (ew.fillna(0.0), monthly),
        "H1 momentum 12-1": (top_by(mom), monthly),
        "H2 momentum 12-1 + market filter": (top_by(mom).mul(market_on.astype(float), axis=0), monthly),
        "H3 low volatility": (top_by(-vol), monthly),
        "H4 trend per stock": (trend, weekly),
        "H5 short-term reversal": (top_by(-rev), weekly),
        "H6 52-week-high momentum": (top_by(hi), monthly),
        "H7 volatility-managed benchmark": (ew.fillna(0.0).mul(scale, axis=0), adjust | monthly),
        "H8 momentum + low volatility": (top_by(z(mom) + z(-vol)), monthly),
    }


def to_curve(equity: pd.Series, start: str, end: str):
    part = equity[(equity.index >= start) & (equity.index < end)]
    part = part / part.iloc[0] * 100000.0
    return [(int(pd.Timestamp(d).tz_localize("UTC").value // 10 ** 6) + 21 * 3_600_000, v) for d, v in part.items()]


def yearly(equity: pd.Series):
    out = []
    for s, e in YEARS:
        st = summarize(to_curve(equity, s, e), 100000.0)
        out.append((st["total_return"], st["max_drawdown"], st["composite"]))
    return out


def main() -> None:
    names = stocks.members_since()
    data = stocks.load(names)
    close = data["close"][data["close"].index >= stocks.START]
    volume = data["volume"].reindex(close.index)
    print("prices for %d of %d tickers that were members since 2010 (%.0f%%)" % (
        close.shape[1], len(names), 100 * close.shape[1] / len(names)))
    member = stocks.member_mask(close.index, list(close.columns))
    uni = universe(close, volume, member)
    results = {}
    for name, (targets, reb) in designs(close, uni).items():
        results[name] = run(close, targets, reb)

    base = yearly(results["B0 benchmark: universe equal weight"])
    rows, passed = [], []
    for name, eq in results.items():
        y = yearly(eq)
        comps = [c for _, _, c in y]
        better = sum(c > b for (_, _, c), (_, _, b) in zip(y, base))
        worst = max(d for _, d, _ in y)
        ok = (np.median(comps) > np.median([b for _, _, b in base]) and better >= 9
              and worst <= max(d for _, d, _ in base) + 0.05 and not name.startswith("B0"))
        if ok:
            passed.append(name)
        growth = np.prod([1 + r for r, _, _ in y]) - 1
        rows.append({"design": name, "median comp": round(float(np.median(comps)), 2),
                     "better than B0": "%d/15" % better, "worst drawdown": "%.0f%%" % (worst * 100),
                     "15-year growth": "%+.0f%%" % (growth * 100),
                     "losing years": sum(r < 0 for r, _, _ in y), "gate 1": "PASS" if ok else ""})
    pd.set_option("display.width", 250)
    print("\nGate 1: point-in-time S&P 500 top 50, October-to-October years 2011-2026")
    print(pd.DataFrame(rows).to_string(index=False))

    # Gate 2 (information): the same rules on the tokenized shares.
    tickers = [underlying(r["pair"]) for r in candidate_table() if r["asset_type"] == "stock"]
    tclose = pd.DataFrame({t: daily_stock(t) for t in tickers}).sort_index()
    tclose = tclose[tclose.index >= stocks.START]
    tuni = (tclose.notna().cumsum() >= 252) & tclose.notna()
    tres = {n: run(tclose, t, r) for n, (t, r) in designs(tclose, tuni).items()}
    tbase = yearly(tres["B0 benchmark: universe equal weight"])
    print("\nGate 2 (information): tokenized shares, chosen with hindsight")
    for name, eq in tres.items():
        y = yearly(eq)
        print("  %-40s median comp %5.2f, better than B0 in %2d/15" % (
            name, float(np.median([c for _, _, c in y])), sum(c > b for (_, _, c), (_, _, b) in zip(y, tbase))))

    # Gate 3: the gate-1 returns as a 20% sleeve beside the crypto bot.
    print("\nGate 3: crypto bot alone vs with a 20% sleeve of the gate-1 strategy, composite per crypto fold")
    for name in ["B0 benchmark: universe equal weight"] + passed:
        eq = results[name]
        rows3 = []
        for s, e in FOLDS:
            with open(os.path.join("runs", "research", "h18_incumbent_%s.json" % s[:4])) as f:
                inc = [tuple(x) for x in json.load(f)]
            stamps = pd.Series(dict(to_curve(eq, s, e))).sort_index()
            hourly = [(ts, float(stamps[stamps.index <= ts].iloc[-1]) if (stamps.index <= ts).any() else 100000.0)
                      for ts, _ in inc]
            a, m = summarize(inc, 100000.0), summarize(blend(inc, hourly, SLEEVE), 100000.0)
            rows3.append((s[:4], a["composite"], m["composite"], a["max_drawdown"], m["max_drawdown"]))
        better = sum(m > a for _, a, m, _, _ in rows3)
        med_a = float(np.median([a for _, a, _, _, _ in rows3]))
        med_m = float(np.median([m for _, _, m, _, _ in rows3]))
        worst_a = max(r[3] for r in rows3)
        worst_m = max(r[4] for r in rows3)
        verdict = "PASS" if med_m > med_a and better >= 4 and worst_m <= worst_a + 0.02 else "fail"
        print("  %-40s %s | median %.2f vs %.2f, better %d/6, worst drawdown %.0f%% vs %.0f%% -> %s" % (
            name, " ".join("%s %.2f->%.2f" % (y, a, m) for y, a, m, _, _ in rows3), med_m, med_a, better,
            worst_m * 100, worst_a * 100, verdict))


if __name__ == "__main__":
    main()
