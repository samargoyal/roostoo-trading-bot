"""H130: a strategy that wins at least 60% of its trades with wins clearly larger than losses.

The user's brief: a win rate of 60% or more, the average winning trade well above the average losing
one (here at least 1.5 times), returns secondary; fit on years of history, then tested on the
competition's first seven days.

Protocol, fixed before running:
  - development: entries from October 2020 to October 2024; every design and exit is scored here;
  - validation: October 2024 to October 2026, untouched by the search; a design counts only if it
    meets the brief here too;
  - the competition: 4 to 11 October 2026, too short to judge, reported for the designs that
    pass both.
A design meets the brief with a win rate of 60% or more, an average win at least 1.5 times the
average loss and a positive mean net return per trade, on 150 trades or more in development and
50 or more in validation.

Trades: a signal at a daily close (00:00 UTC) on any Roostoo crypto pair with hourly Binance
history (data/binance), entered at the next hour's open at market, one trade per pair at a time.
Exits are checked hour by hour: a target (a resting limit, maker fee), a stop (market; when an hour
reaches both, the stop counts) or the end of the holding time (market). Costs: 0.1% a side at
market plus half the pair's spread (research/candidates.csv), 0.05% for a target; shorts pay 0.1%
to open and close. Targets and stops are multiples of the pair's 14-day ATR at entry.

Signals (longs; shorts mirror them in downtrends):
  dip k sd: in an uptrend (close above its 200-day average, 50-day above 200-day), a day down k
     standard deviations of the last 20 days' returns or more
  rsi2 < t: in an uptrend, the 2-day RSI below t
  three down: in an uptrend, three down days in a row
  pullback: in an uptrend, within 5 days of a 20-day high, closing at or below the 20-day EMA
  capitulation x: BTC down x% or more on the day: buy the pair (only in its own uptrend for "+trend")
  funding squeeze: perpetual funding over the last 3 days below t (shorts crowded), above the 50-day
     average (shorts mirror: funding above t, crowded longs, below the 50-day average)
Exits: targets 0.5, 1, 1.5, 2 and 3 ATRs, stops 1, 1.5, 2, 3 and 4 ATRs, held at most 2, 5 or 10 days.

    python -m research.h130_high_win_rate
"""
import csv
import os
import warnings

import numpy as np
import pandas as pd

DATA = os.path.join("data", "binance")
FUNDING = os.path.join("data", "funding")
TAKER, MAKER, SHORT_FEE = 0.001, 0.0005, 0.001
DEV = ("2020-10-01", "2024-10-01")
VAL = ("2024-10-01", "2026-10-01")
COMP = ("2026-10-04", "2026-10-12")
TARGETS = (0.5, 1.0, 1.5, 2.0, 3.0)
STOPS = (1.0, 1.5, 2.0, 3.0, 4.0)
HOLDS = (2, 5, 10)
MAX_H = max(HOLDS) * 24
SKIP = {"PAXG/USD", "MUB/USD"}
MIN_WR, MIN_RATIO, MIN_DEV, MIN_VAL = 0.60, 1.5, 150, 50


def ms(day):
    return pd.Timestamp(day, tz="UTC").value // 10 ** 6


def pairs_and_spreads():
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["asset_type"] == "crypto" and r["pair"] not in SKIP]
    return {r["pair"]: float(r["spread"]) for r in rows}


def load_hourly(pair):
    path = os.path.join(DATA, pair.replace("/USD", "USDT") + "_1h.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).drop_duplicates("ts").sort_values("ts")
    return df.reset_index(drop=True)


def daily_bars(h):
    day = h["ts"] // 86_400_000
    g = h.groupby(day)
    d = pd.DataFrame({"o": g["open"].first(), "h": g["high"].max(), "l": g["low"].min(), "c": g["close"].last(),
                      "n": g["close"].size(), "first": g.apply(lambda x: x.index[0])})
    d = d[d["n"] >= 20]
    d.index = d.index * 86_400_000          # the day's start, ms
    return d


def rsi(c, n):
    diff = c.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def funding_daily(coin, days):
    path = os.path.join(FUNDING, coin + ".csv")
    if not os.path.exists(path):
        return pd.Series(np.nan, index=days)
    f = pd.read_csv(path).sort_values("ts")
    s = pd.Series(f["rate"].values, index=f["ts"].values)
    # the mean of the prints in the 3 days up to each day's close
    out = []
    ts, rate = s.index.values, s.values
    for d in days:
        end = d + 86_400_000
        sel = (ts > end - 3 * 86_400_000) & (ts <= end)
        out.append(rate[sel].mean() if sel.any() else np.nan)
    return pd.Series(out, index=days)


def signals(d, btc_ret, fund):
    c = d["c"]
    r = c.pct_change()
    lr = np.log(c).diff()
    sd = lr.rolling(20).std()
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    up = (c > sma200) & (sma50 > sma200)
    down = (c < sma200) & (sma50 < sma200)
    r2 = rsi(c, 2)
    ema20 = c.ewm(span=20, adjust=False).mean()
    near_high = (d["h"].rolling(20).max() == d["h"]).astype(float).rolling(5).max() > 0
    near_low = (d["l"].rolling(20).min() == d["l"]).astype(float).rolling(5).max() > 0
    s = {}
    for k in (1.5, 2.0, 2.5):
        s[("dip %.1f sd" % k, 1)] = up & (lr <= -k * sd)
        s[("dip %.1f sd" % k, -1)] = down & (lr >= k * sd)
    for t in (5, 10):
        s[("rsi2 < %d" % t, 1)] = up & (r2 < t)
        s[("rsi2 < %d" % t, -1)] = down & (r2 > 100 - t)
    s[("three down", 1)] = up & (r < 0) & (r.shift(1) < 0) & (r.shift(2) < 0)
    s[("three down", -1)] = down & (r > 0) & (r.shift(1) > 0) & (r.shift(2) > 0)
    s[("pullback", 1)] = up & near_high & (c <= ema20)
    s[("pullback", -1)] = down & near_low & (c >= ema20)
    for x in (0.04, 0.06):
        cap = btc_ret.reindex(d.index) <= -x
        s[("capitulation %d%%" % (100 * x), 1)] = cap
        s[("capitulation %d%% +trend" % (100 * x), 1)] = cap & up
    for t in (-0.0001, -0.0003):
        s[("funding < %.2f%%" % (100 * t), 1)] = (fund < t) & (c > sma50)
        s[("funding < %.2f%%" % (100 * t), -1)] = (fund > -t * 3) & (c < sma50)
    return {k: v.fillna(False).values for k, v in s.items()}


def build():
    spreads = pairs_and_spreads()
    btc = daily_bars(load_hourly("BTC/USD"))
    btc_ret = btc["c"].pct_change()
    coins = {}
    for pair, spread in spreads.items():
        h = load_hourly(pair)
        if h is None or len(h) < 24 * 260:
            continue
        d = daily_bars(h)
        prev = d["c"].shift(1)
        tr = np.maximum(d["h"] - d["l"], np.maximum((d["h"] - prev).abs(), (d["l"] - prev).abs()))
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        fund = funding_daily(pair.split("/")[0], d.index.values)
        sig = signals(d, btc_ret, fund)
        # entry: the first hour of the next day
        nxt = d["first"].shift(-1)
        ok = nxt.notna() & (d.index.to_series().shift(-1) - d.index.to_series() == 86_400_000)
        coins[pair] = dict(h=h, days=d.index.values, atr=atr.values, entry_row=nxt.values, ok=ok.values,
                           sig=sig, half=max(spread / 2, 0.00005))
        print("  loaded", pair, len(d), "days", flush=True)
    return coins


def paths(cn, idx):
    """Entry price and the next MAX_H hours' highs, lows and closes (NaN-padded) for day indices idx."""
    h = cn["h"]
    hi, lo, cl, op = h["high"].values, h["low"].values, h["close"].values, h["open"].values
    rows = cn["entry_row"][idx].astype(int)
    n = len(h)
    k = np.arange(MAX_H)
    j = rows[:, None] + k[None, :]
    valid = j < n
    j = np.minimum(j, n - 1)
    H, L, C = hi[j], lo[j], cl[j]
    H[~valid], L[~valid], C[~valid] = np.nan, np.nan, np.nan
    return op[rows], H, L, C, h["ts"].values[rows]


def evaluate(coins, name, side):
    """All exit combinations for one signal: {(target, stop, hold): list of (entry ts, net, gross)}."""
    out = {}
    for pair, cn in coins.items():
        sig = cn["sig"].get((name, side))
        if sig is None:
            continue
        idx = np.flatnonzero(sig & cn["ok"] & np.isfinite(cn["atr"]) & (cn["atr"] > 0))
        if not len(idx):
            continue
        p0, H, L, C, ts = paths(cn, idx)
        atr = cn["atr"][idx] / p0                                 # ATR as a fraction of the entry price
        half = cn["half"]
        for tm in TARGETS:
            for sm in STOPS:
                up_ = 1 + side * tm * atr
                dn_ = 1 - side * sm * atr
                tgt, stp = p0 * up_, p0 * dn_
                if side > 0:
                    hit_t = H >= tgt[:, None]
                    hit_s = L <= stp[:, None]
                else:
                    hit_t = L <= tgt[:, None]
                    hit_s = H >= stp[:, None]
                ft = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 6)
                fs = np.where(hit_s.any(1), hit_s.argmax(1), 10 ** 6)
                for hold in HOLDS:
                    last = hold * 24 - 1
                    stop_first = (fs <= ft) & (fs <= last)
                    target_first = (ft < fs) & (ft <= last)
                    end_close = C[:, last]
                    finished = np.isfinite(end_close) | stop_first | target_first
                    exit_px = np.where(stop_first, stp, np.where(target_first, tgt, end_close))
                    exit_at = np.where(stop_first, fs, np.where(target_first, ft, last))
                    if side > 0:
                        entry = p0 * (1 + half)
                        out_px = np.where(target_first, exit_px, exit_px * (1 - half))
                        gross = out_px / entry - 1
                        net = gross - TAKER - np.where(target_first, MAKER, TAKER)
                    else:
                        entry = p0 * (1 - half)
                        out_px = exit_px * (1 + half)
                        gross = 1 - out_px / entry
                        net = gross - 2 * SHORT_FEE
                    # one trade per pair at a time
                    free, keep = -1, []
                    for i in range(len(idx)):
                        if not finished[i]:
                            continue
                        if ts[i] >= free:
                            keep.append(i)
                            free = ts[i] + (exit_at[i] + 1) * 3_600_000
                    keep = np.array(keep, dtype=int)
                    if len(keep):
                        out.setdefault((tm, sm, hold), []).append(
                            np.c_[ts[keep], net[keep], gross[keep]])
    return {k: np.vstack(v) for k, v in out.items()}


def stats(a, start, end):
    sel = (a[:, 0] >= ms(start)) & (a[:, 0] < ms(end))
    x = a[sel, 1]
    if not len(x):
        return dict(n=0, wr=np.nan, ratio=np.nan, mean=np.nan, win=np.nan, loss=np.nan)
    w, l = x[x > 0], x[x <= 0]
    ratio = (w.mean() / -l.mean()) if len(w) and len(l) and l.mean() < 0 else (np.inf if len(w) else 0.0)
    return dict(n=len(x), wr=len(w) / len(x), ratio=ratio, mean=x.mean(),
                win=w.mean() if len(w) else 0.0, loss=l.mean() if len(l) else 0.0)


def meets(s, minimum):
    return s["n"] >= minimum and s["wr"] >= MIN_WR and s["ratio"] >= MIN_RATIO and s["mean"] > 0


def fmt(s):
    if not s["n"]:
        return "   0 trades"
    return "%5d trades WR %4.1f%% win %+5.1f%% loss %+5.1f%% ratio %4.2f mean %+5.2f%%" % (
        s["n"], 100 * s["wr"], 100 * s["win"], 100 * s["loss"], s["ratio"], 100 * s["mean"])


def main() -> None:
    warnings.filterwarnings("ignore")
    coins = build()
    names = sorted({k for cn in coins.values() for k in cn["sig"]})
    rows = []
    for name, side in names:
        res = evaluate(coins, name, side)
        for exit_, a in res.items():
            rows.append((name, side, exit_, stats(a, *DEV), stats(a, *VAL), stats(a, *COMP)))
        print("  scored", name, "long" if side > 0 else "short", flush=True)

    dev_pass = [r for r in rows if meets(r[3], MIN_DEV)]
    print("\nH130: %d designs (signal x side x exit); %d meet the brief in development" % (len(rows), len(dev_pass)))
    both = [r for r in dev_pass if meets(r[4], MIN_VAL)]
    print("Of those, %d meet it again in validation:" % len(both))
    for name, side, (tm, sm, hold), dv, vl, cp in sorted(both, key=lambda r: -r[4]["mean"]):
        print("  %-24s %-5s target %.1f stop %.1f ATR, %2dd | dev %s\n%62s val %s\n%62s comp %s" % (
            name, "long" if side > 0 else "short", tm, sm, hold, fmt(dv), "", fmt(vl), "", fmt(cp)))
    print("\nClosest in development by win rate x ratio (100+ trades), whether or not they pass:")
    near = sorted([r for r in rows if r[3]["n"] >= 100 and r[3]["mean"] > 0],
                  key=lambda r: -(min(r[3]["wr"] / MIN_WR, 1.2) * min(r[3]["ratio"] / MIN_RATIO, 1.2)))[:25]
    for name, side, (tm, sm, hold), dv, vl, cp in near:
        print("  %-24s %-5s t %.1f s %.1f %2dd | dev %s | val %s" % (
            name, "long" if side > 0 else "short", tm, sm, hold, fmt(dv), fmt(vl)))


if __name__ == "__main__":
    main()
