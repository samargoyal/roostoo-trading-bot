"""H133: watching every pair minute by minute, holding for minutes to hours.

The user: the bot may call Roostoo 30 times a minute and one ticker call prices every pair, so it
can watch the whole market every few seconds instead of once an hour. H126-H128 tested scalps held
under a minute, which lose to fees; this keeps the brief of H130-H132 (a win rate of 60% or more,
wins at least 1.5 times losses, positive after costs) and lets trades run for 30 minutes to 6
hours, long enough for moves to dwarf the fees.

Data: 1-minute Binance bars of the 64 crypto pairs Roostoo lists (research/klines1m.py), September
2025 to 11 October 2026. Development: September 2025 to March 2026; validation: March to October
2026; then the competition, from its first cycle (4 October 2026, 13:00 UTC). A design must meet the
brief in development and validation (300 trades or more in each) and trade in the competition.

Signals at a minute's close, in at the next minute's open at market (longs; shorts mirror them):
  flush 1m k: a minute down k sd (of the last day's 1-minute returns), k = 4 or 6, on 3x the last
      hour's median volume, closing in the upper half of its range
  flush 5m k: five minutes down k sd (scaled to 5 minutes), k = 3 or 4, on 3x volume, the last
      minute closing in the upper half of the five minutes' range
  reclaim: a minute that trades below the last hour's low and closes back above it, on 2x volume
  each alone or only above the day's EMA (shorts: below it)
Exits, in units of the pair's hourly move (the 1-minute sd times the square root of 60): targets
1, 2 and 3, stops 2, 3 and 4, held 30, 120 or 360 minutes. Roostoo's costs as before.

    python -m research.h133_minute_watch
"""
import os
import warnings

import numpy as np
import pandas as pd

from research import h130_high_win_rate as h

DEV = ("2025-09-01", "2026-03-01")
VAL = ("2026-03-01", "2026-10-01")
COMP = ("2026-10-04 13:00", "2026-10-13")
TARGETS, STOPS, HOLDS = (1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (30, 120, 360)
MIN_N = 300
CHUNK = 4000


def load(symbol):
    parts = []
    for folder in ("binance_1m", "binance_1m_recent"):
        path = os.path.join("data", folder, symbol + ".parquet")
        if os.path.exists(path):
            parts.append(pd.read_parquet(path)[["ts", "open", "high", "low", "close", "volume"]])
    if not parts:
        return None
    return pd.concat(parts).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def signals(df):
    c, hi, lo, v = df["close"], df["high"], df["low"], df["volume"]
    r1 = np.log(c).diff()
    sd1 = r1.rolling(1440, min_periods=720).std().shift(1)
    med = v.rolling(60, min_periods=30).median().shift(1)
    pos = (c - lo) / (hi - lo).replace(0, np.nan)
    r5 = np.log(c / c.shift(5))
    v5 = v.rolling(5).sum()
    med5 = med * 5
    hi5, lo5 = hi.rolling(5).max(), lo.rolling(5).min()
    pos5 = (c - lo5) / (hi5 - lo5).replace(0, np.nan)
    low60, high60 = lo.rolling(60).min().shift(1), hi.rolling(60).max().shift(1)
    ema = c.ewm(span=1440, adjust=False).mean()
    up, down = c > ema, c < ema
    s = {}
    for k in (4, 6):
        flush = (r1 <= -k * sd1) & (v >= 3 * med) & (pos >= 0.5)
        spike = (r1 >= k * sd1) & (v >= 3 * med) & (pos <= 0.5)
        s[("flush 1m %d sd" % k, 1)] = flush
        s[("flush 1m %d sd, uptrend" % k, 1)] = flush & up
        s[("spike 1m %d sd, downtrend" % k, -1)] = spike & down
    for k in (3, 4):
        flush = (r5 <= -k * sd1 * np.sqrt(5)) & (v5 >= 3 * med5) & (pos5 >= 0.5)
        spike = (r5 >= k * sd1 * np.sqrt(5)) & (v5 >= 3 * med5) & (pos5 <= 0.5)
        s[("flush 5m %d sd" % k, 1)] = flush
        s[("flush 5m %d sd, uptrend" % k, 1)] = flush & up
        s[("spike 5m %d sd, downtrend" % k, -1)] = spike & down
    reclaim = (lo < low60) & (c > low60) & (v >= 2 * med)
    reject = (hi > high60) & (c < high60) & (v >= 2 * med)
    s[("reclaim hour low", 1)] = reclaim
    s[("reclaim hour low, uptrend", 1)] = reclaim & up
    s[("reject hour high, downtrend", -1)] = reject & down
    unit = (sd1 * np.sqrt(60)).values                    # the hourly move, as a fraction
    return {k: x.fillna(False).values for k, x in s.items()}, unit


def evaluate(df, sig, unit, side, half):
    ts = df["ts"].values
    op, hi, lo, cl = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(df)
    nxt_ok = np.r_[ts[1:] - ts[:-1] == 60_000, False]
    idx = np.flatnonzero(sig & nxt_ok & np.isfinite(unit) & (unit > 0))
    out = {}
    if not len(idx):
        return out
    maxh = max(HOLDS)
    for start in range(0, len(idx), CHUNK):
        part = idx[start:start + CHUNK]
        e = part + 1
        j = np.minimum(e[:, None] + np.arange(maxh)[None, :], n - 1)
        valid = (e[:, None] + np.arange(maxh)[None, :]) < n
        H, L, C = hi[j], lo[j], cl[j]
        H[~valid], L[~valid], C[~valid] = np.nan, np.nan, np.nan
        p0 = op[e]
        u = unit[part]
        for tm in TARGETS:
            for sm in STOPS:
                tgt, stp = p0 * (1 + side * tm * u), p0 * (1 - side * sm * u)
                hit_t = (H >= tgt[:, None]) if side > 0 else (L <= tgt[:, None])
                hit_s = (L <= stp[:, None]) if side > 0 else (H >= stp[:, None])
                ft = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 7)
                fs = np.where(hit_s.any(1), hit_s.argmax(1), 10 ** 7)
                for hold in HOLDS:
                    last = hold - 1
                    stop_first = (fs <= ft) & (fs <= last)
                    target_first = (ft < fs) & (ft <= last)
                    end_close = C[:, last]
                    done = np.isfinite(end_close) | stop_first | target_first
                    px = np.where(stop_first, stp, np.where(target_first, tgt, end_close))
                    if side > 0:
                        entry = p0 * (1 + half)
                        gross = np.where(target_first, px, px * (1 - half)) / entry - 1
                        net = gross - h.TAKER - np.where(target_first, h.MAKER, h.TAKER)
                    else:
                        entry = p0 * (1 - half)
                        gross = 1 - px * (1 + half) / entry
                        net = gross - 2 * h.SHORT_FEE
                    at = np.where(stop_first, fs, np.where(target_first, ft, last))
                    out.setdefault((tm, sm, hold), []).append(np.c_[ts[e], net, gross, at, done])
    res = {}
    for k, parts in out.items():
        a = np.vstack(parts)
        free, keep = -1, []
        for i in range(len(a)):                      # one trade per pair at a time
            if a[i, 4] and a[i, 0] >= free:
                keep.append(i)
                free = a[i, 0] + (a[i, 3] + 1) * 60_000
        res[k] = a[keep, :3]
    return res


def main() -> None:
    warnings.filterwarnings("ignore")
    spreads = h.pairs_and_spreads()
    acc = {}
    for pair, spread in spreads.items():
        df = load(pair.replace("/USD", "USDT"))
        if df is None or len(df) < 1440 * 60:
            continue
        sigs, unit = signals(df)
        half = max(spread / 2, 0.00005)
        for (name, side), sig in sigs.items():
            for exit_, a in evaluate(df, sig, unit, side, half).items():
                acc.setdefault((name, side, exit_), []).append(a)
        print("  done", pair, flush=True)
    rows = []
    for (name, side, exit_), parts in acc.items():
        a = np.vstack(parts)
        rows.append((name, side, exit_, h.stats(a, *DEV), h.stats(a, *VAL), h.stats(a, *COMP)))

    def show(r):
        name, side, (tm, sm, hold), dv, vl, cp = r
        print("  %s %s | target %.0f stop %.0f hourly moves, %d min | %.1f trades a day\n     dev  %s\n"
              "     val  %s\n     comp %s" % (name, "long" if side > 0 else "short", tm, sm, hold,
                                             (dv["n"] + vl["n"]) / 395.0, h.fmt(dv), h.fmt(vl), h.fmt(cp)))
    full = [r for r in rows if h.meets(r[3], MIN_N) and h.meets(r[4], MIN_N) and r[5]["n"] >= 1]
    print("\nH133: %d designs; meeting the brief in development and validation and trading in the competition: %d"
          % (len(rows), len(full)))
    for r in sorted(full, key=lambda r: -r[4]["mean"]):
        show(r)
    print("\nNearest (300+ trades in each, positive in both), by win rate x ratio in both:")
    near = [r for r in rows if r[3]["n"] >= MIN_N and r[4]["n"] >= MIN_N and r[3]["mean"] > 0 and r[4]["mean"] > 0]
    near.sort(key=lambda r: -(min(r[3]["wr"] / h.MIN_WR, 1.2) * min(r[3]["ratio"] / h.MIN_RATIO, 1.2)
                              * min(r[4]["wr"] / h.MIN_WR, 1.2) * min(r[4]["ratio"] / h.MIN_RATIO, 1.2)))
    for r in near[:15]:
        show(r)


if __name__ == "__main__":
    main()
