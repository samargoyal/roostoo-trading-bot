"""H132: a 60% win rate with wins 1.5 times losses, from signals that trade every few days.

H131's design met the brief but trades only on BTC crash days (about six a year) and made no trade
in the competition's first week. What raised its win rate and kept its wins large was buying a
pair that fell hard but closed in the upper half of its range (buyers came back). This round looks
for that pattern where it is common, and requires a design to trade: at least 500 trades in
development (four years) and at least one trade in the competition's first week.

Fixed before running, longs (and the mirror short in a downtrend where noted):
  daily, at the close (entries at the next hour, exits in daily ATRs: targets 0.5, 1, 1.5; stops 1.5,
  2, 3; held 1, 2 or 3 days):
    coin bounce k: the pair fell k sd (20-day) or more and closed in the upper half of its range,
        k = 1, 1.5, 2; alone, in its own uptrend, or with BTC above its 200-day average (shorts: rose
        k sd and closed in the lower half, in a downtrend)
    BTC down x: BTC fell x% (2, 3, 4) and the pair closed in the upper half of its range
    breadth y: y% (70, 80) of pairs fell on the day and the pair closed in the upper half
  hourly (entries at the next hour, exits in 24-hour ATRs of hourly bars: targets 1, 2, 3; stops 2, 3,
  4; held 6, 12 or 24 hours):
    hour bounce k: an hour that fell k sd (of the last week's hourly returns) or more on twice the
        median volume of the last day and closed in the upper half of its range, k = 3, 4
Protocol and pass marks as H130 and H131 (development 2020-24, validation 2024-26 with 100 trades or
more, the 2018-20 holdout), Roostoo's fees and half the spread.

    python -m research.h132_frequent_bounces
"""
import warnings

import numpy as np
import pandas as pd

from research import h130_high_win_rate as h

HOLDOUT = ("2018-10-01", "2020-10-01")
MIN_DEV, MIN_VAL, MIN_HOLD = 500, 100, 50


def daily_setup():
    spreads = h.pairs_and_spreads()
    btc = h.daily_bars(h.load_hourly("BTC/USD"))
    btc_ret = btc["c"].pct_change()
    btc_up = btc["c"] > btc["c"].rolling(200).mean()
    coins, rets = {}, {}
    for pair, spread in spreads.items():
        hr = h.load_hourly(pair)
        if hr is None or len(hr) < 24 * 260:
            continue
        d = h.daily_bars(hr)
        prev = d["c"].shift(1)
        tr = np.maximum(d["h"] - d["l"], np.maximum((d["h"] - prev).abs(), (d["l"] - prev).abs()))
        nxt = d["first"].shift(-1)
        ok = nxt.notna() & (d.index.to_series().shift(-1) - d.index.to_series() == 86_400_000)
        rets[pair] = d["c"].pct_change()
        coins[pair] = dict(h=hr, d=d, atr=tr.ewm(alpha=1 / 14, adjust=False).mean().values,
                           entry_row=nxt.values, ok=ok.values, half=max(spread / 2, 0.00005))
    down_share = (pd.DataFrame(rets) < 0).mean(axis=1)
    for pair, cn in coins.items():
        d = cn["d"]
        c = d["c"]
        lr = np.log(c).diff()
        sd = lr.rolling(20).std()
        pos = (c - d["l"]) / (d["h"] - d["l"]).replace(0, np.nan)
        upper, lower = pos >= 0.5, pos <= 0.5
        sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
        up = (c > sma200) & (sma50 > sma200)
        down = (c < sma200) & (sma50 < sma200)
        bu = btc_up.reindex(d.index).fillna(False)
        s = {}
        for k in (1.0, 1.5, 2.0):
            fell = (lr <= -k * sd) & upper
            s[("coin bounce %.1f sd" % k, 1)] = fell
            s[("coin bounce %.1f sd, own uptrend" % k, 1)] = fell & up
            s[("coin bounce %.1f sd, BTC bull" % k, 1)] = fell & bu
            s[("coin fade %.1f sd, downtrend" % k, -1)] = (lr >= k * sd) & lower & down
        for x in (0.02, 0.03, 0.04):
            s[("BTC down %d%%, coin upper half" % (100 * x), 1)] = (btc_ret.reindex(d.index) <= -x) & upper
        for y in (0.7, 0.8):
            s[("breadth %d%% down, coin upper half" % (100 * y), 1)] = (down_share.reindex(d.index) >= y) & upper
        cn["sig"] = {k: v.fillna(False).values for k, v in s.items()}
    return coins


def hourly_setup():
    spreads = h.pairs_and_spreads()
    coins = {}
    for pair, spread in spreads.items():
        hr = h.load_hourly(pair)
        if hr is None or len(hr) < 24 * 260:
            continue
        c, hi, lo, v = hr["close"], hr["high"], hr["low"], hr["volume"]
        lr = np.log(c).diff()
        sd = lr.rolling(168).std().shift(1)
        med = v.rolling(24).median().shift(1)
        pos = (c - lo) / (hi - lo).replace(0, np.nan)
        prev = c.shift(1)
        tr = np.maximum(hi - lo, np.maximum((hi - prev).abs(), (lo - prev).abs()))
        atr = tr.ewm(alpha=1 / 24, adjust=False).mean() * np.sqrt(24)    # about a day's range
        ts = hr["ts"]
        ok = (ts.shift(-1) - ts == 3_600_000).values
        sig = {}
        for k in (3, 4):
            sig[("hour bounce %d sd" % k, 1)] = ((lr <= -k * sd) & (v >= 2 * med) & (pos >= 0.5)).fillna(False).values
        coins[pair] = dict(h=hr, atr=atr.values, entry_row=np.arange(len(hr)) + 1.0, ok=ok, sig=sig,
                           half=max(spread / 2, 0.00005))
    return coins


def run(coins, targets, stops, holds_h, step_h):
    """Like H130's evaluate, with holds in hours; returns rows of (name, side, exit, array)."""
    h.MAX_H = max(holds_h)
    out = []
    names = sorted({k for cn in coins.values() for k in cn["sig"]})
    for name, side in names:
        res = {}
        for pair, cn in coins.items():
            sig = cn["sig"].get((name, side))
            if sig is None:
                continue
            idx = np.flatnonzero(sig & cn["ok"] & np.isfinite(cn["atr"]) & (cn["atr"] > 0))
            if not len(idx):
                continue
            p0, H, L, C, ts = h.paths(cn, idx)
            atr = cn["atr"][idx] / p0
            half = cn["half"]
            for tm in targets:
                for sm in stops:
                    tgt, stp = p0 * (1 + side * tm * atr), p0 * (1 - side * sm * atr)
                    hit_t = (H >= tgt[:, None]) if side > 0 else (L <= tgt[:, None])
                    hit_s = (L <= stp[:, None]) if side > 0 else (H >= stp[:, None])
                    ft = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 6)
                    fs = np.where(hit_s.any(1), hit_s.argmax(1), 10 ** 6)
                    for hold in holds_h:
                        last = hold - 1
                        stop_first = (fs <= ft) & (fs <= last)
                        target_first = (ft < fs) & (ft <= last)
                        end_close = C[:, last]
                        finished = np.isfinite(end_close) | stop_first | target_first
                        exit_px = np.where(stop_first, stp, np.where(target_first, tgt, end_close))
                        if side > 0:
                            entry = p0 * (1 + half)
                            gross = np.where(target_first, exit_px, exit_px * (1 - half)) / entry - 1
                            net = gross - h.TAKER - np.where(target_first, h.MAKER, h.TAKER)
                        else:
                            entry = p0 * (1 - half)
                            gross = 1 - exit_px * (1 + half) / entry
                            net = gross - 2 * h.SHORT_FEE
                        exit_at = np.where(stop_first, fs, np.where(target_first, ft, last))
                        free, keep = -1, []
                        for i in range(len(idx)):
                            if finished[i] and ts[i] >= free:
                                keep.append(i)
                                free = ts[i] + (exit_at[i] + 1) * 3_600_000
                        keep = np.array(keep, dtype=int)
                        if len(keep):
                            res.setdefault((tm, sm, hold // step_h), []).append(np.c_[ts[keep], net[keep], gross[keep]])
        for exit_, parts in res.items():
            out.append((name, side, exit_, np.vstack(parts)))
        print("  scored", name, flush=True)
    return out


def comp_open(a):
    """Trades entered in the competition week, including any still open (marked at the last close)."""
    return h.stats(a, *h.COMP)


def main() -> None:
    warnings.filterwarnings("ignore")
    rows = []
    for label, setup, targets, stops, holds, step in (
            ("daily", daily_setup, (0.5, 1.0, 1.5), (1.5, 2.0, 3.0), (24, 48, 72), 24),
            ("hourly", hourly_setup, (1.0, 2.0, 3.0), (2.0, 3.0, 4.0), (6, 12, 24), 1)):
        for name, side, exit_, a in run(setup(), targets, stops, holds, step):
            rows.append((label, name, side, exit_, a, h.stats(a, *h.DEV), h.stats(a, *h.VAL),
                         h.stats(a, *HOLDOUT), comp_open(a)))
    unit = {"daily": "d", "hourly": "h"}

    def show(r):
        label, name, side, (tm, sm, hd), a, dv, vl, ho, cp = r
        per_day = dv["n"] / (4 * 365.0)
        print("  [%s] %s %s | target %.1f stop %.1f ATR, %d%s | %.1f trades a day in development\n"
              "     dev     %s\n     val     %s\n     2018-20 %s\n     comp    %s" % (
                  label, name, "long" if side > 0 else "short", tm, sm, hd, unit[label], per_day,
                  h.fmt(dv), h.fmt(vl), h.fmt(ho), h.fmt(cp)))

    full = [r for r in rows if h.meets(r[5], MIN_DEV) and h.meets(r[6], MIN_VAL) and h.meets(r[7], MIN_HOLD)
            and r[8]["n"] >= 1]
    print("\nH132: %d designs. Meeting the brief in development, validation and the holdout, and trading "
          "in the competition week: %d" % (len(rows), len(full)))
    for r in sorted(full, key=lambda r: -r[6]["mean"]):
        show(r)
    print("\nNearest (500+ development trades, positive in development and validation), by win rate x ratio:")
    near = [r for r in rows if r[5]["n"] >= MIN_DEV and r[5]["mean"] > 0 and r[6]["n"] and r[6]["mean"] > 0]
    near.sort(key=lambda r: -(min(r[5]["wr"] / h.MIN_WR, 1.2) * min(r[5]["ratio"] / h.MIN_RATIO, 1.2)
                              * min(r[6]["wr"] / h.MIN_WR, 1.2) * min(r[6]["ratio"] / h.MIN_RATIO, 1.2)))
    for r in near[:20]:
        show(r)


if __name__ == "__main__":
    main()
