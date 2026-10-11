"""H126: scalps held under a minute, on 1-minute bars of every crypto pair Roostoo lists.

The scalper (bot/scalper.py) must close every trade within a minute. H86 and H125 used 5-minute
bars of 16 coins; a check of 1-minute bars covered only BTC, ETH, SOL and DOGE, whose typical
minute moves (2-6 basis points) cannot pay a 15-20 point round trip. This widens the test to all
65 crypto pairs, including the cheap, volatile ones whose minutes move most, for September 2025 to
September 2026 (research/klines1m.py).

Each trade is decided at a minute's close from that minute and earlier ones, bought (or shorted) at
the next minute's open at market, and closed within that minute:
  - a target m sd (the standard deviation of 1-minute returns over the last day) from the entry,
    a resting limit for longs (maker fee), a market close for shorts (/v6 closes at market);
  - a stop 2 sd the other way, at market; when a minute reaches both, the stop counts;
  - otherwise out at the minute's close, at market.
Prices pay half of each pair's spread (research/candidates.csv) on every market order; fees are
0.1% a side at market, 0.05% for a limit, 0.1% to open and close a short.

Written down before running, longs and shorts each, with targets of 1 and 2 sd:
  M1 burst follow (k = 3, 5): a minute of k sd or more on 3x the volume of the last hour's
     median minute: trade its way
  M2 burst fade (k = 3, 5): the same minute, traded against
  M3 taker flow: taker buys 75% or more of a 3x-volume minute: buy (25% or less: short)
  M4 hour breakout: a close above the last 60 minutes' high on 2x volume: buy (below the low: short)
  M5 BTC leads: BTC moved 3 sd in a minute and the coin less than 1 sd: trade the coin BTC's way
  M6 trend pullback: above its 4-hour EMA, up over the hour, and a -2 sd minute: buy (mirror: short)

Pass: the mean net return per trade positive in at least 3 of the 4 quarters (with 20 trades or
more) and over the year, and again with twice the spread.

    python -m research.h126_minute_scalping
"""
import csv
import glob
import os
import warnings

import numpy as np
import pandas as pd

DATA = os.path.join("data", "binance_1m")
TAKER, MAKER, SHORT_FEE = 0.001, 0.0005, 0.001
QUARTERS = [pd.Timestamp(s, tz="UTC").value // 10 ** 6 for s in
            ("2025-09-01", "2025-12-01", "2026-03-01", "2026-06-01", "2026-10-01")]
NAMES = ["Sep-Nov", "Dec-Feb", "Mar-May", "Jun-Sep"]


def spreads():
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        return {r["pair"].replace("/USD", "USDT"): float(r["spread"]) for r in csv.DictReader(f)}


def load(path):
    df = pd.read_parquet(path)
    df.index = df["ts"].values
    return df


def signals(df, btc_z):
    o, h, l, c, v = (df[k].values for k in ("open", "high", "low", "close", "volume"))
    r = np.r_[np.nan, np.diff(np.log(c))]
    sd = pd.Series(r).rolling(1440, min_periods=720).std().values
    z = r / sd
    med = pd.Series(v).shift(1).rolling(60, min_periods=30).median().values
    vr = v / np.where(med > 0, med, np.nan)
    tb = df["taker_buy_base"].values / np.where(v > 0, v, np.nan)
    hi60 = pd.Series(h).shift(1).rolling(60).max().values
    lo60 = pd.Series(l).shift(1).rolling(60).min().values
    ema = pd.Series(c).ewm(span=240, adjust=False).mean().values
    up_hour = c > np.r_[np.full(60, np.nan), c[:-60]]
    bz = btc_z
    s = {}
    for k in (3, 5):
        burst = vr >= 3
        s["M1 burst follow k=%d" % k] = np.where((z >= k) & burst, 1, np.where((z <= -k) & burst, -1, 0))
        s["M2 burst fade k=%d" % k] = -s["M1 burst follow k=%d" % k]
    s["M3 taker flow"] = np.where((tb >= 0.75) & (vr >= 3), 1, np.where((tb <= 0.25) & (vr >= 3), -1, 0))
    s["M4 hour breakout"] = np.where((c > hi60) & (vr >= 2), 1, np.where((c < lo60) & (vr >= 2), -1, 0))
    if bz is not None:
        calm = np.abs(z) < 1
        s["M5 BTC leads"] = np.where((bz >= 3) & calm, 1, np.where((bz <= -3) & calm, -1, 0))
    s["M6 trend pullback"] = np.where((c > ema) & up_hour & (z <= -2), 1,
                                      np.where((c < ema) & ~up_hour & (z >= 2), -1, 0))
    return s, sd


def trades(df, sig, sd, half, m, side):
    """Net and gross return of each trade: signal at minute i's close, in at i+1's open, out within i+1."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    i = np.flatnonzero(sig[:-1] == side)
    i = i[np.isfinite(sd[i]) & (sd[i] > 0)]
    j = i + 1
    ok = np.isfinite(o[j]) & (df.index.values[j] - df.index.values[i] == 60_000)   # no gap in the data
    i, j = i[ok], j[ok]
    mid, d = o[j], sd[i]
    if side > 0:
        entry = mid * (1 + half)
        target, stop = mid * (1 + m * d), mid * (1 - 2 * d)
        hit_stop, hit_target = l[j] <= stop, (h[j] > target) & ~(l[j] <= stop)
        out = np.where(hit_stop, stop * (1 - half), np.where(hit_target, target, c[j] * (1 - half)))
        fee = TAKER + np.where(hit_target, MAKER, TAKER)
        gross = out / entry - 1
    else:
        entry = mid * (1 - half)
        target, stop = mid * (1 - m * d), mid * (1 + 2 * d)
        hit_stop, hit_target = h[j] >= stop, (l[j] < target) & ~(h[j] >= stop)
        out = np.where(hit_stop, stop * (1 + half), np.where(hit_target, target * (1 + half), c[j] * (1 + half)))
        fee = 2 * SHORT_FEE
        gross = 1 - out / entry
    return df.index.values[j], gross, gross - fee


def main() -> None:
    warnings.filterwarnings("ignore")
    spread = spreads()
    paths = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
    btc = load(os.path.join(DATA, "BTCUSDT.parquet"))
    rb = np.r_[np.nan, np.diff(np.log(btc["close"].values))]
    btc_z = pd.Series(rb / pd.Series(rb).rolling(1440, min_periods=720).std().values, index=btc.index)
    results = {}       # (design, m, side, spread multiple) -> list of (ts, net, gross, spread)
    for path in paths:
        sym = os.path.basename(path).replace(".parquet", "")
        df = load(path)
        if len(df) < 1440 * 30:
            continue
        sigs, sd = signals(df, None if sym == "BTCUSDT" else btc_z.reindex(df.index).values)
        for mult in (1, 2):
            half = max(spread.get(sym, 0.002) / 2, 0.00005) * mult
            for name, sig in sigs.items():
                for m in (1, 2):
                    for side in (1, -1):
                        ts, g, n = trades(df, sig, sd, half, m, side)
                        results.setdefault((name, m, side, mult), []).append(
                            (ts, n, g, np.full(len(ts), spread.get(sym, 0.002))))
        print("  done", sym, len(df), flush=True)

    print("\nH126 minute scalps, %d coins. Net bp per trade by quarter %s" % (len(paths), " ".join(NAMES)))
    passed = []
    for key in sorted(k for k in results if k[3] == 1):
        name, m, side, _ = key
        verdicts = []
        for mult in (1, 2):
            parts = results[(name, m, side, mult)]
            ts = np.concatenate([p[0] for p in parts])
            net = np.concatenate([p[1] for p in parts])
            gross = np.concatenate([p[2] for p in parts])
            sp = np.concatenate([p[3] for p in parts])
            q = np.searchsorted(QUARTERS, ts, side="right") - 1
            cells, good, have = [], 0, 0
            for k in range(4):
                x = net[q == k]
                if len(x) < 20:
                    cells.append("     --")
                    continue
                have += 1
                good += x.mean() > 0
                cells.append("%+7.1f" % (1e4 * x.mean()))
            ok = len(net) > 0 and good >= min(3, have) and have > 0 and net.mean() > 0
            verdicts.append(ok)
            if mult == 1:
                groups = " ".join("%s %+.1f" % (lab, 1e4 * net[sel].mean()) if sel.sum() else "%s --" % lab
                                  for lab, sel in (("tight", sp < 0.0002), ("mid", (sp >= 0.0002) & (sp < 0.0008)),
                                                   ("wide", sp >= 0.0008)))
                print("  %-22s m=%d %-5s %s | %7d trades (%5.1f/day) | bp gross %+6.1f net %+6.1f | by spread: %s"
                      " | quarters %d/%d" % (name, m, "long" if side > 0 else "short", " ".join(cells), len(net),
                                             len(net) / 395.0, 1e4 * gross.mean() if len(gross) else 0,
                                             1e4 * net.mean() if len(net) else 0, groups, good, have), flush=True)
        if all(verdicts):
            passed.append((name, m, "long" if side > 0 else "short"))
    print("Passed (and with twice the spread):", passed or "none")


if __name__ == "__main__":
    main()
