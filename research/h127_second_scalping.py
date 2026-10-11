"""H127: scalps held under a minute, simulated second by second on 1-second Binance bars.

This runs the scalper (bot/scalper.py) as it trades: a signal is read at each second's close, the
order goes in one second later (time to react), and the exit is checked on every following
second's high and low until the target, the stop or 55 seconds, whichever comes first. Data: 20
pairs, the majors and the cheap, volatile coins Roostoo lists, July to September 2026
(research/klines1s.py); each month is one fold.

Costs as H126: a market buy pays half the pair's spread (research/candidates.csv) and 0.1%; a
target is a resting limit sell (0.05%, no spread); a stop or the 55-second exit sells at market.
Shorts pay 0.1% to open and to close (/v6) and close at market. When a second reaches both the
target and the stop, the stop counts. One trade per pair at a time; no new signal in a pair for 60
seconds after an entry.

Written down before running, longs and shorts each:
  S1 move follow (the scalper's rule): the price moved 15 bp or more over 20 seconds (and 30 bp):
     trade its way
  S2 move fade: the same moves, traded against
  S3 volume burst: 10 seconds of 5x the hour's median 10-second volume, moving 2 sd or more:
     trade its way
  S4 taker flow: taker buys 80% or more of 10 seconds of 3x volume: buy (20% or less: short)
  S5 BTC leads: BTC moved 3 sd over 10 seconds and the pair less than 1 sd: trade it BTC's way
  S6 spike fade: a 5-second move of 4 sd or more, traded against
Exits: the scalper's fixed 25 bp target and 20 bp stop, and targets scaled to the pair, 1.5 sd of
its 1-minute returns over the last hour, with a 2 sd stop.

Pass: the mean net return per trade positive in all 3 months (with 20 trades or more) and over
all of them, and again with twice the spread.

    python -m research.h127_second_scalping
"""
import csv
import glob
import os
import warnings

import numpy as np
import pandas as pd

DATA = os.path.join("data", "binance_1s")
TAKER, MAKER, SHORT_FEE = 0.001, 0.0005, 0.001
HOLD, COOLDOWN = 55, 60
MONTHS = ["2026-07", "2026-08", "2026-09"]


def spreads():
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        return {r["pair"].replace("/USD", "USDT"): float(r["spread"]) for r in csv.DictReader(f)}


def load(symbol, month):
    path = os.path.join(DATA, "%s-%s.parquet" % (symbol, month))
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    start = pd.Timestamp(month + "-01", tz="UTC").value // 10 ** 6
    end = (pd.Timestamp(month + "-01", tz="UTC") + pd.offsets.MonthBegin(1)).value // 10 ** 6
    idx = np.arange(start, end, 1000)
    df = df.set_index("ts").reindex(idx)
    c = df["close"].ffill().values.astype("float64")
    fill = np.isnan(df["open"].values)
    o = np.where(fill, c, df["open"].values)
    h = np.where(fill, c, df["high"].values)
    l = np.where(fill, c, df["low"].values)
    v = np.nan_to_num(df["volume"].values.astype("float64"))
    tb = np.nan_to_num(df["taker_buy_base"].values.astype("float64"))
    return dict(ts=idx, o=o, h=h, l=l, c=c, v=v, tb=tb)


def lag(x, n):
    return np.r_[np.full(n, np.nan), x[:-n]]


def features(d):
    c = d["c"]
    lc = np.log(c)
    r1 = np.r_[0.0, np.diff(lc)]
    sd1 = pd.Series(r1).rolling(3600, min_periods=1800).std().values      # 1-second sd over the hour
    f = dict(sd1=sd1, sd60=sd1 * np.sqrt(60))
    for n in (5, 10, 20):
        f["r%d" % n] = lc - lag(lc, n)
    v10 = pd.Series(d["v"]).rolling(10).sum().values
    f["v10"] = v10
    f["v10med"] = pd.Series(v10).shift(10).rolling(3600, min_periods=1800).median().values
    f["tb10"] = pd.Series(d["tb"]).rolling(10).sum().values / np.where(v10 > 0, v10, np.nan)
    return f


def signals(f, btc):
    s = {}
    for thr in (0.0015, 0.003):
        move = np.where(f["r20"] >= thr, 1, np.where(f["r20"] <= -thr, -1, 0))
        s["S1 follow %dbp/20s" % (thr * 1e4)] = move
        s["S2 fade %dbp/20s" % (thr * 1e4)] = -move
    z10 = f["r10"] / (f["sd1"] * np.sqrt(10))
    burst = f["v10"] >= 5 * f["v10med"]
    s["S3 volume burst"] = np.where(burst & (z10 >= 2), 1, np.where(burst & (z10 <= -2), -1, 0))
    busy = f["v10"] >= 3 * f["v10med"]
    s["S4 taker flow"] = np.where(busy & (f["tb10"] >= 0.8), 1, np.where(busy & (f["tb10"] <= 0.2), -1, 0))
    if btc is not None:
        calm = np.abs(z10) < 1
        s["S5 BTC leads"] = np.where((btc >= 3) & calm, 1, np.where((btc <= -3) & calm, -1, 0))
    z5 = f["r5"] / (f["sd1"] * np.sqrt(5))
    s["S6 spike fade"] = np.where(z5 <= -4, 1, np.where(z5 >= 4, -1, 0))
    return s, z10


def entries(sig, side, ok):
    """Signal seconds for one side, at most one per COOLDOWN seconds."""
    cand = np.flatnonzero((sig == side) & ok)
    out, nxt = [], -1
    for i in cand:
        if i >= nxt:
            out.append(i)
            nxt = i + COOLDOWN
    return np.array(out, dtype=np.int64)


def simulate(d, i, side, tgt, stp, half):
    """Trades from signal seconds i: in at i+1's close, out by i+1+HOLD. Returns gross, net."""
    n = len(d["c"])
    i = i[i + 1 + HOLD < n]
    e = i + 1
    p0 = d["c"][e]
    ok = np.isfinite(p0) & np.isfinite(tgt[i]) & np.isfinite(stp[i])
    e, i = e[ok], i[ok]
    p0 = d["c"][e]
    target = p0 * (1 + side * tgt[i])
    stop = p0 * (1 - side * stp[i])
    out = np.full(len(e), np.nan)
    how = np.zeros(len(e), dtype=np.int8)          # 1 target, 2 stop, 3 time
    for k in range(1, HOLD + 1):
        open_ = how == 0
        if not open_.any():
            break
        hi, lo = d["h"][e + k], d["l"][e + k]
        st = open_ & ((lo <= stop) if side > 0 else (hi >= stop))
        tg = open_ & ~st & ((hi > target) if side > 0 else (lo < target))
        out[st], how[st] = stop[st], 2
        out[tg], how[tg] = target[tg], 1
    t = how == 0
    out[t], how[t] = d["c"][e[t] + HOLD], 3
    if side > 0:
        entry = p0 * (1 + half)
        exit_ = np.where(how == 1, out, out * (1 - half))
        gross = exit_ / entry - 1
        fee = TAKER + np.where(how == 1, MAKER, TAKER)
    else:
        entry = p0 * (1 - half)
        exit_ = out * (1 + half)
        gross = 1 - exit_ / entry
        fee = 2 * SHORT_FEE
    return gross, gross - fee, how


def main() -> None:
    warnings.filterwarnings("ignore")
    spread = spreads()
    symbols = sorted({os.path.basename(p).rsplit("-", 2)[0] for p in glob.glob(os.path.join(DATA, "*.parquet"))})
    res = {}
    for month in MONTHS:
        b = load("BTCUSDT", month)
        btc_z = None
        if b is not None:
            fb = features(b)
            btc_z = fb["r10"] / (fb["sd1"] * np.sqrt(10))
        for sym in symbols:
            d = load(sym, month)
            if d is None:
                continue
            f = features(d)
            sigs, _ = signals(f, None if sym == "BTCUSDT" else btc_z)
            ok = np.isfinite(f["sd1"]) & (f["sd1"] > 0)
            exits = {"fixed 25/20bp": (np.full(len(d["c"]), 0.0025), np.full(len(d["c"]), 0.002)),
                     "scaled 1.5/2sd": (1.5 * f["sd60"], 2 * f["sd60"])}
            for name, sig in sigs.items():
                for side in (1, -1):
                    i = entries(sig, side, ok)
                    if not len(i):
                        continue
                    for ex, (tgt, stp) in exits.items():
                        for mult in (1, 2):
                            half = max(spread.get(sym, 0.002) / 2, 0.00005) * mult
                            g, n, how = simulate(d, i, side, tgt, stp, half)
                            res.setdefault((name, ex, side, mult), []).append(
                                (month, sym, g, n, how, spread.get(sym, 0.002)))
            print("  done", month, sym, flush=True)

    print("\nH127 scalps second by second, %d pairs, %s. Net bp per trade by month" % (len(symbols), " ".join(MONTHS)))
    passed = []
    for key in sorted(k for k in res if k[3] == 1):
        name, ex, side, _ = key
        verdicts = []
        for mult in (1, 2):
            rows = res[(name, ex, side, mult)]
            cells, good, have = [], 0, 0
            for month in MONTHS:
                x = np.concatenate([r[3] for r in rows if r[0] == month] or [np.array([])])
                if len(x) < 20:
                    cells.append("     --")
                    continue
                have += 1
                good += x.mean() > 0
                cells.append("%+7.1f" % (1e4 * x.mean()))
            net = np.concatenate([r[3] for r in rows])
            ok = have == len(MONTHS) and good == have and net.mean() > 0
            verdicts.append(ok)
            if mult == 1:
                gross = np.concatenate([r[2] for r in rows])
                how = np.concatenate([r[4] for r in rows])
                sp = np.concatenate([np.full(len(r[3]), r[5]) for r in rows])
                tiers = " ".join("%s %+.1f" % (lab, 1e4 * net[sel].mean()) if sel.sum() else "%s --" % lab
                                 for lab, sel in (("tight", sp < 0.0002), ("mid", (sp >= 0.0002) & (sp < 0.0008)),
                                                  ("wide", sp >= 0.0008)))
                print("  %-20s %-14s %-5s %s | %6d trades (%5.0f/day) target %3.0f%% | bp gross %+6.1f net %+6.1f"
                      " | %s | months %d/%d" % (name, ex, "long" if side > 0 else "short", " ".join(cells), len(net),
                                                len(net) / 92.0, 100 * (how == 1).mean(), 1e4 * gross.mean(),
                                                1e4 * net.mean(), tiers, good, have), flush=True)
        if all(verdicts):
            passed.append((name, ex, "long" if side > 0 else "short"))
    print("Passed (and with twice the spread):", passed or "none")


if __name__ == "__main__":
    main()
