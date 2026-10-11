"""H131: refining H130's nearest miss, buying altcoins at the close of a BTC capitulation day.

H130 searched 1,650 designs for a win rate of 60% or more with average wins at least 1.5 times the
average loss; none passed in development. Its nearest miss, buying every pair at the close of a day
BTC fell 6% or more (out at +1 ATR, a stop 2 ATRs down, or after 2 days), won 58.0% with a ratio of
1.26 in development and 57.2% with 1.80 in validation. This round refines that family only, with
the same protocol (develop October 2020 to October 2024, validate October 2024 to October 2026, the
same costs and pass marks), and adds a holdout no search has touched, October 2018 to October 2020
(the pairs listed then), which a design must also pass.

Fixed before running:
  crash: BTC down 5, 6, 7 or 8% on the day
  filters on the pair, each alone: none; closed in the upper half of its day's range (buyers came
     back); its own uptrend (close above the 200-day average); BTC above its 200-day average (a
     crash in a bull market); funding over the last 3 days below zero (shorts crowded); among the
     5 pairs that fell most that day; among the 5 that fell least
  exits: targets 0.5, 1 and 1.5 ATRs, stops 1.5, 2 and 3 ATRs, held 1, 2 or 3 days; and, instead
     of a target, out at the first hourly close above the entry (a stop as above)
Trades on the same day are not independent, so the number of distinct crash days is reported too.

    python -m research.h131_capitulation
"""
import warnings

import numpy as np
import pandas as pd

from research import h130_high_win_rate as h

HOLDOUT = ("2018-10-01", "2020-10-01")
TARGETS = (0.5, 1.0, 1.5, None)          # None: the first hourly close above the entry
STOPS = (1.5, 2.0, 3.0)
HOLDS = (1, 2, 3)
h.MAX_H = 3 * 24


def build():
    spreads = h.pairs_and_spreads()
    btc_h = h.load_hourly("BTC/USD")
    btc = h.daily_bars(btc_h)
    btc_ret = btc["c"].pct_change()
    btc_up = btc["c"] > btc["c"].rolling(200).mean()
    coins, day_ret = {}, {}
    for pair, spread in spreads.items():
        hr = h.load_hourly(pair)
        if hr is None or len(hr) < 24 * 260:
            continue
        d = h.daily_bars(hr)
        prev = d["c"].shift(1)
        tr = np.maximum(d["h"] - d["l"], np.maximum((d["h"] - prev).abs(), (d["l"] - prev).abs()))
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        fund = h.funding_daily(pair.split("/")[0], d.index.values)
        nxt = d["first"].shift(-1)
        ok = nxt.notna() & (d.index.to_series().shift(-1) - d.index.to_series() == 86_400_000)
        r = d["c"].pct_change()
        day_ret[pair] = r
        coins[pair] = dict(h=hr, d=d, days=d.index.values, atr=atr.values, entry_row=nxt.values, ok=ok.values,
                           half=max(spread / 2, 0.00005), fund=fund, r=r,
                           upper=((d["c"] - d["l"]) / (d["h"] - d["l"]).replace(0, np.nan) >= 0.5),
                           up=d["c"] > d["c"].rolling(200).mean())
    rets = pd.DataFrame(day_ret)
    rank_fall = rets.rank(axis=1, ascending=True)          # 1 = fell most
    rank_hold = rets.rank(axis=1, ascending=False)         # 1 = fell least
    for pair, cn in coins.items():
        idx = cn["d"].index
        for x in (0.05, 0.06, 0.07, 0.08):
            crash = (btc_ret.reindex(idx) <= -x).fillna(False)
            filters = {
                "none": crash,
                "closed upper half": crash & cn["upper"].fillna(False),
                "own uptrend": crash & cn["up"].fillna(False),
                "BTC bull market": crash & btc_up.reindex(idx).fillna(False),
                "funding negative": crash & (cn["fund"] < 0).fillna(False),
                "fell most (top 5)": crash & (rank_fall[pair].reindex(idx) <= 5).fillna(False),
                "fell least (top 5)": crash & (rank_hold[pair].reindex(idx) <= 5).fillna(False),
            }
            for f, s in filters.items():
                cn.setdefault("sig", {})[("BTC -%d%%, %s" % (100 * x, f), 1)] = s.values
    return coins


def evaluate(coins, name):
    out = {}
    for pair, cn in coins.items():
        sig = cn["sig"].get((name, 1))
        idx = np.flatnonzero(sig & cn["ok"] & np.isfinite(cn["atr"]) & (cn["atr"] > 0))
        if not len(idx):
            continue
        p0, H, L, C, ts = h.paths(cn, idx)
        atr = cn["atr"][idx] / p0
        half = cn["half"]
        entry = p0 * (1 + half)
        for tm in TARGETS:
            for sm in STOPS:
                stp = p0 * (1 - sm * atr)
                if tm is None:
                    # the first hourly close above the entry (net of the costs to get out at market)
                    hit_t = C * (1 - half) * (1 - h.TAKER) > entry[:, None] * (1 + h.TAKER)
                    tgt = None
                else:
                    tgt = p0 * (1 + tm * atr)
                    hit_t = H >= tgt[:, None]
                hit_s = L <= stp[:, None]
                ft = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 6)
                fs = np.where(hit_s.any(1), hit_s.argmax(1), 10 ** 6)
                for hold in HOLDS:
                    last = hold * 24 - 1
                    stop_first = (fs <= ft) & (fs <= last)
                    target_first = (ft < fs) & (ft <= last)
                    end_close = C[:, last]
                    finished = np.isfinite(end_close) | stop_first | target_first
                    if tm is None:
                        at_t = C[np.arange(len(idx)), np.minimum(ft, h.MAX_H - 1)]
                        exit_px = np.where(stop_first, stp, np.where(target_first, at_t, end_close))
                        out_px = exit_px * (1 - half)
                        fee_out = h.TAKER
                    else:
                        exit_px = np.where(stop_first, stp, np.where(target_first, tgt, end_close))
                        out_px = np.where(target_first, exit_px, exit_px * (1 - half))
                        fee_out = np.where(target_first, h.MAKER, h.TAKER)
                    gross = out_px / entry - 1
                    net = gross - h.TAKER - fee_out
                    exit_at = np.where(stop_first, fs, np.where(target_first, ft, last))
                    free, keep = -1, []
                    for i in range(len(idx)):
                        if finished[i] and ts[i] >= free:
                            keep.append(i)
                            free = ts[i] + (exit_at[i] + 1) * 3_600_000
                    keep = np.array(keep, dtype=int)
                    if len(keep):
                        out.setdefault((tm, sm, hold), []).append(np.c_[ts[keep], net[keep], gross[keep]])
    return {k: np.vstack(v) for k, v in out.items()}


def days(a, start, end):
    sel = (a[:, 0] >= h.ms(start)) & (a[:, 0] < h.ms(end))
    return len(np.unique(a[sel, 0] // 86_400_000))


def main() -> None:
    warnings.filterwarnings("ignore")
    coins = build()
    names = sorted({k[0] for cn in coins.values() for k in cn["sig"]})
    rows = []
    for name in names:
        for exit_, a in evaluate(coins, name).items():
            rows.append((name, exit_, a, h.stats(a, *h.DEV), h.stats(a, *h.VAL), h.stats(a, *HOLDOUT),
                         h.stats(a, *h.COMP)))
    print("H131: %d designs" % len(rows))
    dev = [r for r in rows if h.meets(r[3], 100)]
    val = [r for r in dev if h.meets(r[4], 40)]
    hold = [r for r in val if h.meets(r[5], 30)]
    print("meet the brief: development %d, and validation %d, and the 2018-20 holdout %d" % (len(dev), len(val), len(hold)))

    def show(r):
        name, (tm, sm, hd), a, dv, vl, ho, cp = r
        t = "first close up" if tm is None else "target %.1f" % tm
        print("  %s | %s, stop %.1f ATR, %dd\n     dev  %s (%d crash days)\n     val  %s (%d)\n     2018-20 %s (%d)\n     comp %s"
              % (name, t, sm, hd, h.fmt(dv), days(a, *h.DEV), h.fmt(vl), days(a, *h.VAL), h.fmt(ho),
                 days(a, *HOLDOUT), h.fmt(cp)))
    print("\nPassing development and validation:")
    for r in sorted(val, key=lambda r: -r[4]["mean"]):
        show(r)
    print("\nNearest in development (60+ trades), whether or not they pass:")
    near = sorted([r for r in rows if r[3]["n"] >= 60 and r[3]["mean"] > 0],
                  key=lambda r: -(min(r[3]["wr"] / h.MIN_WR, 1.2) * min(r[3]["ratio"] / h.MIN_RATIO, 1.2)))[:15]
    for r in near:
        show(r)


if __name__ == "__main__":
    main()
