"""H88: Williams %R, long and short, on hourly bars (the user's request).

Williams %R(n) = -100 x (highest high of n bars - close) / (highest high - lowest low): 0 is a
close at the top of the range, -100 at the bottom. Hourly bars, as the bot runs, on the bot's
coins, 2020-26. Written down before running, each long and short, one position per coin:

  WR1 pullback in a trend: with the bot's trend up (EMA240 above EMA960), %R(14) below -80 buys;
      out when %R rises above -20, after 48 hours, or 3 ATRs against. Shorts mirror it in a
      downtrend (above -20 shorts, out below -80)
  WR2 breakout momentum: %R(14) crossing up through -20 with the trend up buys (strength, as
      Williams also used it); out when %R falls below -50 or after 72 hours, 3 ATR stop. Mirror
  WR3 plain reversal: %R(14) below -95 buys, above -5 shorts, no trend filter; out at -50,
      after 24 hours, 3 ATR stop
  WR4 two timeframes: the 14-day %R (336 hours) above -20 (near its high) and the hourly %R(14)
      below -80 (a pullback) buys; out when the hourly %R is above -20 or after 72 hours, 3 ATR
      stop. Mirror: 14-day below -80 and hourly above -20 shorts
  WR5 the slow signal alone: the 14-day %R crossing up through -20 buys and holds while it
      stays above -50 (at most 14 days, 4 ATR stop); crossing down through -80 shorts while it
      stays below -50

Neighbours: lookbacks 10 and 21 in place of 14 (WR4 and WR5: 10 and 21 days). Costs and the test
as H86 (fee and half the spread a side): the mean net return per trade positive in at least 5 of
the 6 folds and over all, and both neighbours too. Passing designs then go into the bot.

    python -m research.h88_williams_r
"""
import warnings

import numpy as np
import pandas as pd

from research.h60_swing_mft import costs, load_1h
from research.h86_scalping import STARTS, YEARS, engine
from research.h87_orb import score


def wr(D, n):
    hh, ll = D.h.rolling(n).max(), D.l.rolling(n).min()
    return -100 * (hh - D.c) / (hh - ll).replace(0, np.nan)


def designs(D, n):
    c, h, l = D.c, D.h, D.l
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    up = c.ewm(span=240, adjust=False).mean() > c.ewm(span=960, adjust=False).mean()
    r = wr(D, n)
    slow = wr(D, n * 24)
    nan = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    out = {}
    out["WR1 pullback in a trend"] = dict(
        entry=((r < -80) & up).astype(int) - ((r > -20) & ~up).astype(int), bars=48, lx=r > -20, sx=r < -80)
    cross_up, cross_down = (r > -20) & (r.shift() <= -20), (r < -80) & (r.shift() >= -80)
    out["WR2 breakout momentum"] = dict(
        entry=(cross_up & up).astype(int) - (cross_down & ~up).astype(int), bars=72, lx=r < -50, sx=r > -50)
    out["WR3 plain reversal"] = dict(
        entry=(r < -95).astype(int) - (r > -5).astype(int), bars=24, lx=r > -50, sx=r < -50)
    out["WR4 two timeframes"] = dict(
        entry=((slow > -20) & (r < -80)).astype(int) - ((slow < -80) & (r > -20)).astype(int), bars=72,
        lx=r > -20, sx=r < -80)
    s_up, s_down = (slow > -20) & (slow.shift() <= -20), (slow < -80) & (slow.shift() >= -80)
    out["WR5 the slow signal alone"] = dict(
        entry=s_up.astype(int) - s_down.astype(int), bars=336, lx=slow < -50, sx=slow > -50, atr_mult=4)
    for spec in out.values():
        spec["stop"] = spec.pop("atr_mult", 3) * atr
        spec["target"] = nan
    return out


def run(D, spec, cost):
    stamps = D.idx.asi8
    out = []
    for pair in D.c.columns:
        c = D.c[pair].values
        if np.isfinite(c).sum() < 24 * 120:
            continue
        entry = spec["entry"][pair].fillna(0).values.astype(int)
        trades = []
        for side, ex in ((1, spec["lx"]), (-1, spec["sx"])):            # each side leaves on its own rule
            trades += engine(c, D.h[pair].values, D.l[pair].values, np.where(entry == side, side, 0),
                             spec["stop"][pair].values, spec["target"][pair].values, spec["bars"],
                             ex[pair].fillna(False).values.astype(bool))
        for i, side, g in trades:
            k = np.searchsorted(STARTS, stamps[i], side="right") - 1
            if 0 <= k < len(YEARS):
                m = g - 2 * cost[pair]
                out.append((k, side, g, m, m))
    return np.array(out).reshape(-1, 5)


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_1h()
    cost = costs(list(D.c.columns))
    print("H88 Williams %%R, hourly bars, %d coins. Net basis points per trade by fold %s" % (
        D.c.shape[1], " ".join(YEARS)), flush=True)
    results = {}
    for n in (14, 10, 21):
        for name, spec in designs(D, n).items():
            a = run(D, spec, cost)
            for side, label in ((1, "long"), (-1, "short")):
                s = score(a, side)
                results[(name, label, n)] = s
                print("  %-26s %-5s n=%-2d %s | %6d trades win %3.0f%% | bp gross %+6.1f net %+6.1f | folds %d/%d%s" % (
                    name, label, n, " ".join("   --" if x != x else "%+5.0f" % x for x in s["per"]), s["n"],
                    s["win"], s["gross"], s["net"], s["good"], s["have"], "  pass" if s["ok"] else ""), flush=True)
    print("\nVerdicts (the design and both neighbours must pass):")
    for name, label, n in results:
        if n == 14:
            ok = all(results[(name, label, m)]["ok"] for m in (14, 10, 21))
            print("  %-26s %-5s %s" % (name, label, "PASS" if ok else "fail"))


if __name__ == "__main__":
    main()
