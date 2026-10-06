"""H89: the RSI(3) + EMA + VWAP dip-buy (a published QQQ strategy the user sent), on crypto.

The published rules, for QQQ on daily bars: RSI(3) closes below 22; the 100-day EMA is above
its level 5 days ago; the close is 1% or more below the 10-day rolling VWAP; buy at the next
session's open; sell when RSI(3) closes above 55; one position at a time. Crypto trades round
the clock, so the next open is this close. Written down before running:

  R1 as published, on daily bars of the bot's coins (one position per coin)
  R2 the mirror for shorts: RSI(3) above 78, the 100-day EMA below its level 5 days ago, the
     close 1% or more above the VWAP; cover when RSI(3) closes below 45
  R3 and R4: R1 and R2 on 4-hour bars (the EMA, VWAP and shift counted in bars)
  R5 and R6: R1 and R2 on hourly bars, as the bot runs

No stop (as published), at most 60 bars a trade. Neighbours: RSI thresholds 18 and 26 (shorts:
74, 82); VWAP gaps 0.5% and 2%. Costs and the test as H86: the mean net return per trade
positive in at least 5 of the 6 folds and over all, and all four neighbours too. Our QQQ file
has closes only (no volume or opens), so the original is not re-run on QQQ.

    python -m research.h89_rsi3_vwap
"""
import warnings

import numpy as np
import pandas as pd

from research.h60_swing_mft import costs, load_1h
from research.h86_scalping import STARTS, YEARS, engine
from research.h87_orb import score


class Bars:
    def __init__(self, D, rule):
        if rule == "1h":
            self.c, self.h, self.l, self.v = D.c, D.h, D.l, D.vol
        else:
            self.c = D.c.resample(rule).last()
            self.h = D.h.resample(rule).max()
            self.l = D.l.resample(rule).min()
            self.v = D.vol.resample(rule).sum(min_count=1)
        self.idx = self.c.index


def rsi(x, n):
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def spec(B, side, level=22.0, gap=0.01):
    c = B.c
    r = rsi(c, 3)
    ema = c.ewm(span=100, adjust=False, min_periods=100).mean()
    v = B.v.replace(0, np.nan)
    vwap = ((B.h + B.l + c) / 3 * v).rolling(10).sum() / v.rolling(10).sum()
    if side > 0:
        entry = (r < level) & (ema > ema.shift(5)) & (c <= vwap * (1 - gap))
        leave = r > 55
    else:
        entry = (r > 100 - level) & (ema < ema.shift(5)) & (c >= vwap * (1 + gap))
        leave = r < 45
    nan = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    return dict(entry=entry.astype(int) * side, leave=leave, stop=nan, target=nan)


def run(B, sp, cost):
    stamps = B.idx.asi8
    out = []
    for pair in B.c.columns:
        c = B.c[pair].values
        if np.isfinite(c).sum() < 120:
            continue
        trades = engine(c, B.h[pair].values, B.l[pair].values, sp["entry"][pair].fillna(0).values.astype(int),
                        sp["stop"][pair].values, sp["target"][pair].values, 60,
                        sp["leave"][pair].fillna(False).values.astype(bool))
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
    print("H89 RSI(3) + EMA + VWAP, %d coins. Net basis points per trade by fold %s" % (D.c.shape[1], " ".join(YEARS)))
    variants = [("as published", 22.0, 0.01), ("RSI 18", 18.0, 0.01), ("RSI 26", 26.0, 0.01),
                ("gap 0.5%", 22.0, 0.005), ("gap 2%", 22.0, 0.02)]
    for label, rule in (("daily", "1D"), ("4-hour", "4h"), ("hourly", "1h")):
        B = Bars(D, rule)
        for side, name in ((1, "long"), (-1, "short")):
            oks = []
            for vname, level, gap in variants:
                s = score(run(B, spec(B, side, level, gap), cost), side)
                oks.append(s["ok"])
                print("  %-6s %-5s %-12s %s | %6d trades win %3.0f%% | bp gross %+7.1f net %+7.1f | folds %d/%d" % (
                    label, name, vname, " ".join("    --" if x != x else "%+6.0f" % x for x in s["per"]), s["n"],
                    s["win"], s["gross"], s["net"], s["good"], s["have"]), flush=True)
            print("  -> %s %s: %s" % (label, name, "PASS" if all(oks) else "fail"), flush=True)


if __name__ == "__main__":
    main()
