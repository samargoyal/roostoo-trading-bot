"""H87: opening-range breakouts (ORB), long and short, on 5-minute bars (the user's request).

H86's one ORB (the US open, 30 minutes, a 2R target) lost about 29 basis points a trade. This
is the full grid, written down before running. A session opens; its first minutes set a range;
the first close above the range buys and the first close below shorts (one trade at a time per
coin; after a stop, the opposite break may still trade that day):

  sessions  Asia 00:00 UTC, London 08:00, US 13:30, each held to 8 hours after the open;
            and the UTC day, 00:00, held to the next 00:00
  range     the first 15, 30 or 60 minutes (the UTC day: 60, 120 or 240)
  exits     2R: stop at the range's other side, target twice the range from the entry
            hold: stop at the other side, no target, out at the session's end
            tight: stop at the range's middle, no target, out at the session's end
  filters   none; trend: longs only while the 1-day EMA is above the 4-day, shorts only below;
            volume: the breakout bar trades twice the usual volume for that time of day;
            hourly: the bot as it runs, acting only at hour closes (60-minute ranges and longer)

That is 4 x 3 x 3 x 4 = 144 designs a side (fewer for hourly). Costs and the test as H86: the
mean net return per trade at market positive in at least 5 of the 6 folds and over all, and the
design's neighbours (the other range lengths, same everything else) must mostly pass too. With
so many designs, a few passing by chance alone is expected.

    python -m research.h87_orb
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from research.h60_swing_mft import costs, load_5m
from research.h86_scalping import FEE_LIMIT, STARTS, YEARS, engine

SESSIONS = {"Asia": (0, 480, (15, 30, 60)), "London": (480, 960, (15, 30, 60)),
            "US": (810, 1290, (15, 30, 60)), "UTC day": (0, 1440, (60, 120, 240))}
EXITS = ("2R", "hold", "tight")
FILTERS = ("none", "trend", "volume", "hourly")


def prepare(D):
    c, qv = D.c, D.qv
    t = c.index
    mod = pd.Series(t.hour * 60 + t.minute, index=t)
    slot = mod.values // 5
    med = qv.groupby(slot).transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    trend = c.ewm(span=288, adjust=False).mean() > c.ewm(span=1152, adjust=False).mean()
    return dict(mod=mod.values, day=t.normalize(), rvol=qv / med, trend=trend, hour_close=(t.minute == 55))


def wide(mask, like):
    return pd.DataFrame(np.repeat(np.asarray(mask)[:, None], like.shape[1], axis=1), index=like.index,
                        columns=like.columns)


def design(D, P, session, minutes, exit_kind, filt):
    c, h, l = D.c, D.h, D.l
    start, end, _ = SESSIONS[session]
    mod, day = P["mod"], P["day"]
    in_range = wide((mod >= start) & (mod < start + minutes), c)
    hi = h.where(in_range).groupby(day).transform("max")
    lo = l.where(in_range).groupby(day).transform("min")
    window = wide((mod >= start + minutes) & (mod < end), c)
    up, down = (c > hi) & window, (c < lo) & window
    if filt == "trend":
        up, down = up & P["trend"], down & ~P["trend"]
    elif filt == "volume":
        up, down = up & (P["rvol"] >= 2), down & (P["rvol"] >= 2)
    elif filt == "hourly":
        hc = wide(P["hour_close"], c)
        up, down = up & hc, down & hc
    up = up & (up.astype(int).groupby(day).cumsum() == 1)
    down = down & (down.astype(int).groupby(day).cumsum() == 1)
    entry = up.astype(int) - down.astype(int)
    width = (hi - lo) / c
    if exit_kind == "tight":
        mid = (hi + lo) / 2
        stop = ((c - mid) / c).where(entry >= 0, (mid - c) / c)
    else:
        stop = ((c - lo) / c).where(entry >= 0, (hi - c) / c)
    target = 2 * width if exit_kind == "2R" else pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    # out at the session's end; for the UTC day, at the next day's range (00:00)
    leave = (mod >= end) | (mod < start) if end < 1440 else (mod < start + minutes)
    return dict(entry=entry, stop=stop, target=target, bars=(end - start) // 5, exit_signal=wide(leave, c))


def run(D, spec, cost):
    stamps = D.idx.asi8
    out = []
    for pair in D.c.columns:
        if D.c[pair].notna().sum() < 288 * 60:
            continue
        trades = engine(D.c[pair].values, D.h[pair].values, D.l[pair].values,
                        spec["entry"][pair].fillna(0).values.astype(int), spec["stop"][pair].values,
                        spec["target"][pair].values, spec["bars"], spec["exit_signal"][pair].values.astype(bool))
        for i, side, g in trades:
            k = np.searchsorted(STARTS, stamps[i], side="right") - 1
            if 0 <= k < len(YEARS):
                m = g - 2 * cost[pair]
                out.append((k, side, g, m, g - 2 * FEE_LIMIT if side > 0 else m))
    return np.array(out).reshape(-1, 5)


def score(a, side):
    a = a[a[:, 1] == side]
    folds = [a[a[:, 0] == k] for k in range(len(YEARS))]
    per = [f[:, 3].mean() * 1e4 if len(f) >= 20 else np.nan for f in folds]
    have = sum(1 for x in per if x == x)
    good = sum(1 for x in per if x == x and x > 0)
    ok = have > 0 and good >= min(5, have) and len(a) and a[:, 3].mean() > 0
    return dict(per=per, n=len(a), gross=a[:, 2].mean() * 1e4 if len(a) else np.nan,
                net=a[:, 3].mean() * 1e4 if len(a) else np.nan, limit=a[:, 4].mean() * 1e4 if len(a) else np.nan,
                win=(a[:, 3] > 0).mean() * 100 if len(a) else np.nan, good=good, have=have, ok=bool(ok))


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_5m()
    P = prepare(D)
    cost = costs(list(D.c.columns))
    rows = []
    for session, (_, _, ranges) in SESSIONS.items():
        for minutes, exit_kind, filt in itertools.product(ranges, EXITS, FILTERS):
            if filt == "hourly" and minutes < 60:
                continue
            a = run(D, design(D, P, session, minutes, exit_kind, filt), cost)
            for side, label in ((1, "long"), (-1, "short")):
                s = score(a, side)
                rows.append(dict(session=session, minutes=minutes, exit=exit_kind, filter=filt, side=label, **s))
                print("  %-7s %3dm %-5s %-6s %-5s %s | %6d trades win %3.0f%% | bp gross %+6.1f market %+6.1f "
                      "limit %+6.1f | folds %d/%d%s" % (
                          session, minutes, exit_kind, filt, label,
                          " ".join("   --" if x != x else "%+5.0f" % x for x in s["per"]), s["n"], s["win"],
                          s["gross"], s["net"], s["limit"], s["good"], s["have"], "  PASS" if s["ok"] else ""),
                      flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["per"]).to_csv("runs/research/h87_orb.csv", index=False)
    print("\nPassed (before neighbours): %d of %d" % (df.ok.sum(), len(df)))
    for (session, exit_kind, filt, side), g in df.groupby(["session", "exit", "filter", "side"]):
        if g.ok.any():
            print("  %s %s %s %s: %d of %d range lengths pass" % (session, exit_kind, filt, side, g.ok.sum(), len(g)))
    print("\nBest 10 by net per trade:")
    print(df.sort_values("net", ascending=False).head(10)[
        ["session", "minutes", "exit", "filter", "side", "n", "gross", "net", "limit", "good", "have"]].to_string())


if __name__ == "__main__":
    main()
