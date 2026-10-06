"""H90: order-flow concepts (a list of 62 the user sent), alone and combined, long and short.

Binance bars carry each hour's taker-buy volume (aggressive buyers; the rest of the volume is
aggressive sellers) and trade count, so delta, cumulative delta, bar and session delta, their
divergences, absorption, exhaustion, average trade size (large orders), volume profiles (POC,
value area), VWAPs and price structure can be built. The concepts that need the order book or
prices tick by tick cannot be tested with any data we have: footprint charts, bid x ask volume,
diagonal and stacked imbalances, DOM, level 2, the order book, iceberg orders, hidden liquidity,
spoofing, liquidity pulling and stacking, and limit and stop orders as such. The initial balance
breakout is H87's UTC-day ORB.

Hourly bars (the bot's cadence) of 49 coins, 2020-26. Every signal acts at an hour's close and
exits alike, so they compare and combine cleanly: held 24 hours, with a stop 3 ATRs away.
Written down before running (z: against the last 30 days; delta = taker buys - taker sells):

  OF01 bar delta (13, 3)            delta / volume z above 2 buys (aggressive buyers), below -2 shorts
  OF02 CVD trend (4)                24-hour delta / volume z above 1.5 buys, below -1.5 shorts
  OF03 CVD divergence (50, 52)      close at its 24-hour high with negative 24-hour delta shorts;
                                    at its low with positive delta buys
  OF04 absorption at levels (8, 53) at the 7-day low, 2x volume, close in the bar's top half and
                                    negative delta (selling absorbed) buys; mirror at the high
  OF05 exhaustion (9)               3x volume, return z beyond 2.5 and the close back in the
                                    opposite half of the bar: fade it
  OF06 session delta (14)           at 08:00 UTC, Asia's delta / volume z above 1 buys, below -1 shorts
  OF07 large orders (20)            average trade size z above 2: follow the bar's delta
  OF08 price-volume divergence (51) a 72-hour high on below-median 24-hour volume shorts; a low buys
  OF09 initiative (58)              close above the 7-day value area (70% of volume) after a bar
                                    inside it buys; below it shorts
  OF10 responsive (59)              back inside the value area from below buys (towards the POC);
                                    from above shorts
  OF11 developing POC (36, 38)      the last 24 hours' POC above the previous 24 hours' by an
                                    ATR or more buys; below shorts
  OF12 VWAP trend (47, 48)          close above the week's anchored VWAP (from Monday 00:00 UTC)
                                    with that VWAP rising over 24 hours buys; mirror shorts
  OF13 VWAP bands (49)              below the anchored VWAP's -2 sd band buys; above +2 sd shorts
  OF14 liquidity sweep (55, 56)     the low takes out the 48-hour low and the close is back above
                                    it buys (stops hunted); mirror shorts
  OF15 failed auction (57)          the same at the 7-day extremes
  OF16 trapped traders (54)         a close above the 48-hour high, then within 3 bars a close
                                    back below it shorts (longs trapped); mirror buys
  OF17 structure shift (60)         in a downtrend (EMA24 below EMA96), a close above the last
                                    swing high (a 5-bar fractal) buys; mirror shorts

Combinations (61 order-flow confirmation, 62 buying and selling pressure): a vote, entering when
at least 2 or at least 3 signals agree in the last 6 hours and none disagrees; and every pair of
signals, both firing the same way within 6 hours.

The test, as H86: the mean net return per trade (fee and half the spread a side) positive in at
least 5 of the 6 folds and over all, and the same with 12- and 48-hour holds (neighbours). With
17 signals and 136 pairs a side, a few will pass by chance; a pass is then re-run inside the bot.

    python -m research.h90_order_flow
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from research.h60_swing_mft import costs, load_1h
from research.h86_scalping import STARTS, YEARS, engine
from research.h87_orb import score

HOLDS = (24, 12, 48)


def z(x, n=720):
    return (x - x.rolling(n, min_periods=n // 3).mean()) / x.rolling(n, min_periods=n // 3).std()


def profile(tp, vol, window, buckets=30):
    """At each UTC day's start: POC, VAH and VAL of the previous `window` hours (typical
    prices weighted by volume), carried through the day."""
    out = {k: pd.DataFrame(np.nan, index=tp.index, columns=tp.columns) for k in ("poc", "vah", "val")}
    days = np.flatnonzero((tp.index.hour == 0))
    P, V = tp.values, vol.fillna(0).values
    for j in range(tp.shape[1]):
        poc = np.full(len(tp), np.nan)
        vah, val = poc.copy(), poc.copy()
        for d in days:
            if d < window:
                continue
            p, v = P[d - window:d, j], V[d - window:d, j]
            ok = np.isfinite(p) & (v > 0)
            if ok.sum() < window // 2:
                continue
            p, v = p[ok], v[ok]
            lo, hi = p.min(), p.max()
            if hi <= lo:
                continue
            edges = np.linspace(lo, hi, buckets + 1)
            k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, buckets - 1)
            hist = np.bincount(k, weights=v, minlength=buckets)
            mids = (edges[:-1] + edges[1:]) / 2
            order = np.argsort(-hist)
            keep = order[:np.searchsorted(np.cumsum(hist[order]), 0.7 * hist.sum()) + 1]
            end = min(d + 24, len(tp))
            poc[d:end], vah[d:end], val[d:end] = mids[order[0]], edges[keep.max() + 1], edges[keep.min()]
        out["poc"].iloc[:, j], out["vah"].iloc[:, j], out["val"].iloc[:, j] = poc, vah, val
    return out


def signals(D):
    c, h, l, v = D.c, D.h, D.l, D.vol.replace(0, np.nan)
    t = c.index
    delta = 2 * D.taker - D.vol
    dr = delta / v
    r = np.log(c).diff()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    rvol = v / v.rolling(720, min_periods=240).median()
    where = (c - l) / (h - l).replace(0, np.nan)                       # 1 = closed at the high
    S = {}

    def sig(long_, short_):
        return long_.fillna(False).astype(int) - short_.fillna(False).astype(int)

    zd = z(dr)
    S["OF01 bar delta"] = sig(zd > 2, zd < -2)
    cvd = delta.rolling(24).sum() / v.rolling(24).sum()
    zc = z(cvd)
    S["OF02 CVD trend"] = sig(zc > 1.5, zc < -1.5)
    hi24, lo24 = c.rolling(24).max(), c.rolling(24).min()
    S["OF03 CVD divergence"] = sig((c <= lo24) & (cvd > 0), (c >= hi24) & (cvd < 0))
    low7, high7 = l.rolling(168).min(), h.rolling(168).max()
    S["OF04 absorption at levels"] = sig((l <= low7) & (rvol >= 2) & (where >= 0.5) & (dr < 0),
                                         (h >= high7) & (rvol >= 2) & (where <= 0.5) & (dr > 0))
    zr = r / r.rolling(720, min_periods=240).std().shift()
    S["OF05 exhaustion"] = sig((rvol >= 3) & (zr <= -2.5) & (where >= 0.5), (rvol >= 3) & (zr >= 2.5) & (where <= 0.5))
    morning = pd.DataFrame(np.repeat(np.asarray(t.hour < 8)[:, None], c.shape[1], axis=1), index=t, columns=c.columns)
    asia = delta.where(morning).groupby(t.normalize()).transform("sum")
    asia_v = D.vol.where(morning).groupby(t.normalize()).transform("sum")
    at8 = np.asarray(t.hour == 7)                                      # the 07:00 bar closes at 08:00
    daily = (asia / asia_v)[at8]
    zs = ((daily - daily.rolling(30, min_periods=10).mean()) / daily.rolling(30, min_periods=10).std()).reindex(t)
    S["OF06 session delta"] = sig(zs > 1, zs < -1)
    size = D.qv / D.trades.replace(0, np.nan)
    big = z(size) > 2
    S["OF07 large orders"] = sig(big & (dr > 0), big & (dr < 0))
    quiet = v.rolling(24).sum() < v.rolling(24).sum().rolling(720, min_periods=240).median()
    S["OF08 price-volume divergence"] = sig((c <= c.rolling(72).min()) & quiet, (c >= c.rolling(72).max()) & quiet)
    tp = (h + l + c) / 3
    P7 = profile(tp, D.vol, 168)
    inside = (c <= P7["vah"]) & (c >= P7["val"])
    S["OF09 initiative"] = sig((c > P7["vah"]) & inside.shift(), (c < P7["val"]) & inside.shift())
    S["OF10 responsive"] = sig(inside & (c.shift() < P7["val"]), inside & (c.shift() > P7["vah"]))
    P1 = profile(tp, D.vol, 24)
    move = (P1["poc"] - P1["poc"].shift(24)) / atr
    first = pd.DataFrame(np.repeat(np.asarray(t.hour == 0)[:, None], c.shape[1], axis=1), index=t, columns=c.columns)
    S["OF11 developing POC"] = sig((move >= 1) & first, (move <= -1) & first)
    week = t.to_period("W-SUN")
    pv = (tp * D.vol).groupby(week).cumsum()
    vv = D.vol.groupby(week).cumsum()
    avwap = pv / vv.replace(0, np.nan)
    var = (tp * tp * D.vol).groupby(week).cumsum() / vv.replace(0, np.nan) - avwap ** 2
    sd = np.sqrt(var.clip(lower=0))
    rising = avwap > avwap.shift(24)
    S["OF12 VWAP trend"] = sig((c > avwap) & rising, (c < avwap) & ~rising)
    S["OF13 VWAP bands"] = sig(c < avwap - 2 * sd, c > avwap + 2 * sd)
    pl48, ph48 = l.shift().rolling(48).min(), h.shift().rolling(48).max()
    S["OF14 liquidity sweep"] = sig((l < pl48) & (c > pl48), (h > ph48) & (c < ph48))
    pl7, ph7 = l.shift().rolling(168).min(), h.shift().rolling(168).max()
    S["OF15 failed auction"] = sig((l < pl7) & (c > pl7), (h > ph7) & (c < ph7))
    brk_up = (c > ph48)
    brk_dn = (c < pl48)
    level_up = ph48.where(brk_up).ffill(limit=3)
    level_dn = pl48.where(brk_dn).ffill(limit=3)
    recent_up = brk_up.rolling(3).max().shift().fillna(0) > 0
    recent_dn = brk_dn.rolling(3).max().shift().fillna(0) > 0
    S["OF16 trapped traders"] = sig(recent_dn & (c > level_dn), recent_up & (c < level_up))
    swing_hi = h.shift(2).where(h.shift(2) == h.rolling(5).max()).ffill()   # a 5-bar fractal, known 2 bars later
    swing_lo = l.shift(2).where(l.shift(2) == l.rolling(5).min()).ffill()
    down = c.ewm(span=24, adjust=False).mean() < c.ewm(span=96, adjust=False).mean()
    S["OF17 structure shift"] = sig(down & (c > swing_hi) & (c.shift() <= swing_hi.shift()),
                                    ~down & (c < swing_lo) & (c.shift() >= swing_lo.shift()))
    return S, atr / c


def run(D, entry, stop, hold, cost, side):
    stamps = D.idx.asi8
    out = []
    nan = np.full(len(D.idx), np.nan)
    for pair in D.c.columns:
        c = D.c[pair].values
        if np.isfinite(c).sum() < 24 * 120:
            continue
        e = entry[pair].fillna(0).values.astype(int)
        e = np.where(e == side, side, 0)
        for i, s, g in engine(c, D.h[pair].values, D.l[pair].values, e, 3 * stop[pair].values, nan, hold):
            k = np.searchsorted(STARTS, stamps[i], side="right") - 1
            if 0 <= k < len(YEARS):
                m = g - 2 * cost[pair]
                out.append((k, s, g, m, m))
    return np.array(out).reshape(-1, 5)


def judge(D, entry, stop, cost, side):
    """(score with a 24-hour hold, passes with all three holds)."""
    scores = [score(run(D, entry, stop, h, cost, side), side) for h in HOLDS]
    return scores[0], all(s["ok"] for s in scores)


def line(name, label, s, ok):
    return "  %-34s %-5s %s | %6d trades win %3.0f%% | bp gross %+6.1f net %+6.1f | folds %d/%d%s" % (
        name, label, " ".join("   --" if x != x else "%+5.0f" % x for x in s["per"]), s["n"], s["win"],
        s["gross"], s["net"], s["good"], s["have"], "  PASS (all holds)" if ok else "")


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_1h()
    cost = costs(list(D.c.columns))
    S, stop = signals(D)
    print("H90 order flow, hourly bars, %d coins; held 24 hours, 3 ATR stop. Net bp per trade by fold %s"
          % (D.c.shape[1], " ".join(YEARS)), flush=True)
    for name, e in S.items():
        for side, label in ((1, "long"), (-1, "short")):
            s, ok = judge(D, e, stop, cost, side)
            print(line(name, label, s, ok), flush=True)

    print("\nCombinations: votes in the last 6 hours", flush=True)
    recent = {n: (e.rolling(6, min_periods=1).max(), e.rolling(6, min_periods=1).min()) for n, e in S.items()}
    ups = sum((mx > 0).astype(int) for mx, _ in recent.values())
    downs = sum((mn < 0).astype(int) for _, mn in recent.values())
    for k in (2, 3):
        e = ((ups >= k) & (downs == 0)).astype(int) - ((downs >= k) & (ups == 0)).astype(int)
        for side, label in ((1, "long"), (-1, "short")):
            s, ok = judge(D, e, stop, cost, side)
            print(line("vote: %d or more agree" % k, label, s, ok), flush=True)

    print("\nCombinations: every pair, both within 6 hours", flush=True)
    passed, tried = [], 0
    for (a, ea), (b, eb) in itertools.combinations(S.items(), 2):
        for side, label in ((1, "long"), (-1, "short")):
            fa = (ea == side).astype(int).rolling(6, min_periods=1).max() > 0
            fb = (eb == side).astype(int).rolling(6, min_periods=1).max() > 0
            now = (ea == side) | (eb == side)
            e = (fa & fb & now).astype(int) * side
            s, ok = judge(D, e, stop, cost, side)
            tried += 1
            if s["n"] >= 200 and (ok or s["good"] >= 5):
                print(line("%s + %s" % (a[:4], b[:4]), label, s, ok), flush=True)
            if ok:
                passed.append((a, b, label))
    print("\nPairs passing with all three holds: %d of %d: %s" % (len(passed), tried, passed), flush=True)


if __name__ == "__main__":
    main()
