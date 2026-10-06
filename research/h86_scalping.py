"""H86: scalping, long and short, on 5-minute bars (the user's request).

H60 tested 28 medium-frequency strategies on 5-minute bars and none survived costs. Scalps are
shorter still, so a trade must clear about 0.25% (0.1% to enter and to leave at market, plus
half the spread) before it earns anything. Written down before running, each long and short,
one position per coin at a time:

  SC1 VWAP snap-back: 2.5 standard deviations (of 5-minute returns over a day) below the
      rolling 1-hour VWAP, buy; above, short; out at the VWAP, a further 1.5 sd stop, or 2 hours
  SC2 US-open breakout: the 13:30-14:00 UTC range sets the US session's direction; a close
      above it buys, below it shorts; stop at the range's other side, target twice the range,
      out by 16:00 UTC
  SC3 momentum burst: a 5-minute return of 3 sd on 3 times the usual volume for that time of
      day keeps going; follow it for 30 minutes, stop 1.5 sd
  SC4 pullbacks in a trend: with the 4-hour EMA above the 16-hour, RSI(2) below 10 buys; with it
      below, RSI(2) above 90 shorts; out when RSI(2) crosses 50, a 1 ATR stop, or an hour
  SC5 BTC leads: BTC moves 2 sd in 5 minutes and an altcoin has moved less than half as much:
      trade the altcoin BTC's way for 15 minutes
  SC6 cascade fade: a -4 sd bar on 5 times the usual volume that closes in the top half of its
      range is forced selling that overshoots: buy for an hour; the mirror shorts a blow-off top
  SC7 squeeze breakout: Bollinger band width (20 bars) at its lowest of the day, then a close
      outside the bands: follow, stop at the middle band, out after 4 hours

Costs: market orders (0.1% a side and half the spread, as H60), and a best
case with limit orders at 0.05% a side and no spread for longs (shorts are market-only on
Roostoo). Longs and shorts are judged apart: the mean net return per trade positive in at least 5 of
the 6 folds (folds with 20 trades or more) and over all of them.

    python -m research.h86_scalping
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import costs, load_5m

FEE_LIMIT = 0.0005
STARTS = np.array([pd.Timestamp(s, tz="UTC").value for s, _ in FOLDS] + [pd.Timestamp(FOLDS[-1][1], tz="UTC").value])
YEARS = [s[:4] for s, _ in FOLDS]


def engine(c, h, l, entry, stop_dist, target_dist, max_bars, exit_signal=None, exit_price=None):
    """One coin, one position at a time. entry: +1/-1/0 per bar, filled at that bar's close.
    stop_dist, target_dist: distances (fractions of price) read at entry, NaN for none.
    exit_signal: leave at the close of a bar where it is true; exit_price: leave when a bar
    touches it (the VWAP). Returns (entry index, side, gross return) per trade."""
    n = len(c)
    trades = []
    free = 0
    for i in np.flatnonzero(entry):
        if i < free or i >= n - 1 or c[i] != c[i]:
            continue
        side = entry[i]
        p0, sd, td = c[i], stop_dist[i], target_dist[i]
        stop = p0 * (1 - side * sd) if sd == sd and sd > 0 else None
        target = p0 * (1 + side * td) if td == td and td > 0 else None
        out, j = None, i
        last = min(n - 1, i + max_bars)
        for j in range(i + 1, last + 1):
            if c[j] != c[j]:
                continue
            if stop is not None and ((side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop)):
                out = stop
                break
            if target is not None and ((side > 0 and h[j] >= target) or (side < 0 and l[j] <= target)):
                out = target
                break
            if exit_price is not None and exit_price[j] == exit_price[j] and (
                    (side > 0 and h[j] >= exit_price[j]) or (side < 0 and l[j] <= exit_price[j])):
                out = exit_price[j]
                break
            if exit_signal is not None and exit_signal[j]:
                out = c[j]
                break
        if out is None:
            k = j
            while k > i and c[k] != c[k]:
                k -= 1
            out = c[k]
        trades.append((i, side, side * (out / p0 - 1)))
        free = j + 1
    return trades


def rsi(x, n):
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def wide(mask, like):
    """A per-bar boolean or number repeated across the coins."""
    return pd.DataFrame(np.repeat(np.asarray(mask)[:, None], like.shape[1], axis=1), index=like.index,
                        columns=like.columns)


def strategies(D):
    c, h, l, qv = D.c, D.h, D.l, D.qv
    t = c.index
    r = np.log(c).diff()
    sd = r.rolling(288, min_periods=144).std().shift(1)                 # a day of 5-minute bars, known before the bar
    z = r / sd
    slot = t.hour * 12 + t.minute // 5
    med = qv.groupby(slot).transform(lambda s: s.shift(1).rolling(20, min_periods=5).median())
    rvol = qv / med                                                     # against the same time of day, last 20 days
    vol = D.vol.replace(0, np.nan)
    vwap = ((h + l + c) / 3 * vol).rolling(12).sum() / vol.rolling(12).sum()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    nan = pd.DataFrame(np.nan, index=t, columns=c.columns)
    out = {}

    dev = np.log(c / vwap) / sd
    out["SC1 VWAP snap-back"] = dict(entry=(dev <= -2.5).astype(int) - (dev >= 2.5).astype(int),
                                     stop=1.5 * sd, target=nan, bars=24, exit_price=vwap)

    day = t.normalize()
    in_range = wide((t.hour == 13) & (t.minute >= 30), c)
    hi = h.where(in_range).groupby(day).transform("max")
    lo = l.where(in_range).groupby(day).transform("min")
    window = wide((t.hour >= 14) & (t.hour < 16), c)
    above, below = (c > hi) & window, (c < lo) & window
    first_up = above & (above.astype(int).groupby(day).cumsum() == 1)
    first_down = below & (below.astype(int).groupby(day).cumsum() == 1)
    rng = (hi - lo) / c
    out["SC2 US-open breakout"] = dict(entry=first_up.astype(int) - first_down.astype(int), stop=rng,
                                       target=2 * rng, bars=24, exit_signal=wide(t.hour >= 16, c))

    burst = rvol >= 3
    out["SC3 momentum burst"] = dict(entry=((z >= 3) & burst).astype(int) - ((z <= -3) & burst).astype(int),
                                     stop=1.5 * sd, target=nan, bars=6)

    up = c.ewm(span=48, adjust=False).mean() > c.ewm(span=192, adjust=False).mean()
    r2 = rsi(c, 2)
    out["SC4 pullbacks in a trend"] = dict(entry=((r2 < 10) & up).astype(int) - ((r2 > 90) & ~up).astype(int),
                                           stop=atr / c, target=nan, bars=12,
                                           exit_signal=((r2 > 50) & up) | ((r2 < 50) & ~up))

    zb, rb = z["BTC/USD"].values[:, None], r["BTC/USD"].abs().values[:, None]
    lag = r.abs() < 0.5 * rb
    e = (lag & (zb >= 2)).astype(int) - (lag & (zb <= -2)).astype(int)
    e["BTC/USD"] = 0
    out["SC5 BTC leads"] = dict(entry=e, stop=1.5 * sd, target=nan, bars=3)

    where = (c - l) / (h - l).replace(0, np.nan)                       # 1 = closed at the high
    spike = rvol >= 5
    out["SC6 cascade fade"] = dict(entry=((z <= -4) & spike & (where >= 0.5)).astype(int)
                                   - ((z >= 4) & spike & (where <= 0.5)).astype(int),
                                   stop=2 * sd, target=nan, bars=12)

    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    width = std / mid
    tight = (width <= 1.05 * width.rolling(288, min_periods=144).min()).astype(float).rolling(12).max() > 0
    out["SC7 squeeze breakout"] = dict(entry=(tight & (c > mid + 2 * std)).astype(int)
                                       - (tight & (c < mid - 2 * std)).astype(int),
                                       stop=2 * std / c, target=nan, bars=48)
    return out


def run(D, spec, cost):
    """Trades by fold: (side, gross, net at market, net with limit orders for longs)."""
    by_year = {y: [] for y in YEARS}
    stamps = D.idx.asi8
    for pair in D.c.columns:
        if D.c[pair].notna().sum() < 288 * 60:
            continue
        ex, xp = spec.get("exit_signal"), spec.get("exit_price")
        trades = engine(D.c[pair].values, D.h[pair].values, D.l[pair].values,
                        spec["entry"][pair].fillna(0).values.astype(int), spec["stop"][pair].values,
                        spec["target"][pair].values, spec["bars"],
                        None if ex is None else ex[pair].fillna(False).values.astype(bool),
                        None if xp is None else xp[pair].values)
        for i, side, g in trades:
            k = np.searchsorted(STARTS, stamps[i], side="right") - 1
            if 0 <= k < len(YEARS):
                market = g - 2 * cost[pair]
                by_year[YEARS[k]].append((side, g, market, g - 2 * FEE_LIMIT if side > 0 else market))
    return by_year


def report(name, by_year):
    for label, keep in (("long", 1), ("short", -1)):
        cells, good_m, good_l, have, rows = [], 0, 0, 0, []
        for y in YEARS:
            a = np.array([x[1:] for x in by_year[y] if x[0] == keep]).reshape(-1, 3)
            if len(a) < 20:
                cells.append("     --")
                continue
            have += 1
            good_m += a[:, 1].mean() > 0
            good_l += a[:, 2].mean() > 0
            cells.append("%+7.1f" % (a[:, 1].mean() * 1e4))
            rows.append(a)
        if not rows:
            print("  %-26s %-5s no trades" % (name, label))
            continue
        a = np.vstack(rows)
        verdict = ("PASS" if good_m >= min(5, have) and a[:, 1].mean() > 0 else
                   "passes only with limit orders" if good_l >= min(5, have) and a[:, 2].mean() > 0 else "fail")
        print("  %-26s %-5s %s | %7d trades, win %3.0f%% | per trade bp: gross %+6.1f market %+6.1f limit %+6.1f "
              "| folds net>0 %d/%d -> %s" % (
                  name, label, " ".join(cells), len(a), 100 * (a[:, 1] > 0).mean(), a[:, 0].mean() * 1e4,
                  a[:, 1].mean() * 1e4, a[:, 2].mean() * 1e4, good_m, have, verdict), flush=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_5m()
    cost = costs(list(D.c.columns))                                     # fee and half the spread, a side
    print("H86 scalping, 5-minute bars, %d coins. Net basis points per trade at market, by fold %s"
          % (D.c.shape[1], " ".join(YEARS)), flush=True)
    for name, spec in strategies(D).items():
        report(name, run(D, spec, cost))


if __name__ == "__main__":
    main()
