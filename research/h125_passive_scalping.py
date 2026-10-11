"""H125: passive scalping on 5-minute bars, entries and targets by resting limit orders.

H86 entered every scalp at a bar's close, so it paid the taker fee and half the spread to get in,
and its "limit order" case only repriced that same entry. This tests the structure that suits a
0.05% maker fee: a buy limit resting below the price (a sell limit above it, for shorts), filled
only when the market trades down (up) through it, and a profit target that is itself a resting
limit. These are different trades from H86's, and they carry the adverse selection that passive
fills do: an order fills when the price is moving against it.

Fills. An order priced from bar i-1's close is live in bar i and fills at its price when bar i's
low (high, for shorts) reaches it, and in the stress case only when it trades 10 basis points
through it. Once filled, the stop is checked first, including in the fill bar; the target only
from the next bar, and in the same bar as a stop the stop wins. Stops and time exits are market
orders.

Costs. Longs: 0.05% in, 0.05% out at the target; 0.1% plus half the spread out at a stop or time
exit. Shorts (/v6 charges 0.1% to open and close and closes at market): 0.1% in, with no spread
because the entry rests; 0.1% plus half the spread out.

Written down before running, longs and shorts each:

  PS1 VWAP band: rest k sd (of 5-minute returns over a day) beyond the 1-hour VWAP, k = 2 or 3;
      target the VWAP at the fill, stop a further k sd, out after 2 hours
  PS2 trend dips: PS1 with k = 2, but only buying in a 4-hour-over-16-hour EMA uptrend and only
      shorting in a downtrend
  PS3 ATR ladder: rest 1.5 ATR (5-minute, 14) from the close; target 1 ATR from the fill, stop
      2 ATR, out after an hour
  PS4 the bot's hour: at each hour's close rest m hourly ATRs (14 hours) from it, m = 0.5 or 1,
      live for that hour only; target half an hourly ATR, stop 1.5, out at the next hour's close

Pass, as H86: the mean net return per trade positive in at least 5 of the 6 folds (folds with 20
trades or more) and over all of them, and again with the stress fills.

    python -m research.h125_passive_scalping
"""
import warnings

import numpy as np
import pandas as pd

from research.h60_swing_mft import costs, load_5m
from research.h86_scalping import STARTS, YEARS

MAKER, TAKER, SHORT_FEE = 0.0005, 0.001, 0.001
THROUGH = (0.0, 0.001)                                                  # base fills, stress fills


def engine(h, l, c, level, target, stop, max_bars, side, through, live_until=None):
    """One coin, one position at a time. level/target/stop are prices set at bar i-1's close for
    an order live in bar i (NaN: no order). live_until: the last bar a filled position may be
    held to (PS4), else fill + max_bars. Returns (fill bar, gross return, exit kind) per trade:
    kind 0 = target (resting limit), 1 = stop or time (market)."""
    n = len(c)
    lv = np.r_[np.nan, level[:-1]]
    tg = np.r_[np.nan, target[:-1]]
    st = np.r_[np.nan, stop[:-1]]
    if side > 0:
        fills = np.flatnonzero(l <= lv * (1 - through))
    else:
        fills = np.flatnonzero(h >= lv * (1 + through))
    trades, free = [], 0
    for i in fills:
        if i < free:
            continue
        p0, t, s = lv[i], tg[i], st[i]
        if not (t == t and s == s) or (t - p0) * side <= 0 or (p0 - s) * side <= 0:
            continue
        last = min(n - 1, i + max_bars if live_until is None else int(live_until[i]))
        out, kind, j = None, 1, i
        for j in range(i, last + 1):
            if c[j] != c[j]:
                continue
            if (side > 0 and l[j] <= s) or (side < 0 and h[j] >= s):
                out = s
                break
            if j > i and ((side > 0 and h[j] >= t * (1 + through)) or (side < 0 and l[j] <= t * (1 - through))):
                out, kind = t, 0
                break
        if out is None:
            k = j
            while k > i and c[k] != c[k]:
                k -= 1
            out = c[k]
        trades.append((i, side * (out / p0 - 1), kind))
        free = j + 1
    return trades


def designs(D):
    c, h, l = D.c, D.h, D.l
    t = c.index
    r = np.log(c).diff()
    sd = r.rolling(288, min_periods=144).std()
    vol = D.vol.replace(0, np.nan)
    vwap = ((h + l + c) / 3 * vol).rolling(12).sum() / vol.rolling(12).sum()
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr5 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up = c.ewm(span=48, adjust=False).mean() > c.ewm(span=192, adjust=False).mean()
    out = {}

    for k in (2, 3):
        for side in (1, -1):
            lv = vwap * np.exp(-side * k * sd)
            out[("PS1 VWAP band k=%d" % k, side)] = dict(level=lv, target=vwap, stop=lv * np.exp(-side * k * sd), bars=24)
    for side in (1, -1):
        lv = (vwap * np.exp(-side * 2 * sd)).where(up if side > 0 else ~up)
        out[("PS2 trend dips", side)] = dict(level=lv, target=vwap, stop=lv * np.exp(-side * 2 * sd), bars=24)
    for side in (1, -1):
        lv = c - side * 1.5 * atr5
        out[("PS3 ATR ladder", side)] = dict(level=lv, target=lv + side * atr5, stop=lv - side * 2 * atr5, bars=12)

    # PS4: hourly bars from the 5-minute ones; the order is set at an hour's last 5-minute close
    # and stays live for the 12 bars of the next hour; a fill is held to that hour's close.
    hc, hh, hl = c.resample("1h").last(), h.resample("1h").max(), l.resample("1h").min()
    htr = np.maximum(hh - hl, np.maximum((hh - hc.shift()).abs(), (hl - hc.shift()).abs()))
    hatr = htr.ewm(alpha=1 / 14, adjust=False).mean()
    hour = t.floor("1h")
    base = hc.shift(1).reindex(hour).set_axis(t)                        # the previous hour's close
    a = hatr.shift(1).reindex(hour).set_axis(t)
    pos = np.arange(len(t))
    end_of_hour = pd.Series(pos, index=t).groupby(hour).transform("max").values
    for m in (0.5, 1.0):
        for side in (1, -1):
            lv = base - side * m * a
            # level[i] is read for bar i+1, so shift back one bar to keep the hour's bars aligned
            out[("PS4 hourly m=%.1f" % m, side)] = dict(level=lv.shift(-1), target=(lv + side * 0.5 * a).shift(-1),
                                                       stop=(lv - side * 1.5 * a).shift(-1), bars=12,
                                                       live_until=end_of_hour)
    return out


def run(D, spec, side, cost, through):
    by_year = {y: [] for y in YEARS}
    stamps = D.idx.asi8
    for pair in D.c.columns:
        if D.c[pair].notna().sum() < 288 * 60:
            continue
        trades = engine(D.h[pair].values, D.l[pair].values, D.c[pair].values, spec["level"][pair].values,
                        spec["target"][pair].values, spec["stop"][pair].values, spec["bars"], side, through,
                        spec.get("live_until"))
        for i, g, kind in trades:
            k = np.searchsorted(STARTS, stamps[i], side="right") - 1
            if not 0 <= k < len(YEARS):
                continue
            if side > 0:
                fee = MAKER + (MAKER if kind == 0 else TAKER + cost[pair] - 0.001)
            else:
                fee = SHORT_FEE + SHORT_FEE + cost[pair] - 0.001
            by_year[YEARS[k]].append((g, g - fee, kind))
    return by_year


def report(name, side, by_year, label):
    cells, good, have, rows = [], 0, 0, []
    for y in YEARS:
        a = np.array(by_year[y]).reshape(-1, 3)
        if len(a) < 20:
            cells.append("     --")
            continue
        have += 1
        good += a[:, 1].mean() > 0
        cells.append("%+7.1f" % (a[:, 1].mean() * 1e4))
        rows.append(a)
    if not rows:
        print("  %-20s %-5s %-6s no trades" % (name, "long" if side > 0 else "short", label))
        return False
    a = np.vstack(rows)
    ok = good >= min(5, have) and a[:, 1].mean() > 0
    print("  %-20s %-5s %-6s %s | %7d trades, %3.0f%% at target | bp/trade gross %+6.1f net %+6.1f | folds %d/%d %s"
          % (name, "long" if side > 0 else "short", label, " ".join(cells), len(a), 100 * (a[:, 2] == 0).mean(),
             a[:, 0].mean() * 1e4, a[:, 1].mean() * 1e4, good, have, "PASS" if ok else "fail"), flush=True)
    return ok


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_5m()
    cost = costs(list(D.c.columns))                                     # taker fee plus half the spread
    print("H125 passive scalping, 5-minute bars, %d coins. Net bp per trade by fold %s"
          % (D.c.shape[1], " ".join(YEARS)), flush=True)
    passed = []
    for (name, side), spec in designs(D).items():
        base = report(name, side, run(D, spec, side, cost, THROUGH[0]), "touch")
        stress = report(name, side, run(D, spec, side, cost, THROUGH[1]), "10bp")
        if base and stress:
            passed.append((name, side))
    print("Passed both:", passed or "none")


if __name__ == "__main__":
    main()
