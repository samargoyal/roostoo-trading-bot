"""H115: support and resistance, breakout and reversal, long and short, one parameter: the lookback
(the user: "support/resistance breakout or reversal strategies, the only parameter the lookback").

Resistance is the highest high of the previous N hours, support the lowest low (the latest bar
left out). Written down before running, on hourly bars of the bot's coins (2018-26), each coin
equally weighted, the fee and half the spread paid on every change of position:

  SR-B  breakout: a close above resistance goes long, a close below support short, held until
        the opposite level breaks (stop and reverse)
  SR-R  reversal: a low below support with the close back above it (a failed breakdown) goes
        long, a high above resistance with the close back below it short, held until the close
        crosses the middle of the range

Lookbacks N: 24, 48, 72, 168, 336 and 720 hours. Reported: each year's return, the worst
drawdown, the six-year total, the mean median 14-day composite, every BTC crash since 2018, and
the long and short sides' own six-year totals.

    python -m research.h115_support_resistance
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import costs, curve_stats
from research.h61_swing import load
from research.h98_crash_validation import chained, crashes
from research.holdout2018 import HOLDOUT

LOOKBACKS = [24, 48, 72, 168, 336, 720]


def positions(D, n, kind):
    c, h, l = D.c, D.h, D.l
    res = h.shift(1).rolling(n, min_periods=n).max()
    sup = l.shift(1).rolling(n, min_periods=n).min()
    if kind == "breakout":
        sig = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        sig[c > res] = 1.0
        sig[c < sup] = -1.0
        return sig.ffill().fillna(0.0).where(D.ok, 0.0)
    mid = (res + sup) / 2
    up_entry = (l < sup) & (c > sup)
    dn_entry = (h > res) & (c < res)
    pos = np.zeros(c.shape)
    cv, mv, ue, de = c.values, mid.values, up_entry.values, dn_entry.values
    for j in range(c.shape[1]):
        p = 0.0
        for t in range(c.shape[0]):
            if p > 0 and cv[t, j] >= mv[t, j]:
                p = 0.0
            elif p < 0 and cv[t, j] <= mv[t, j]:
                p = 0.0
            if ue[t, j]:
                p = 1.0
            elif de[t, j]:
                p = -1.0
            pos[t, j] = p
    return pd.DataFrame(pos, index=c.index, columns=c.columns).where(D.ok, 0.0)


def curve(D, pos, cost, side=0):
    if side:
        pos = pos.where(np.sign(pos) == side, 0.0)
    n = D.ok.sum(axis=1).clip(lower=1)
    w = pos.div(n, axis=0)                                    # each coin 1 / (coins trading that hour)
    hold = w.shift(1).fillna(0.0)
    r = (D.c / D.c.shift(1) - 1).fillna(0.0)
    fees = (hold.diff().abs().fillna(0.0) * cost.reindex(w.columns).fillna(0.002)).sum(axis=1)
    net = (hold * r).sum(axis=1) - fees
    net.index = net.index + pd.Timedelta(hours=1)
    return net, hold.abs().sum(axis=1).mean()


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load()
    cost = costs(list(D.c.columns))
    folds = sorted(HOLDOUT + FOLDS)
    btc = D.c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    print("Crashes: %d | yearly (%s) | worst DD | six years | mean median 14d | crash returns, mean | six years long / short"
          % (len(events), " ".join(s[:4] for s, _ in folds)))
    for kind in ("breakout", "reversal"):
        for n in LOOKBACKS:
            pos = positions(D, n, kind)
            net, gross = curve(D, pos, cost)
            sides = [curve(D, pos, cost, s)[0] for s in (1, -1)]
            by, side_tot = [], []
            for s, e in folds:
                sl = (net.index > pd.Timestamp(s, tz="UTC")) & (net.index <= pd.Timestamp(e, tz="UTC"))
                by.append((1 + net[sl]).cumprod())
            stats = [curve_stats(x) for x in by]
            six = [x for (s, _), x in zip(folds, stats) if s >= FOLDS[0][0]]
            dev = (net.index > pd.Timestamp(FOLDS[0][0], tz="UTC")) & (net.index <= pd.Timestamp(FOLDS[-1][1], tz="UTC"))
            for sd in sides:
                side_tot.append(np.prod(1 + sd[dev]) - 1)
            eq = chained(by)
            crash = []
            for a, b, _ in events:
                seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
                crash.append(seg.iloc[-1] / seg.iloc[0] - 1 if len(seg) > 1 else np.nan)
            print("  SR-%s %4dh | %s | %3.0f%% | %+8.0f%% | %6.2f | %s, %+5.1f%% | %+7.0f%% / %+7.0f%% | gross %.2f" % (
                kind[0].upper(), n, " ".join("%+6.0f%%" % (x["ret"] * 100) for x in stats),
                max(x["mdd"] for x in six) * 100, (np.prod([1 + x["ret"] for x in six]) - 1) * 100,
                np.mean([x["w14"] for x in six]), " ".join("%+4.0f%%" % (x * 100) for x in crash),
                np.nanmean(crash) * 100, side_tot[0] * 100, side_tot[1] * 100, gross), flush=True)


if __name__ == "__main__":
    main()
