"""H91: forecasting each coin's daily range, and trading it (the user: "try making forecasting
range").

The forecast: a HAR model (Corsi, 2009) of the log Parkinson variance, (ln high/low)^2 / 4 ln 2,
on yesterday's, last week's and last month's values, pooled over the bot's 49 coins and refitted
before each fold on the data up to it (walk-forward; nothing is fitted on the year it predicts).
It gives sigma, the day's expected volatility; the day's band is the open x exp(+- q sigma), q
set on the training data so that each side is touched on 20% of days. Checked first: how well it
predicts the next day's range against yesterday's range and the last 30 days' average, and how
often the band holds.

Then, written down before running, on hourly bars within each UTC day, long and short:

  RF1 fade the band: an hourly close beyond the band's top shorts (the move has used up the
      day's expected range), beyond its bottom buys; out at the day's open price, a stop half a
      sigma further, or the day's end
  RF2 break the band: the same closes followed instead (a range bigger than forecast is news);
      stop half a sigma back, out at the day's end

Neighbours: bands touched on 30% and 10% of days. Costs and the test as H86 (mean net per trade
positive in at least 5 of 6 folds and over all, neighbours too).

    python -m research.h91_range_forecast
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import costs, load_1h
from research.h86_scalping import STARTS, YEARS, engine
from research.h87_orb import score

LN2x4 = 4 * np.log(2)


def daily(D):
    day = D.c.index.normalize()
    o = D.o.groupby(day).first()
    h = D.h.groupby(day).max()
    l = D.l.groupby(day).min()
    c = D.c.groupby(day).last()
    full = D.c.notna().groupby(day).sum() >= 20
    return o.where(full), h.where(full), l.where(full), c.where(full)


def har_features(lv):
    """log Parkinson variance: yesterday's, the last 7 days' and last 30 days' means (known at
    the day's open), and the day's own value (the target)."""
    v = np.exp(lv)
    return dict(d=lv.shift(1), w=np.log(v.shift(1).rolling(7, min_periods=5).mean()),
                m=np.log(v.shift(1).rolling(30, min_periods=20).mean()), y=lv)


def stack(F, mask):
    cols = ["d", "w", "m", "y"]
    a = np.column_stack([F[k].where(mask).values.ravel() for k in cols])
    return a[np.isfinite(a).all(axis=1)]


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_1h()
    o, h, l, c = daily(D)
    park = (np.log(h / l) ** 2 / LN2x4).replace(0, np.nan)
    lv = np.log(park)
    F = har_features(lv)
    days = lv.index
    sigma = pd.DataFrame(np.nan, index=days, columns=lv.columns)
    qs = {}
    print("H91 range forecast: HAR on log Parkinson variance, walk-forward by fold, %d coins" % lv.shape[1])
    print("  fold   R^2 of log range: HAR  yesterday  30-day mean | band (20%% a side) touched: top  bottom  inside")
    for s, e in FOLDS:
        s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
        train = pd.DataFrame(np.repeat(np.asarray(days < s)[:, None], lv.shape[1], axis=1), index=days, columns=lv.columns)
        test = pd.DataFrame(np.repeat(np.asarray((days >= s) & (days < e))[:, None], lv.shape[1], axis=1),
                            index=days, columns=lv.columns)
        a = stack(F, train)
        X = np.column_stack([np.ones(len(a)), a[:, :3]])
        beta, *_ = np.linalg.lstsq(X, a[:, 3], rcond=None)
        bias = np.var(a[:, 3] - X @ beta) / 2
        pred = beta[0] + beta[1] * F["d"] + beta[2] * F["w"] + beta[3] * F["m"]
        sig = np.sqrt(np.exp(pred + bias))
        sigma = sigma.where(~test, sig)
        # band multipliers from the training days
        up = (np.log(h / o) / sig).where(train).values.ravel()
        dn = (np.log(o / l) / sig).where(train).values.ravel()
        quant = {}
        for touch in (0.3, 0.2, 0.1):
            quant[touch] = (np.nanquantile(up, 1 - touch), np.nanquantile(dn, 1 - touch))
        qs[s.year] = quant
        # forecast quality on the test year
        y = lv.where(test)
        def r2(f):
            m = np.isfinite(y.values) & np.isfinite(f.values)
            yy, ff = y.values[m], f.values[m]
            return 1 - np.mean((yy - ff - np.mean(yy - ff)) ** 2) / np.var(yy)
        naive_m = F["m"]
        qu, qd = quant[0.2]
        top = (np.log(h / o) > qu * sig).where(test)
        bot = (np.log(o / l) > qd * sig).where(test)
        n = test.values.sum()
        print("  %d   %26.2f  %9.2f  %11.2f | %33.0f%%  %6.0f%%  %6.0f%%" % (
            s.year, r2(pred), r2(F["d"]), r2(naive_m), 100 * np.nansum(top.values) / n,
            100 * np.nansum(bot.values) / n, 100 * (1 - np.nansum((top.fillna(0) + bot.fillna(0) > 0).values) / n)))

    # trading the band on hourly bars
    hour_day = D.c.index.normalize()
    sig_h = sigma.reindex(hour_day).set_axis(D.c.index)
    open_h = o.reindex(hour_day).set_axis(D.c.index)
    year_of = pd.Series([x.year if x.month >= 10 else x.year - 1 for x in hour_day], index=D.c.index)
    end = pd.DataFrame(np.repeat(np.asarray(D.c.index.hour == 23)[:, None], D.c.shape[1], axis=1),
                       index=D.c.index, columns=D.c.columns)
    cost = costs(list(D.c.columns))
    nan = np.full(len(D.idx), np.nan)
    stamps = D.idx.asi8
    print("\nTrading the band, hourly bars. Net bp per trade by fold %s" % " ".join(YEARS))
    for name in ("RF1 fade the band", "RF2 break the band"):
        for side in (1, -1):
            oks = []
            for touch in (0.2, 0.3, 0.1):
                qu = pd.Series({k: qs.get(k, {touch: (np.nan, np.nan)})[touch][0] for k in range(2020, 2026)})
                qd = pd.Series({k: qs.get(k, {touch: (np.nan, np.nan)})[touch][1] for k in range(2020, 2026)})
                mu = year_of.map(qu).values[:, None]
                md = year_of.map(qd).values[:, None]
                top = open_h * np.exp(mu * sig_h)
                bottom = open_h * np.exp(-md * sig_h)
                above, below = D.c > top, D.c < bottom
                first_above = above & (above.astype(int).groupby(hour_day).cumsum() == 1)
                first_below = below & (below.astype(int).groupby(hour_day).cumsum() == 1)
                if name.startswith("RF1"):
                    e = first_below.astype(int) - first_above.astype(int)
                    target = ((D.c - open_h).abs() / D.c)
                else:
                    e = first_above.astype(int) - first_below.astype(int)
                    target = None
                stop = 0.5 * sig_h
                out = []
                for pair in D.c.columns:
                    cc = D.c[pair].values
                    if np.isfinite(cc).sum() < 24 * 120:
                        continue
                    ee = np.where(e[pair].fillna(0).values == side, side, 0)
                    for i, sd, g in engine(cc, D.h[pair].values, D.l[pair].values, ee, stop[pair].values,
                                           nan if target is None else target[pair].values, 24,
                                           end[pair].values):
                        k = np.searchsorted(STARTS, stamps[i], side="right") - 1
                        if 0 <= k < len(YEARS):
                            m = g - 2 * cost[pair]
                            out.append((k, sd, g, m, m))
                sc = score(np.array(out).reshape(-1, 5), side)
                oks.append(sc["ok"])
                print("  %-20s %-5s band %2d%% %s | %6d trades win %3.0f%% | bp gross %+6.1f net %+6.1f | folds %d/%d" % (
                    name, "long" if side > 0 else "short", touch * 100,
                    " ".join("   --" if x != x else "%+5.0f" % x for x in sc["per"]), sc["n"], sc["win"],
                    sc["gross"], sc["net"], sc["good"], sc["have"]), flush=True)
            print("  -> %s %s: %s" % (name, "long" if side > 0 else "short", "PASS" if all(oks) else "fail"), flush=True)


if __name__ == "__main__":
    main()
