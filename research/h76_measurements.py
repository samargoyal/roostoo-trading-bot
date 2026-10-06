"""H76: the measurements the research queue asks for before its experiments (RESEARCH_QUEUE.md).

On hourly Binance bars, 2020-26 by fold:

  C3  weekends: is a weekend's return (Saturday 00:00 to Monday 00:00 UTC) followed by the
      opposite on Monday and Tuesday? Correlation per fold, for BTC and the equal-weighted market
  C5  hour-of-week relative volume, RVOL = this hour's quote volume over its median at the same
      hour of the week in the 4 weeks before; its distribution
  B4/E5  shock bars: a 1-hour return of -3 standard deviations (over 168 hours) on RVOL of 3 or
      more, and the capitulation subset that closes in the upper part of its range (wick 0.4
      or more). B4 sells them, E5 buys them; their next 4, 12 and 24 hours decide which, if
      either, by fold and by BTC's regime (the live filter, 168h EMA against 672h)
  B8  correlation: the mean pairwise correlation of hourly returns over 48 hours among the 10
      most traded coins; how often it is at 0.85 or more

    python -m research.h76_measurements
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.fullbars import load as load_full
from research.h60_swing_mft import ALL_1H, Data, ema

YEARS = [s[:4] for s, _ in FOLDS]


def fold_of(index):
    out = pd.Series(None, index=index, dtype=object)
    for s, e in FOLDS:
        out[(index >= pd.Timestamp(s, tz="UTC")) & (index < pd.Timestamp(e, tz="UTC"))] = s[:4]
    return out


def rvol(qv: pd.DataFrame) -> pd.DataFrame:
    lags = [qv.shift(168 * k) for k in (1, 2, 3, 4)]
    med = pd.concat([l.stack(dropna=False) for l in lags], axis=1).median(axis=1).unstack()
    return qv / med.reindex(index=qv.index, columns=qv.columns)


def weekends(D):
    print("C3: weekend return against the following Monday-Tuesday return (correlation per fold)")
    c = D.c
    days = c.resample("1D").last()
    for label, series in (("BTC", days["BTC/USD"]), ("market", None)):
        if series is None:
            r = np.log(days).diff()
            series = np.exp(r.where(D.ok.resample("1D").last().astype(bool)).mean(axis=1).cumsum())
        out = []
        for y, (s, e) in zip(YEARS, FOLDS):
            d = series.loc[s:e]
            sat = d.index[d.index.dayofweek == 5]
            pairs = []
            for t in sat:
                fri, mon, wed = t - pd.Timedelta(days=1), t + pd.Timedelta(days=2), t + pd.Timedelta(days=4)
                if fri in d.index and mon in d.index and wed in d.index:
                    pairs.append((d[mon] / d[fri] - 1, d[wed] / d[mon] - 1))
            a = np.array(pairs)
            out.append("%s %+.2f" % (y, np.corrcoef(a[:, 0], a[:, 1])[0, 1]))
        print("  %-7s %s" % (label, "  ".join(out)))


def shocks(D):
    c, h, l, qv = D.c, D.h, D.l, D.qv
    r = np.log(c / c.shift(1))
    z = r / r.rolling(168, min_periods=100).std().shift(1)
    rv = rvol(qv)
    wick = (c - l) / (h - l).replace(0, np.nan)
    btc = c["BTC/USD"]
    on = (ema(btc, 168) > ema(btc, 672))
    folds = fold_of(c.index)
    print("\nC5: RVOL quantiles (all coins, 2020-26): %s" % "  ".join(
        "p%d %.2f" % (q * 100, v) for q, v in zip((0.5, 0.9, 0.99), np.nanquantile(rv.values[rv.values > 0], [0.5, 0.9, 0.99]))))
    fwd = {k: c.shift(-k) / c.shift(-1) - 1 for k in (5, 13, 25)}   # bought at the next bar's open ~ this close
    shock = (z <= -3) & (rv >= 3) & D.ok
    capit = shock & (wick >= 0.4)
    for label, ev in (("B4 shock bars (z <= -3, RVOL >= 3)", shock), ("E5 capitulation (and wick >= 0.4)", capit)):
        print("\n%s: next 4, 12, 24 hours, mean and share positive" % label)
        for regime, mask in (("BTC filter on", on), ("BTC filter off", ~on)):
            row = []
            for y in YEARS:
                sel = ev & mask.values[:, None] & (folds == y).values[:, None]
                vals = [fwd[k].values[sel.values] for k in (5, 13, 25)]
                n = sel.values.sum()
                row.append("%s n=%d %s" % (y, n, "/".join("%+.2f%%" % (np.nanmean(v) * 100) if n else "--" for v in vals)))
            print("  %-15s %s" % (regime, " | ".join(row)))


def correlation(D):
    c, qv = D.c, D.qv
    r = np.log(c / c.shift(1))
    print("\nB8: mean pairwise 48-hour correlation among the 10 most traded coins (every 24 hours)")
    for y, (s, e) in zip(YEARS, FOLDS):
        idx = c.loc[s:e].index[::24]
        vals = []
        for t in idx:
            top = qv.loc[:t].tail(720).sum().nlargest(10).index
            w = r.loc[:t, top].tail(48).dropna(axis=1)
            if w.shape[1] < 5:
                continue
            m = w.corr().values
            vals.append(m[np.triu_indices_from(m, 1)].mean())
        v = np.array(vals)
        print("  %s median %.2f, p90 %.2f, at 0.85 or more %.0f%% of days" % (
            y, np.median(v), np.quantile(v, 0.9), (v >= 0.85).mean() * 100))


def main() -> None:
    warnings.filterwarnings("ignore")
    f = load_full(ALL_1H)
    D = Data({k: v.loc["2020-06-01":"2026-10-08"] for k, v in f.items()}, 1)
    weekends(D)
    shocks(D)
    correlation(D)


if __name__ == "__main__":
    main()
