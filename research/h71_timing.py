"""Is there a right time to be short? Evidence first (the user: "make long short at the right time").

Every UTC day at 00:00, 2020-26: fast "market turning down" signals on BTC and on all coins,
against the next 24 and 72 hours of BTC, of the equal-weighted market and of the rotation's top 2
(by 14-day return). Shorting at those times pays only if what is shorted then falls, in most
years, by more than the round trip (about 0.25%). A hedge pays if the rotation's picks beat BTC
while BTC is shorted against them.

    python -m research.h71_timing
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.fullbars import load as load_full
from research.h60_swing_mft import ALL_1H, Data, ema



def main() -> None:
    warnings.filterwarnings("ignore")
    f = load_full(ALL_1H)
    D = Data({k: v.loc["2020-06-01":"2026-10-08"] for k, v in f.items()}, 1)
    c = D.c
    ok = D.ok.astype(bool)
    b = c["BTC/USD"]
    mid = D.idx[(D.idx + pd.Timedelta(hours=1)).hour == 0]
    fwd = {h: (c.shift(-h) / c - 1) for h in (24, 72)}
    r336 = (c / c.shift(336) - 1).where(ok)
    n = ok.sum(axis=1).astype(float).replace(0.0, np.nan)
    signals = {
        "BTC 3-day return below 0": b / b.shift(72) - 1 < 0,
        "BTC fell over 3% in a day": b / b.shift(24) - 1 < -0.03,
        "BTC 24h EMA below its 72h EMA": ema(b, 24) < ema(b, 72),
        "BTC below its 72h EMA": b < ema(b, 72),
        "BTC 7-day EMA below 28-day (live filter off)": ema(b, 168) < ema(b, 672),
        "under half the coins above their 240h EMA": ((c > ema(c, 240)) & ok).sum(axis=1).astype(float) / n < 0.5,
    }
    rows = []
    for t in mid:
        if t not in r336.index or t > D.idx[-73]:
            continue
        fy = next((s[:4] for s, e in FOLDS if pd.Timestamp(s, tz="UTC") <= t < pd.Timestamp(e, tz="UTC")), None)
        if fy is None:
            continue
        r = r336.loc[t].dropna()
        top = r[r > 0].sort_values(ascending=False).index[:2]
        row = {"t": t, "fold": fy}
        for h in (24, 72):
            f_ = fwd[h].loc[t].where(ok.loc[t])
            row["btc%d" % h] = f_["BTC/USD"]
            row["mkt%d" % h] = f_.mean()
            row["rot%d" % h] = f_[top].mean() if len(top) else np.nan
        for k, s in signals.items():
            row[k] = bool(s.loc[t])
        rows.append(row)
    df = pd.DataFrame(rows)
    years = [s[:4] for s, _ in FOLDS]
    print("%d days, 2020-26; next 24h: BTC %+.2f%%, market %+.2f%%, rotation %+.2f%%" % (
        len(df), df.btc24.mean() * 100, df.mkt24.mean() * 100, df.rot24.mean() * 100))
    for k in signals:
        on = df[df[k]]
        btc_neg = sum(on[on.fold == y].btc24.mean() < 0 for y in years)
        mkt_neg = sum(on[on.fold == y].mkt24.mean() < 0 for y in years)
        spread = on.rot24 - on.btc24
        hedge = sum(spread[on.fold == y].mean() > 0 for y in years)
        print("\n%s (on %.0f%% of days)" % (k, df[k].mean() * 100))
        print("  next 24h when on: BTC %+.2f%% (falls in %d/6 years), market %+.2f%% (%d/6), rotation %+.2f%%" % (
            on.btc24.mean() * 100, btc_neg, on.mkt24.mean() * 100, mkt_neg, on.rot24.mean() * 100))
        print("  next 72h when on: BTC %+.2f%%, market %+.2f%%, rotation %+.2f%%" % (
            on.btc72.mean() * 100, on.mkt72.mean() * 100, on.rot72.mean() * 100))
        print("  BTC next 24h by year: %s" % " ".join("%s %+.2f%%" % (y, on[on.fold == y].btc24.mean() * 100) for y in years))
        print("  hedge (picks minus BTC) next 24h %+.2f%%, positive in %d/6 years" % (spread.mean() * 100, hedge))


if __name__ == "__main__":
    main()
