"""H29: perpetual-futures funding rates, a crowding signal the bot has not used.

Binance's perpetual futures charge longs and shorts a funding rate every 8 hours; when longs
pay a lot, the long side is crowded. Funding history is public (fapi/v1/fundingRate).
Written down before running:

  Cross-sectional  funding_7d (the mean of a coin's last 21 funding prints) against the next
                   24 and 168 hours' returns across the month's universe; the H20 rule (same
                   sign in at least 5 of 6 folds, mean |IC| at least 0.02, and with low
                   volatility partialled out the same sign in at least 4).
  Timing           BTC's funding_7d against BTC's next 7 days, and the universe's average
                   funding_7d against the equal-weight universe's next 7 days (Spearman per
                   fold); worth a strategy test with the same sign in at least 5 of 6 folds
                   and a mean |correlation| of at least 0.05.

    python -m research.h29_funding
"""
import os
import time

import numpy as np
import pandas as pd
import requests

from bot.config import UniverseConfig, load_config
from research.folds import FOLDS, candidates
from research.h20_screen import in_fold, rank_ic
from research.panel import DEFENSIVE, load_panel, monthly_universe

CACHE = os.path.join("data", "funding")
URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch(coin: str) -> pd.Series:
    """Funding prints for COINUSDT (or 1000COINUSDT), cached."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, coin + ".csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return pd.Series(df["rate"].values, index=pd.to_datetime(df["ts"], unit="ms", utc=True)) if len(df) else pd.Series(dtype=float)
    rows = []
    for symbol in (coin + "USDT", "1000" + coin + "USDT"):
        start = 1567296000000  # 2019-09-01
        while True:
            r = requests.get(URL, params={"symbol": symbol, "startTime": start, "limit": 1000}, timeout=30)
            if r.status_code != 200:
                break
            page = r.json()
            if not page:
                break
            rows += [(int(p["fundingTime"]), float(p["fundingRate"])) for p in page]
            start = int(page[-1]["fundingTime"]) + 1
            time.sleep(0.2)
            if len(page) < 1000:
                break
        if rows:
            break
    pd.DataFrame(rows, columns=["ts", "rate"]).to_csv(path, index=False)
    return pd.Series([r for _, r in rows], index=pd.to_datetime([t for t, _ in rows], unit="ms", utc=True))


def main() -> None:
    cfg = load_config()
    pairs = sorted(p for p in candidates(cfg))
    panel = load_panel(pairs)
    pairs = list(panel["close"].columns)
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    mask[DEFENSIVE] = False
    close = panel["close"]
    days = close.index[close.index.hour == 0]
    funding = {}
    for p in pairs:
        if p == DEFENSIVE:
            continue
        s = fetch(p.split("/")[0])
        if len(s):
            # mean of the last 21 prints known by each 00:00 bar's close (01:00)
            f7 = s.rolling(21, min_periods=15).mean()
            funding[p] = f7.reindex(days + pd.Timedelta(hours=1), method="ffill").values
    f7 = pd.DataFrame(funding, index=days).reindex(columns=pairs)
    print("funding history for %d of %d coins" % (f7.notna().any().sum(), len(pairs) - 1))
    logp = np.log(close)
    ret = logp.diff()
    low_vol = (-ret.rolling(168, min_periods=150).std()).loc[days].where(mask.loc[days])
    sig = f7.where(mask.loc[days])
    years = [s[:4] for s, _ in FOLDS]
    for h in (24, 168):
        fwd = (logp.shift(-h) - logp).loc[days].where(mask.loc[days])
        ic, partial = rank_ic(sig, fwd), rank_ic(sig, fwd, low_vol)
        means = [in_fold(ic, s, e).mean() for s, e in FOLDS]
        pmeans = [in_fold(partial, s, e).mean() for s, e in FOLDS]
        overall = float(np.nanmean(means))
        same = sum(np.sign(x) == np.sign(overall) for x in means)
        psame = sum(np.sign(x) == np.sign(overall) for x in pmeans)
        ok = same >= 5 and abs(overall) >= 0.02 and psame >= 4
        print("cross-sectional %3dh: IC by fold %s | mean %+.3f, same sign %d/6, partial %d/6 -> %s" % (
            h, " ".join("%+.3f" % x for x in means), overall, same, psame, "PASS" if ok else "fail"))

    daily = close.loc[days]
    fwd7 = np.log(daily).shift(-7) - np.log(daily)
    ew = np.log((daily.pct_change().where(mask.loc[days]).mean(axis=1).fillna(0) + 1).cumprod())
    ew_fwd7 = ew.shift(-7) - ew
    tests = {"BTC funding vs BTC next 7d": (f7["BTC/USD"], fwd7["BTC/USD"]),
             "average funding vs universe next 7d": (sig.mean(axis=1), ew_fwd7)}
    for name, (x, y) in tests.items():
        corrs = []
        for s, e in FOLDS:
            both = pd.concat([x, y], axis=1).dropna()
            both = both[(both.index >= pd.Timestamp(s, tz="UTC")) & (both.index < pd.Timestamp(e, tz="UTC"))]
            corrs.append(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
        mean = float(np.nanmean(corrs))
        same = sum(np.sign(c) == np.sign(mean) for c in corrs)
        print("timing %-36s %s | mean %+.3f, same sign %d/6 -> %s" % (
            name, " ".join("%s %+.3f" % (y_, c) for y_, c in zip(years, corrs)), mean, same,
            "PASS" if same >= 5 and abs(mean) >= 0.05 else "fail"))


if __name__ == "__main__":
    main()
