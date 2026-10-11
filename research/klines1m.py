"""One-minute Binance klines for the minute-scalping research (research/h126_minute_scalping.py).

Monthly files from Binance's public archive (data.binance.vision), September 2025 to September
2026, for every crypto pair in research/candidates.csv. Saved as one Parquet file per symbol under
data/binance_1m/ (git-ignored); times from 2025 on come in microseconds and are converted to ms.

    python -m research.klines1m
"""
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd

import requests

from research.klines5m import month

URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{y}-{m:02d}.zip"
OUT = os.path.join("data", "binance_1m")
MONTHS = [(2025, m) for m in range(9, 13)] + [(2026, m) for m in range(1, 10)]
SKIP = {"PAXG/USD", "MUB/USD"}                  # gold and a bond fund: nothing to scalp


def symbols():
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        return [r["pair"].replace("/USD", "USDT") for r in csv.DictReader(f)
                if r["asset_type"] == "crypto" and r["pair"] not in SKIP]


def fetch(symbol):
    path = os.path.join(OUT, symbol + ".parquet")
    if os.path.exists(path):
        return symbol, "cached"
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = [p for p in pool.map(partial(month, url=URL), [(symbol, ym) for ym in MONTHS]) if p is not None]
    if not parts:
        return symbol, "no data"
    df = pd.concat(parts).astype(float).astype({"ts": "int64"}).drop_duplicates("ts").sort_values("ts")
    df[["ts", "open", "high", "low", "close", "volume", "taker_buy_base"]].to_parquet(path, index=False)
    return symbol, "%d bars from %s" % (len(df), pd.to_datetime(df["ts"].iloc[0], unit="ms").date())


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for symbol in symbols():
        print(*fetch(symbol), flush=True)


if __name__ == "__main__":
    main()


def recent(start="2026-10-01"):
    """1-minute bars from `start` to now from Binance's live API (the monthly archive lags), saved
    to data/binance_1m_recent/<symbol>.parquet."""
    out = os.path.join("data", "binance_1m_recent")
    os.makedirs(out, exist_ok=True)
    begin = int(pd.Timestamp(start, tz="UTC").value // 10 ** 6)
    session = requests.Session()

    def one(symbol):
        rows, cursor = [], begin
        while True:
            r = session.get("https://data-api.binance.vision/api/v3/klines",
                            params={"symbol": symbol, "interval": "1m", "startTime": cursor, "limit": 1000}, timeout=20)
            if r.status_code != 200:
                break
            page = r.json()
            if not page:
                break
            rows += page
            if len(page) < 1000:
                break
            cursor = page[-1][0] + 60_000
        if not rows:
            return symbol, "no data"
        df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
        df.columns = ["ts", "open", "high", "low", "close", "volume", "taker_buy_base"]
        df = df.astype(float).astype({"ts": "int64"})
        df.to_parquet(os.path.join(out, symbol + ".parquet"), index=False)
        return symbol, "%d bars" % len(df)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for symbol, what in pool.map(one, symbols()):
            print(symbol, what, flush=True)
