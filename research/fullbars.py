"""Hourly Binance klines with every field: quote volume, number of trades and taker-buy volume.

bot/market_data.py keeps only OHLCV. Microstructure signals (research H20) also need who was
aggressive: Binance reports, per candle, the volume bought by takers. Cached under
data/binance_full/ as CSV.

    python -m research.fullbars          # fetch or top up every frozen candidate
    python -m research.fullbars --back   # extend them back to 2018 (the holdout)
"""
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pandas as pd
import requests

from bot.market_data import HOUR_MS, binance_symbol
from research.folds import candidate_table, ms

URL = "https://data-api.binance.vision/api/v3/klines"
FIELDS = ["ts", "open", "high", "low", "close", "volume", "quote_volume", "trades",
          "taker_buy_base", "taker_buy_quote"]
START, END = "2020-05-01", "2026-10-01"


def path(pair: str, data_dir: str = "data") -> str:
    return os.path.join(data_dir, "binance_full", binance_symbol(pair) + "_1h.csv")


def fetch(pair: str, start_ms: int, end_ms: int, session: requests.Session) -> List[list]:
    rows, cursor = [], start_ms
    while cursor < end_ms:
        for attempt in range(5):
            r = session.get(URL, params={"symbol": binance_symbol(pair), "interval": "1h",
                                         "startTime": cursor, "endTime": end_ms - 1, "limit": 1000},
                            timeout=30)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError("%s: HTTP %s" % (pair, r.status_code))
        page = r.json()
        if not page:
            break
        rows += [[int(k[0])] + [float(x) for x in k[1:6]] + [float(k[7]), int(k[8]), float(k[9]), float(k[10])]
                 for k in page]
        cursor = int(page[-1][0]) + HOUR_MS
    return rows


def update(pair: str, data_dir: str = "data") -> str:
    file = path(pair, data_dir)
    if os.path.exists(file):
        return "%s cached" % pair
    with requests.Session() as session:
        rows = fetch(pair, ms(START), ms(END), session)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file + ".tmp", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    os.replace(file + ".tmp", file)
    return "%s %d candles" % (pair, len(rows))


def extend_back(pair: str, start: str = "2018-01-01", data_dir: str = "data") -> str:
    """Prepend candles from `start` to a cached file (for the 2018-2020 holdout)."""
    file = path(pair, data_dir)
    if not os.path.exists(file):
        return "%s not cached" % pair
    df = pd.read_csv(file)
    first = int(df["ts"].iloc[0]) if len(df) else ms(END)
    if first <= ms(start):
        return "%s already from %s" % (pair, start)
    with requests.Session() as session:
        rows = fetch(pair, ms(start), first, session)
    if rows:
        old = df.values.tolist()
        merged = {int(r[0]): r for r in rows}
        merged.update({int(r[0]): r for r in old})
        with open(file + ".tmp", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)
            writer.writerows(merged[t] for t in sorted(merged))
        os.replace(file + ".tmp", file)
    return "%s +%d earlier candles" % (pair, len(rows))


def load(pairs: List[str], data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """{field: hour x pair frame} for every field, on a full hourly index."""
    frames = {}
    for pair in pairs:
        file = path(pair, data_dir)
        if os.path.exists(file):
            df = pd.read_csv(file)
            df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
            frames[pair] = df
    out = {}
    for field in FIELDS[1:]:
        out[field] = pd.DataFrame({p: f[field] for p, f in frames.items()}).sort_index()
    full = pd.date_range(out["close"].index[0], out["close"].index[-1], freq="h")
    return {k: v.reindex(full) for k, v in out.items()}


def main() -> None:
    import sys
    pairs = [r["pair"] for r in candidate_table() if r["asset_type"] == "crypto"]
    job = (lambda p: extend_back(p)) if "--back" in sys.argv else update
    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in pool.map(job, pairs):
            print(line)


if __name__ == "__main__":
    main()
