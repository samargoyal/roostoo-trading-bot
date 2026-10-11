"""One-second Binance klines for the seconds scalping research (research/h127_second_scalping.py).

Monthly files from Binance's public archive (data.binance.vision), July to September 2026, for 20
pairs: the majors and the cheap, volatile coins Roostoo lists. About 20-70 MB a pair a month, so one
Parquet file per pair and month under data/binance_1s/ (git-ignored). Times come in microseconds.

    python -m research.klines1s
"""
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from research.klines5m import month

URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1s/{s}-1s-{y}-{m:02d}.zip"
OUT = os.path.join("data", "binance_1s")
MONTHS = [(2026, 7), (2026, 8), (2026, 9)]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT", "ENAUSDT", "NEARUSDT", "ARBUSDT",
           "ASTERUSDT", "TAOUSDT", "WLDUSDT", "PUMPUSDT", "CFXUSDT", "AVNTUSDT", "FETUSDT", "CRVUSDT", "SUSDT",
           "PENGUUSDT", "TRUMPUSDT"]


def fetch(job):
    symbol, (y, m) = job
    path = os.path.join(OUT, "%s-%d-%02d.parquet" % (symbol, y, m))
    if os.path.exists(path):
        return path, "cached"
    df = partial(month, url=URL)((symbol, (y, m)))
    if df is None:
        return path, "no data"
    df = df.astype(float).astype({"ts": "int64"}).drop_duplicates("ts").sort_values("ts")
    cols = ["open", "high", "low", "close", "volume", "taker_buy_base"]
    df[cols] = df[cols].astype("float32")
    df[["ts"] + cols].to_parquet(path, index=False)
    return path, "%d seconds" % len(df)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, ym) for s in SYMBOLS for ym in MONTHS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path, what in pool.map(fetch, jobs):
            print(path, what, flush=True)


if __name__ == "__main__":
    main()
