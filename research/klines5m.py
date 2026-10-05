"""Five-minute Binance klines for the medium-frequency research (research/h60_swing_mft.py).

Monthly files from Binance's public archive (data.binance.vision), October 2020 to September
2026, for the majors and the meme coins Roostoo lists. Saved as one Parquet file per symbol
under data/binance_5m/ (git-ignored). Files from 2025 on stamp times in microseconds; they are
converted to milliseconds.

    python -m research.klines5m
"""
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/5m/{s}-5m-{y}-{m:02d}.zip"
OUT = os.path.join("data", "binance_5m")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "SUIUSDT", "DOGEUSDT", "SHIBUSDT",
           "PEPEUSDT", "BONKUSDT", "FLOKIUSDT", "WIFUSDT", "PENGUUSDT", "TRUMPUSDT", "1000CHEEMSUSDT", "PUMPUSDT"]
MONTHS = [(y, m) for y in range(2020, 2027) for m in range(1, 13) if (2020, 10) <= (y, m) <= (2026, 9)]
COLUMNS = ["ts", "open", "high", "low", "close", "volume", "close_ts", "quote_volume", "trades",
           "taker_buy_base", "taker_buy_quote", "ignore"]


def month(args):
    symbol, (y, m) = args
    r = requests.get(URL.format(s=symbol, y=y, m=m), timeout=60)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLUMNS)
    if not str(df["ts"].iloc[0]).isdigit():          # a header row in some files
        df = df.iloc[1:].astype({"ts": "int64"})
    df["ts"] = df["ts"].astype("int64")
    df.loc[df["ts"] > 10 ** 14, "ts"] //= 1000        # microseconds from 2025 on
    return df[["ts", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]]


def fetch(symbol):
    path = os.path.join(OUT, symbol + ".parquet")
    if os.path.exists(path):
        return symbol, "cached"
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = [p for p in pool.map(month, [(symbol, ym) for ym in MONTHS]) if p is not None]
    if not parts:
        return symbol, "no data"
    df = pd.concat(parts).astype(float).astype({"ts": "int64"}).drop_duplicates("ts").sort_values("ts")
    df.to_parquet(path, index=False)
    return symbol, "%d bars from %s" % (len(df), pd.to_datetime(df["ts"].iloc[0], unit="ms").date())


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for symbol in SYMBOLS:
        print(*fetch(symbol), flush=True)


if __name__ == "__main__":
    main()
