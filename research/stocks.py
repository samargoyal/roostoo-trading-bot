"""US share data for the tokenized-stock research (H26): point-in-time S&P 500 membership and
daily prices, so strategies are tested on the stocks that were in the index at each date
rather than on today's winners.

Membership: github.com/fja05680/sp500, "S&P 500 Historical Components & Changes (Updated).csv"
(one row per change date with the full list of members), saved to data/stocks/sp500_history.csv.
Prices: Yahoo Finance daily closes and volumes, adjusted for splits and dividends, cached per
ticker under data/stocks/prices/. Companies delisted long ago are often missing from Yahoo,
which leaves some survivorship bias; coverage() reports how much.

    python -m research.stocks        # download or top up every member since 2010
"""
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.join("data", "stocks")
PRICES = os.path.join(ROOT, "prices")
MEMBERSHIP = os.path.join(ROOT, "sp500_history.csv")
START = "2009-06-01"


def membership() -> pd.Series:
    """Date -> set of member tickers (Yahoo spelling), from the change-dated file."""
    df = pd.read_csv(MEMBERSHIP, parse_dates=["date"])
    return pd.Series([set(yahoo(t) for t in str(s).split(",")) for s in df["tickers"]], index=df["date"])


def yahoo(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def members_since(start: str = "2010-01-01") -> List[str]:
    m = membership()
    tickers = set()
    for date, s in m.items():
        if date >= pd.Timestamp(start) or date == m.index[m.index <= pd.Timestamp(start)].max():
            tickers |= s
    return sorted(tickers)


def download(tickers: List[str], chunk: int = 40) -> None:
    import yfinance as yf
    os.makedirs(PRICES, exist_ok=True)
    todo = [t for t in tickers if not os.path.exists(os.path.join(PRICES, t + ".csv"))]
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        df = yf.download(batch, start=START, progress=False, auto_adjust=True, group_by="ticker", threads=True)
        for t in batch:
            try:
                part = df[t] if len(batch) > 1 else df
                part = part[["Close", "Volume"]].dropna()
            except KeyError:
                part = pd.DataFrame()
            path = os.path.join(PRICES, t + ".csv")
            (part if not part.empty else pd.DataFrame(columns=["Close", "Volume"])).to_csv(path)
        print("downloaded %d/%d" % (min(i + chunk, len(todo)), len(todo)), flush=True)
        time.sleep(1.0)


def load(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """{"close": day x ticker, "volume": day x ticker} for tickers with data."""
    closes, volumes = {}, {}
    for t in tickers:
        path = os.path.join(PRICES, t + ".csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            continue
        closes[t], volumes[t] = df["Close"], df["Volume"]
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex(close.index)
    return {"close": close, "volume": volume}


def member_mask(index: pd.DatetimeIndex, columns: List[str]) -> pd.DataFrame:
    """True where the ticker was an S&P 500 member on that day."""
    m = membership()
    mask = pd.DataFrame(False, index=index, columns=columns)
    dates = list(m.index) + [pd.Timestamp("2100-01-01")]
    cols = set(columns)
    for d0, d1, members in zip(dates[:-1], dates[1:], m.values):
        rows = (index >= d0) & (index < d1)
        if rows.any():
            inside = [t for t in members if t in cols]
            mask.loc[rows, inside] = True
    return mask


def coverage(tickers: List[str]) -> float:
    have = sum(1 for t in tickers if os.path.exists(os.path.join(PRICES, t + ".csv"))
               and os.path.getsize(os.path.join(PRICES, t + ".csv")) > 40)
    return have / len(tickers) if tickers else 0.0


if __name__ == "__main__":
    names = members_since()
    print("%d tickers were S&P 500 members at some point since 2010" % len(names))
    download(names)
    print("price coverage: %.0f%% of them" % (coverage(names) * 100))
