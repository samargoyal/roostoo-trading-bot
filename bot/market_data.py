"""Hourly candles from Binance's public market-data API (no key needed).

Roostoo streams its prices from Binance, so Binance's USDT candles stand in for
Roostoo history: they warm up the indicators, drive the live signals and feed the
backtest. Backtest history is cached as CSV under data/.
"""
import csv
import logging
import os
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import requests

log = logging.getLogger(__name__)

DEFAULT_BINANCE_URL = "https://data-api.binance.vision"
HOUR_MS = 3600 * 1000
MAX_KLINES = 1000  # Binance's per-request limit


class Bar(NamedTuple):
    ts: int        # open time, ms since epoch (UTC); the bar closes at ts + HOUR_MS
    open: float
    high: float
    low: float
    close: float
    volume: float  # base-asset volume (0 when built from Roostoo prices)


class BinanceError(Exception):
    pass


def binance_symbol(pair: str) -> str:
    """Roostoo "BTC/USD" -> Binance "BTCUSDT"."""
    coin, quote = pair.split("/")
    return coin + ("USDT" if quote == "USD" else quote)


def now_ms() -> int:
    return int(time.time() * 1000)


class BinanceClient:
    def __init__(self, base_url: str = DEFAULT_BINANCE_URL, timeout_sec: float = 10.0,
                 max_retries: int = 3, session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_sec
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def klines(self, symbol: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None,
               limit: int = MAX_KLINES) -> List[Bar]:
        params: Dict[str, Any] = {"symbol": symbol, "interval": "1h", "limit": limit}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        rows = self._get("/api/v3/klines", params)
        return [Bar(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                for r in rows]

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int) -> List[Tuple[int, float]]:
        """Perpetual funding prints (time, rate) in [start_ms, end_ms], from Binance USD-M futures
        (construct the client with the futures URL)."""
        rows = self._get("/fapi/v1/fundingRate",
                         {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000})
        return [(int(r["fundingTime"]), float(r["fundingRate"])) for r in rows]

    def recent_closed(self, symbol: str, count: int, at_ms: Optional[int] = None) -> List[Bar]:
        """The last `count` hourly candles that had closed by `at_ms` (default: now)."""
        at_ms = now_ms() if at_ms is None else at_ms
        # The newest candle is usually still open, so one page of MAX_KLINES may hold one
        # closed candle too few; fetch earlier pages until there are enough.
        bars = [b for b in self.klines(symbol, limit=MAX_KLINES) if b.ts + HOUR_MS <= at_ms]
        while bars and len(bars) < count:
            earlier = self.klines(symbol, end_ms=bars[0].ts - 1,
                                  limit=min(MAX_KLINES, count - len(bars)))
            if not earlier:
                break  # no older history: the pair was listed recently
            bars = earlier + bars
        return bars[-count:]

    def history(self, symbol: str, start_ms: int, end_ms: int) -> List[Bar]:
        """Closed candles opening in [start_ms, end_ms), fetched page by page."""
        out: List[Bar] = []
        cursor = start_ms
        while cursor < end_ms:
            page = self.klines(symbol, start_ms=cursor, end_ms=end_ms - 1)
            if not page:
                break
            out.extend(b for b in page if b.ts < end_ms)
            cursor = page[-1].ts + HOUR_MS
        cutoff = now_ms()
        return [b for b in out if b.ts + HOUR_MS <= cutoff]

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        problem = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(2 ** (attempt - 1))
            try:
                response = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                problem = repr(exc)
                continue
            if response.status_code == 200:
                return response.json()
            problem = "HTTP %d: %s" % (response.status_code, response.text[:200])
            if response.status_code not in (418, 429) and response.status_code < 500:
                break  # a bad request will not improve on retry
        raise BinanceError("%s %s failed: %s" % (path, params.get("symbol"), problem))


def load_history(client: BinanceClient, pair: str, start_ms: int, end_ms: int,
                 data_dir: str) -> List[Bar]:
    """Candles for [start_ms, end_ms), served from a CSV cache and topped up from Binance."""
    symbol = binance_symbol(pair)
    path = os.path.join(data_dir, "binance", symbol + "_1h.csv")
    cached = _read_bars(path)
    fetched: List[Bar] = []
    if not cached:
        fetched = client.history(symbol, start_ms, end_ms)
    else:
        if start_ms < cached[0].ts:
            fetched += client.history(symbol, start_ms, cached[0].ts)
        if end_ms > cached[-1].ts + HOUR_MS:
            fetched += client.history(symbol, cached[-1].ts + HOUR_MS, end_ms)
    if fetched:
        merged = {b.ts: b for b in cached}
        merged.update((b.ts, b) for b in fetched)
        cached = [merged[ts] for ts in sorted(merged)]
        _write_bars(path, cached)
        log.info("cached %d candles for %s", len(cached), symbol)
    return [b for b in cached if start_ms <= b.ts < end_ms]


def _read_bars(path: str) -> List[Bar]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [Bar(int(r["ts"]), float(r["open"]), float(r["high"]), float(r["low"]),
                    float(r["close"]), float(r["volume"])) for r in csv.DictReader(f)]


def _write_bars(path: str, bars: List[Bar]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(Bar._fields)
        writer.writerows(bars)
    os.replace(tmp, path)
