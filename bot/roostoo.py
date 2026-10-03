"""Client for the Roostoo mock exchange REST API.

Handles request signing, the offset between our clock and the server's, a
client-side rate limit, retries with backoff, and a log of every request and
response (the "roostoo.api" logger). The API key and signature travel only in
headers, and headers are never logged.

API reference: https://github.com/roostoo/Roostoo-API-Documents
"""
import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

import requests

log = logging.getLogger(__name__)
api_log = logging.getLogger("roostoo.api")

DEFAULT_BASE_URL = "https://mock-api.roostoo.com"

# Success=false messages that describe a normal empty result, not a failure.
EMPTY_RESULT_MESSAGES = ("no pending order", "no order matched")


class RoostooError(Exception):
    """A request failed: network error, unexpected HTTP status, or Success=false."""

    def __init__(self, message: str, status: Optional[int] = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class RoostooAuthError(RoostooError):
    """HTTP 401: the key pair is wrong or the account is not active."""


class _Retryable(Exception):
    """A failure worth retrying: network error, HTTP 429 or 5xx."""


def build_query(params: Dict[str, Any]) -> str:
    """Parameters sorted by key and joined as k=v&k=v. This exact string is signed and sent."""
    return "&".join("{}={}".format(key, params[key]) for key in sorted(params))


def sign(query: str, secret: str) -> str:
    """HMAC-SHA256 hex digest of the query string, keyed with the secret."""
    return hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()


class RateLimiter:
    """Allows at most `max_calls` in any rolling window of `period` seconds."""

    def __init__(self, max_calls: int, period: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.max_calls = max_calls
        self.period = period
        self.clock = clock
        self.sleep = sleep
        self.calls: Deque[float] = deque()

    def acquire(self) -> None:
        while True:
            now = self.clock()
            while self.calls and now - self.calls[0] >= self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            wait = self.period - (now - self.calls[0])
            log.debug("rate limit reached, waiting %.1fs", wait)
            self.sleep(max(wait, 0.01))


class RoostooClient:
    def __init__(self, api_key: str, secret_key: str, base_url: str = DEFAULT_BASE_URL,
                 timeout_sec: float = 10.0, max_calls_per_minute: int = 20,
                 max_retries: int = 3, backoff_sec: float = 1.0,
                 session: Optional[requests.Session] = None):
        self.api_key = api_key
        self._secret = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_sec
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.session = session or requests.Session()
        self.limiter = RateLimiter(max_calls_per_minute)
        self.clock_offset_ms = 0

    @classmethod
    def from_env(cls, **kwargs: Any) -> "RoostooClient":
        """Credentials come only from ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY."""
        api_key = os.environ.get("ROOSTOO_API_KEY", "").strip()
        secret = os.environ.get("ROOSTOO_SECRET_KEY", "").strip()
        if not api_key or not secret:
            raise RoostooError("ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY must be set")
        return cls(api_key, secret, **kwargs)

    def __repr__(self) -> str:
        return "RoostooClient(%r)" % self.base_url

    # ---- clock ---------------------------------------------------------------

    def now_ms(self) -> int:
        """Current time in ms, corrected to the server's clock."""
        return int(time.time() * 1000) + self.clock_offset_ms

    def sync_clock(self) -> int:
        """Measure the server clock against ours. Signed requests are rejected beyond 60s of skew."""
        sent = time.time()
        server_ms = self.server_time()
        received = time.time()
        self.clock_offset_ms = server_ms - int((sent + received) / 2 * 1000)
        log.debug("clock offset vs server: %d ms", self.clock_offset_ms)
        return self.clock_offset_ms

    # ---- public endpoints ----------------------------------------------------

    def server_time(self) -> int:
        return int(self._request("GET", "/v3/serverTime")["ServerTime"])

    def exchange_info(self) -> Dict[str, Any]:
        return self._request("GET", "/v3/exchangeInfo")

    def ticker(self, pair: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """{"BTC/USD": {"MaxBid": .., "MinAsk": .., "LastPrice": .., ...}, ...} for one or all pairs."""
        params = {"pair": pair} if pair else {}
        return self._request("GET", "/v3/ticker", params, timestamp=True)["Data"]

    # ---- signed endpoints ----------------------------------------------------

    def balance(self) -> Dict[str, Dict[str, float]]:
        """Spot wallet as {"USD": {"Free": .., "Lock": ..}, "BTC": {...}, ...}."""
        body = self._request("GET", "/v3/balance", signed=True)
        # The live API answers with SpotWallet; the docs show Wallet.
        wallet = body.get("SpotWallet")
        if wallet is None:
            wallet = body.get("Wallet")
        if wallet is None:
            raise RoostooError("balance response has no wallet", payload=body)
        return wallet

    def pending_count(self) -> int:
        body = self._request("GET", "/v3/pending_count", signed=True)
        return int(body.get("TotalPending") or 0)

    def place_order(self, pair: str, side: str, quantity: str, order_type: str = "MARKET",
                    price: Optional[str] = None) -> Dict[str, Any]:
        """Returns the OrderDetail. Never retried: a request that timed out may still have
        placed the order, and the next cycle reconciles from the balance instead."""
        params = {"pair": pair, "side": side, "type": order_type, "quantity": quantity}
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("a LIMIT order needs a price")
            params["price"] = price
        return self._request("POST", "/v3/place_order", params, signed=True, retry=False)["OrderDetail"]

    def query_order(self, order_id: Optional[int] = None, pair: Optional[str] = None,
                    pending_only: Optional[bool] = None, offset: Optional[int] = None,
                    limit: Optional[int] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = order_id  # the API accepts no other filter alongside order_id
        else:
            if pair:
                params["pair"] = pair
            if pending_only is not None:
                params["pending_only"] = "TRUE" if pending_only else "FALSE"
            if offset is not None:
                params["offset"] = offset
            if limit is not None:
                params["limit"] = limit
        body = self._request("POST", "/v3/query_order", params, signed=True)
        return body.get("OrderMatched") or []

    def cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None) -> List[int]:
        """Cancel one order, every pending order on a pair, or (with neither) every pending order."""
        if order_id is not None and pair:
            raise ValueError("pass order_id or pair, not both")
        params: Dict[str, Any] = {}
        if order_id is not None:
            params["order_id"] = order_id
        elif pair:
            params["pair"] = pair
        body = self._request("POST", "/v3/cancel_order", params, signed=True)
        return body.get("CanceledList") or []

    # ---- shorts (/v6) ----------------------------------------------------------
    # 1x shorts sized by the USD collateral they lock. Opening and closing each cost 0.1%.
    # A field whose value is zero is left out of these responses: read it as 0.

    def short_open(self, pair: str, collateral: str, price: Optional[str] = None) -> Dict[str, Any]:
        """Open or add to a short; a market order fills at the best bid. Never retried."""
        params: Dict[str, Any] = {"pair": pair, "collateral": collateral}
        if price is not None:
            params["order_type"] = "LIMIT"
            params["price"] = price
        return self._request("POST", "/v6/short_open", params, signed=True, retry=False)

    def short_close(self, pair: str, close_qty: Optional[str] = None) -> Dict[str, Any]:
        """Close part (close_qty) or all of a short at the best ask. Never retried."""
        params: Dict[str, Any] = {"pair": pair}
        if close_qty is not None:
            params["close_qty"] = close_qty
        return self._request("POST", "/v6/short_close", params, signed=True, retry=False)

    def short_positions(self) -> List[Dict[str, Any]]:
        """Open shorts: Pair, EntryPrice, ShortQty, Collateral, CurrentPrice, UnrealizedPNL, PositionValue."""
        body = self._request("GET", "/v6/short_positions", signed=True)
        return body.get("Positions") or []

    # ---- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None,
                 signed: bool = False, timestamp: bool = False, retry: bool = True) -> Dict[str, Any]:
        attempts = 1 + (self.max_retries if retry else 0)
        for attempt in range(1, attempts + 1):
            try:
                # A fresh timestamp and signature for every attempt.
                return self._send(method, path, dict(params or {}), signed, timestamp or signed)
            except _Retryable as exc:
                if attempt == attempts:
                    raise RoostooError("%s %s failed after %d attempt(s): %s" % (method, path, attempts, exc))
                delay = self.backoff_sec * 2 ** (attempt - 1)
                log.warning("%s %s failed (%s), retry %d/%d in %.0fs",
                            method, path, exc, attempt, attempts - 1, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _send(self, method: str, path: str, params: Dict[str, Any], signed: bool,
              timestamp: bool) -> Dict[str, Any]:
        if timestamp:
            params["timestamp"] = str(self.now_ms())
        query = build_query(params)
        headers = {}
        if signed:
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = sign(query, self._secret)
        url = self.base_url + path

        self.limiter.acquire()
        started = time.monotonic()
        try:
            if method == "GET":
                # Send the signed string itself, so the server sees exactly what we signed.
                response = self.session.get(url + ("?" + query if query else ""),
                                            headers=headers, timeout=self.timeout)
            else:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                response = self.session.post(url, data=query, headers=headers, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            self._log(method, path, params, None, started, repr(exc))
            raise _Retryable(repr(exc))

        text = response.text
        self._log(method, path, params, response.status_code, started, text)
        if response.status_code == 401:
            raise RoostooAuthError("HTTP 401 from %s: %s" % (path, text[:200]), status=401)
        if response.status_code == 429 or response.status_code >= 500:
            raise _Retryable("HTTP %d" % response.status_code)
        if response.status_code != 200:
            raise RoostooError("HTTP %d from %s: %s" % (response.status_code, path, text[:200]),
                               status=response.status_code)
        try:
            body = response.json()
        except ValueError:
            raise RoostooError("non-JSON response from %s: %s" % (path, text[:200]), status=200)
        if not isinstance(body, dict):
            raise RoostooError("unexpected response from %s: %s" % (path, text[:200]), status=200)
        if body.get("Success") is False:
            message = str(body.get("ErrMsg") or "")
            if any(m in message.lower() for m in EMPTY_RESULT_MESSAGES):
                return body
            raise RoostooError("%s %s: %s" % (method, path, message), status=200, payload=body)
        return body

    @staticmethod
    def _log(method: str, path: str, params: Dict[str, Any], status: Optional[int],
             started: float, body: str) -> None:
        api_log.info(json.dumps({
            "method": method,
            "path": path,
            "params": params,
            "status": status,
            "ms": int((time.monotonic() - started) * 1000),
            "response": body,
        }, separators=(",", ":")))
