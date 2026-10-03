"""In-memory stand-ins for the Roostoo exchange and Binance, for tests."""
import copy
import math
from typing import Dict, List, Optional

from bot.market_data import HOUR_MS, Bar, BinanceError
from bot.roostoo import RoostooError

NOW = 1_790_000_000_000 // HOUR_MS * HOUR_MS + 60_000  # a minute past an hour


class FakeExchange:
    """Market orders fill at once; limit orders fill on the next query if fill_limits is set."""

    TAKER_FEE = 0.001
    MAKER_FEE = 0.0005

    def __init__(self, prices: Dict[str, float], cash: float = 100000.0, fill_limits: bool = True,
                 amount_decimals: int = 4):
        self.prices = dict(prices)
        self.wallet = {"USD": {"Free": cash, "Lock": 0.0}}
        self.orders: Dict[int, dict] = {}
        self.next_id = 1
        self.fill_limits = fill_limits
        self.amount_decimals = amount_decimals
        self.now = NOW
        self.placed: List[dict] = []
        self.shorts: Dict[str, dict] = {}
        self.short_calls: List[tuple] = []
        self.halted: set = set()      # pairs listed with CanTrade false; orders in them fail
        self.info_calls = 0

    # clock
    def sync_clock(self) -> int:
        return 0

    def now_ms(self) -> int:
        return self.now

    # public
    def exchange_info(self) -> dict:
        self.info_calls += 1
        return {"TradePairs": {p: {"PricePrecision": 2, "AmountPrecision": self.amount_decimals,
                                   "MiniOrder": 1, "CanTrade": p not in self.halted} for p in self.prices}}

    def ticker(self, pair: Optional[str] = None) -> dict:
        return {p: {"MaxBid": round(px * 0.9999, 2), "MinAsk": round(px * 1.0001, 2), "LastPrice": px}
                for p, px in self.prices.items()}

    # signed
    def balance(self) -> dict:
        return copy.deepcopy(self.wallet)

    def pending_count(self) -> int:
        return sum(1 for o in self.orders.values() if o["Status"] == "PENDING")

    def place_order(self, pair, side, quantity, order_type="MARKET", price=None) -> dict:
        if pair in self.halted:
            raise RoostooError("pair %s is not tradable" % pair)
        qty = float(quantity)
        order = {"OrderID": self.next_id, "Pair": pair, "Side": side, "Type": order_type,
                 "Quantity": qty, "Price": float(price) if price else 0.0, "Status": "PENDING",
                 "FilledQuantity": 0.0, "FilledAverPrice": 0.0, "CommissionChargeValue": 0.0,
                 "CommissionCoin": "USD", "Role": "MAKER"}
        self.next_id += 1
        self.orders[order["OrderID"]] = order
        self.placed.append(order)
        quote = self.ticker()[pair]
        if order_type == "MARKET":
            order["Role"] = "TAKER"
            self._fill(order, quote["MinAsk"] if side == "BUY" else quote["MaxBid"], self.TAKER_FEE)
        else:
            self._lock(order, +1)
        return copy.deepcopy(order)

    def query_order(self, order_id=None, pair=None, pending_only=None, offset=None, limit=None):
        order = self.orders.get(order_id)
        if order is None:
            return []
        if order["Status"] == "PENDING" and self.fill_limits:
            self._lock(order, -1)
            self._fill(order, order["Price"], self.MAKER_FEE)
        return [copy.deepcopy(order)]

    # shorts, following Roostoo's /v6 collateral rules
    SHORT_FEE = 0.001

    def short_open(self, pair, collateral, price=None) -> dict:
        self.short_calls.append(("open", pair, collateral))
        c = float(collateral)
        bid = self.ticker()[pair]["MaxBid"]
        qty = int(c / bid * 10 ** self.amount_decimals) / 10 ** self.amount_decimals
        fee = qty * bid * self.SHORT_FEE
        usd = self.wallet["USD"]
        if usd["Free"] < c + fee:
            raise RoostooError("POST /v6/short_open: insufficient balance")
        usd["Free"] -= c + fee
        usd["ShortCollateral"] = usd.get("ShortCollateral", 0.0) + c
        old = self.shorts.get(pair)
        if old:
            total = old["ShortQty"] + qty
            old["EntryPrice"] = (old["ShortQty"] * old["EntryPrice"] + qty * bid) / total
            old["ShortQty"], old["Collateral"] = total, old["Collateral"] + c
        else:
            self.shorts[pair] = {"ID": self.next_id, "Pair": pair, "EntryPrice": bid, "ShortQty": qty,
                                 "Collateral": c}
            self.next_id += 1
        pos = self.shorts[pair]
        return {"Success": True, "ID": pos["ID"], "Pair": pair, "OrderType": "MARKET",
                "EntryPrice": pos["EntryPrice"], "ShortQty": pos["ShortQty"],
                "Collateral": pos["Collateral"], "OpenFee": fee, "Status": "OPEN"}

    def short_close(self, pair, close_qty=None) -> dict:
        self.short_calls.append(("close", pair, close_qty))
        pos = self.shorts.get(pair)
        if pos is None:
            raise RoostooError("POST /v6/short_close: no open short position for this pair")
        ask = self.ticker()[pair]["MinAsk"]
        qty = pos["ShortQty"] if close_qty is None else min(float(close_qty), pos["ShortQty"])
        share = qty / pos["ShortQty"]
        pnl = qty * (pos["EntryPrice"] - ask)
        fee = qty * ask * self.SHORT_FEE
        returned = pos["Collateral"] * share + pnl - fee
        self.wallet["USD"]["Free"] += returned
        self.wallet["USD"]["ShortCollateral"] -= pos["Collateral"] * share
        full = share >= 1 - 1e-12
        if full:
            del self.shorts[pair]
        else:
            pos["ShortQty"] -= qty
            pos["Collateral"] *= 1 - share
        out = {"Success": True, "ClosePrice": ask, "RealizedPNL": pnl, "CloseFee": fee,
               "ReturnAmount": returned, "ClosedQty": qty, "FullyClosed": full}
        if not full:
            out.update(RemainingQty=pos["ShortQty"], RemainingCollateral=pos["Collateral"])
        return out

    def short_positions(self) -> List[dict]:
        out = []
        for pair, pos in self.shorts.items():
            ask = self.ticker()[pair]["MinAsk"]
            pnl = pos["ShortQty"] * (pos["EntryPrice"] - ask)
            out.append(dict(pos, CurrentPrice=ask, UnrealizedPNL=pnl, PositionValue=pos["Collateral"] + pnl,
                            PositionStatus="OPEN"))
        return out

    def cancel_order(self, order_id=None, pair=None) -> List[int]:
        cancelled = []
        for order in self.orders.values():
            if order["Status"] != "PENDING":
                continue
            if order_id is not None and order["OrderID"] != order_id:
                continue
            if pair is not None and order["Pair"] != pair:
                continue
            self._lock(order, -1)
            order["Status"] = "CANCELED"
            cancelled.append(order["OrderID"])
        return cancelled

    # internals
    def _coin(self, pair: str) -> dict:
        return self.wallet.setdefault(pair.split("/")[0], {"Free": 0.0, "Lock": 0.0})

    def _lock(self, order: dict, direction: int) -> None:
        if order["Side"] == "BUY":
            account, amount = self.wallet["USD"], order["Quantity"] * order["Price"]
        else:
            account, amount = self._coin(order["Pair"]), order["Quantity"]
        account["Free"] -= direction * amount
        account["Lock"] += direction * amount

    def _fill(self, order: dict, price: float, fee_rate: float) -> None:
        qty = order["Quantity"]
        fee = qty * price * fee_rate
        coin = self._coin(order["Pair"])
        if order["Side"] == "BUY":
            self.wallet["USD"]["Free"] -= qty * price + fee
            coin["Free"] += qty
        else:
            coin["Free"] -= qty
            self.wallet["USD"]["Free"] += qty * price - fee
        order.update(Status="FILLED", FilledQuantity=qty, FilledAverPrice=price,
                     CommissionChargeValue=fee)


class FakeBinance:
    def __init__(self, series: Dict[str, List[Bar]]):
        self.series = series
        self.fail = False

    def recent_closed(self, symbol: str, count: int, at_ms: Optional[int] = None) -> List[Bar]:
        if self.fail:
            raise BinanceError("simulated outage")
        bars = [b for b in self.series[symbol] if b.ts + HOUR_MS <= at_ms]
        return bars[-count:]


def zigzag(end_ms: int, start_price: float, drift: float, hours: int = 1200,
           swing: float = 1.0) -> List[Bar]:
    """Hourly bars alternating up and down around a drift, ending with the bar before end_ms.
    `swing` scales the up and down moves, so a larger value means a more volatile coin."""
    bars = []
    price = start_price
    first = end_ms // HOUR_MS * HOUR_MS - hours * HOUR_MS
    for i in range(hours):
        step = (0.006 if i % 2 == 0 else -0.004) * swing + drift
        new = price * math.exp(step)
        bars.append(Bar(first + i * HOUR_MS, price, max(price, new) * 1.001,
                        min(price, new) * 0.999, new, 1.0))
        price = new
    return bars
