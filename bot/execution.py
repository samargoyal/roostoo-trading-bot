"""Places planned trades on Roostoo.

Quantities are rounded down to each pair's AmountPrecision and limit prices to its
PricePrecision; orders worth less than the pair's minimum are skipped. Each trade
first rests as a limit order at the touch (the best bid for a buy, the best ask for
a sell) to pay the 0.05% maker fee. Whatever has not filled after limit_timeout_sec
is cancelled and sent as a market order (0.1% taker fee). Stop-loss exits go
straight to market.
"""
import logging
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from bot.config import ExecutionConfig
from bot.planner import BUY, SELL, PlannedTrade
from bot.roostoo import RoostooClient, RoostooError
from bot.strategy import EXIT_STOP

log = logging.getLogger(__name__)

LIMIT = "LIMIT"
MARKET = "MARKET"
DRY_RUN = "DRY_RUN"
ERROR = "ERROR"
FINAL_STATUSES = {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED", DRY_RUN, ERROR}


@dataclass
class PairRules:
    price_decimals: int
    amount_decimals: int
    min_order_value: float  # MiniOrder: price x quantity must reach this many USD


def parse_rules(exchange_info: Dict[str, Any]) -> Dict[str, PairRules]:
    rules = {}
    for pair, info in exchange_info.get("TradePairs", {}).items():
        if info.get("CanTrade", True):
            rules[pair] = PairRules(int(info["PricePrecision"]), int(info["AmountPrecision"]),
                                    float(info.get("MiniOrder", 1.0)))
    return rules


def round_down(value: float, decimals: int) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_DOWN)


def round_up(value: float, decimals: int) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_UP)


def to_str(value: Decimal) -> str:
    """Plain decimal notation; the API must never see something like 1E-5."""
    return format(value, "f")


@dataclass
class OrderResult:
    """One order sent to the exchange, or that would have been in dry-run mode."""
    trade: PlannedTrade
    order_type: str
    quantity: Decimal
    price: Optional[Decimal] = None
    order_id: Optional[int] = None
    status: str = ""
    role: str = ""
    filled: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    fee_coin: str = ""
    error: str = ""

    def update(self, detail: Dict[str, Any]) -> None:
        self.order_id = detail.get("OrderID", self.order_id)
        self.status = str(detail.get("Status") or self.status)
        self.role = str(detail.get("Role") or self.role)
        self.filled = float(detail.get("FilledQuantity") or 0.0)
        self.avg_price = float(detail.get("FilledAverPrice") or 0.0)
        self.fee = float(detail.get("CommissionChargeValue") or 0.0)
        self.fee_coin = str(detail.get("CommissionCoin") or self.fee_coin)

    @property
    def done(self) -> bool:
        return self.status in FINAL_STATUSES or self.filled >= float(self.quantity)

    def as_row(self) -> Dict[str, Any]:
        t = self.trade
        return {
            "pair": t.pair, "side": t.side, "type": self.order_type, "order_id": self.order_id,
            "status": self.status, "role": self.role, "quantity": to_str(self.quantity),
            "limit_price": to_str(self.price) if self.price is not None else "",
            "filled": self.filled, "avg_price": self.avg_price, "fee": self.fee,
            "fee_coin": self.fee_coin, "reason": t.reason,
            "current_weight": round(t.current_weight, 5), "target_weight": round(t.target_weight, 5),
            "error": self.error,
        }


class Executor:
    def __init__(self, client: RoostooClient, rules: Dict[str, PairRules], cfg: ExecutionConfig,
                 on_order: Callable[[OrderResult], None], dry_run: bool = False,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.client = client
        self.rules = rules
        self.cfg = cfg
        self.on_order = on_order
        self.dry_run = dry_run
        self.sleep = sleep
        self.clock = clock

    def execute(self, trades: List[PlannedTrade],
                quotes: Dict[str, Dict[str, float]]) -> List[OrderResult]:
        """Sells first, then buys sized to the cash actually available."""
        results: List[OrderResult] = []
        sells = [t for t in trades if t.side == SELL]
        buys = [t for t in trades if t.side == BUY]
        if sells:
            results += self._run(self._size_sells(sells, quotes, self.client.balance()), quotes)
        if buys:
            results += self._run(self._size_buys(buys, quotes, self.client.balance()), quotes)
        return results

    # ---- sizing --------------------------------------------------------------

    def _size_sells(self, trades: List[PlannedTrade], quotes: Dict[str, Dict[str, float]],
                    wallet: Dict[str, Dict[str, float]]) -> List[Tuple[PlannedTrade, Decimal]]:
        sized = []
        for t in trades:
            rules, quote = self.rules.get(t.pair), quotes.get(t.pair)
            if rules is None or quote is None:
                log.warning("skipping %s %s: no trading rules or price", t.side, t.pair)
                continue
            free = float(wallet.get(t.pair.split("/")[0], {}).get("Free") or 0.0)
            bid = float(quote["MaxBid"])
            wanted = free if t.close_position else min(free, t.usd / bid)
            quantity = round_down(wanted, rules.amount_decimals)
            if self._large_enough(t, quantity, bid, rules):
                sized.append((t, quantity))
        return sized

    def _size_buys(self, trades: List[PlannedTrade], quotes: Dict[str, Dict[str, float]],
                   wallet: Dict[str, Dict[str, float]]) -> List[Tuple[PlannedTrade, Decimal]]:
        usable = [t for t in trades if t.pair in self.rules and t.pair in quotes]
        cash = float(wallet.get("USD", {}).get("Free") or 0.0) * (1.0 - self.cfg.cash_buffer)
        wanted = sum(t.usd for t in usable)
        scale = min(1.0, cash / wanted) if wanted > 0 else 0.0
        if scale < 1.0:
            log.warning("free cash %.2f covers %.0f%% of planned buys; scaling them down",
                        cash, scale * 100)
        sized = []
        for t in usable:
            rules = self.rules[t.pair]
            ask = float(quotes[t.pair]["MinAsk"])
            quantity = round_down(t.usd * scale / ask, rules.amount_decimals)
            if self._large_enough(t, quantity, ask, rules):
                sized.append((t, quantity))
        return sized

    def _large_enough(self, t: PlannedTrade, quantity: Decimal, price: float,
                      rules: PairRules) -> bool:
        value = float(quantity) * price
        # Closing out a position only has to meet the exchange minimum.
        minimum = rules.min_order_value if t.close_position else max(rules.min_order_value,
                                                                      self.cfg.min_trade_usd)
        if quantity <= 0 or value < minimum:
            log.info("skipping %s %s: %s units worth %.2f USD is below the %.2f minimum",
                     t.side, t.pair, to_str(quantity), value, minimum)
            return False
        return True

    # ---- order handling ------------------------------------------------------

    def _run(self, sized: List[Tuple[PlannedTrade, Decimal]],
             quotes: Dict[str, Dict[str, float]]) -> List[OrderResult]:
        results = []
        resting = []
        for t, quantity in sized:
            if self.cfg.use_limit_orders and not (self.cfg.market_on_stop and t.reason == EXIT_STOP):
                order = self._place(t, LIMIT, quantity, self._touch_price(t, quotes[t.pair]))
                if order.done:
                    self.on_order(order)
                else:
                    resting.append(order)
            else:
                order = self._place(t, MARKET, quantity)
                self.on_order(order)
            results.append(order)

        if resting:
            self._wait(resting)
            for order in resting:
                if not order.done:
                    self._cancel(order)
                self.on_order(order)
                if not order.done:
                    # It may still fill, so a market order now could double the trade.
                    # The next cycle cancels leftovers and works from the real balance.
                    continue
                rules = self.rules[order.trade.pair]
                remainder = round_down(float(order.quantity) - order.filled, rules.amount_decimals)
                price = float(quotes[order.trade.pair]["MinAsk" if order.trade.side == BUY else "MaxBid"])
                if remainder > 0 and float(remainder) * price >= rules.min_order_value:
                    log.info("limit %s %s unfilled after %ds, sending %s at market", order.trade.side,
                             order.trade.pair, self.cfg.limit_timeout_sec, to_str(remainder))
                    market = self._place(order.trade, MARKET, remainder)
                    self.on_order(market)
                    results.append(market)
        return results

    @staticmethod
    def _touch_price(t: PlannedTrade, quote: Dict[str, float]) -> float:
        return float(quote["MaxBid"]) if t.side == BUY else float(quote["MinAsk"])

    def _place(self, t: PlannedTrade, order_type: str, quantity: Decimal,
               price: Optional[float] = None) -> OrderResult:
        rules = self.rules[t.pair]
        limit_price = None
        if price is not None:
            # Round towards our side of the book so the order cannot cross the spread.
            limit_price = (round_down(price, rules.price_decimals) if t.side == BUY
                           else round_up(price, rules.price_decimals))
        order = OrderResult(t, order_type, quantity, limit_price)
        log.info("%s %s %s %s%s (%s)", "would place" if self.dry_run else "placing", order_type,
                 t.side, to_str(quantity) + " " + t.pair,
                 " @ " + to_str(limit_price) if limit_price is not None else "", t.reason)
        if self.dry_run:
            order.status = DRY_RUN
            return order
        try:
            detail = self.client.place_order(t.pair, t.side, to_str(quantity), order_type,
                                             to_str(limit_price) if limit_price is not None else None)
            order.update(detail)
        except RoostooError as exc:
            order.status = ERROR
            order.error = str(exc)
            log.error("order failed: %s %s %s: %s", t.side, t.pair, to_str(quantity), exc)
        return order

    def _wait(self, orders: List[OrderResult]) -> None:
        deadline = self.clock() + self.cfg.limit_timeout_sec
        pending = list(orders)
        while pending and self.clock() < deadline:
            self.sleep(self.cfg.poll_interval_sec)
            for order in list(pending):
                self._refresh(order)
                if order.done:
                    pending.remove(order)

    def _refresh(self, order: OrderResult) -> None:
        try:
            matched = self.client.query_order(order_id=order.order_id)
        except RoostooError as exc:
            log.warning("could not query order %s: %s", order.order_id, exc)
            return
        if matched:
            order.update(matched[0])

    def _cancel(self, order: OrderResult) -> None:
        try:
            self.client.cancel_order(order_id=order.order_id)
        except RoostooError as exc:
            log.warning("could not cancel order %s: %s", order.order_id, exc)
        self._refresh(order)
        if not order.done:
            log.warning("order %s still shows %s after cancelling", order.order_id, order.status)
