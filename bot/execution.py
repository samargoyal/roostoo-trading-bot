"""Places planned trades on Roostoo.

Quantities are rounded down to each pair's AmountPrecision and limit prices to its
PricePrecision; orders worth less than the pair's minimum are skipped. Each trade
first rests as a limit order at the touch (the best bid for a buy, the best ask for
a sell) to pay the 0.05% maker fee. Whatever has not filled after limit_timeout_sec
is cancelled and sent as a market order (0.1% taker fee). Stop-loss exits go
straight to market.

Shorts use Roostoo's /v6 endpoints and always trade at market, since the 0.1% fee is the
same for limit orders: a new short locks its USD collateral (shared with buys out of free
cash), and a cover closes part or all of the position at the best ask.
"""
import logging
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from bot.config import ExecutionConfig
from bot.planner import BUY, COVER, SELL, SHORT, PlannedTrade
from bot.roostoo import RoostooClient, RoostooError
from bot.reasons import EXIT_STOP

log = logging.getLogger(__name__)

LIMIT = "LIMIT"
MARKET = "MARKET"
SHORT_OPEN = "SHORT_OPEN"
SHORT_CLOSE = "SHORT_CLOSE"
SHORT_FEE = 0.001
DRY_RUN = "DRY_RUN"
ERROR = "ERROR"
FINAL_STATUSES = {"FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED", DRY_RUN, ERROR}


@dataclass
class PairRules:
    price_decimals: int
    amount_decimals: int
    min_order_value: float  # MiniOrder: price x quantity must reach this many USD


def halted_pairs(exchange_info: Dict[str, Any], pairs: Iterable[str]) -> Set[str]:
    """The given pairs Roostoo will not trade now: listed with CanTrade false, or not listed."""
    listed = exchange_info.get("TradePairs", {})
    return {p for p in pairs if p not in listed or not listed[p].get("CanTrade", True)}


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
    collateral: Optional[Decimal] = None   # USD locked by a new short
    realized_pnl: float = 0.0               # profit or loss settled by a cover

    def update_short(self, detail: Dict[str, Any]) -> None:
        """Fill details from a /v6 short_open or short_close response."""
        self.order_id = detail.get("ID", self.order_id)
        self.role = "TAKER"
        self.fee_coin = "USD"
        if self.order_type == SHORT_CLOSE:
            self.status = "FILLED"
            self.filled = float(detail.get("ClosedQty") or 0.0)
            self.avg_price = float(detail.get("ClosePrice") or 0.0)
            self.fee = float(detail.get("CloseFee") or 0.0)
            self.realized_pnl = float(detail.get("RealizedPNL") or 0.0)
        else:
            # After adding to a short, ShortQty is the whole position, so record this order's size.
            self.status = str(detail.get("Status") or "OPEN")
            self.filled = float(self.quantity) if self.status == "OPEN" else 0.0
            self.avg_price = float(detail.get("EntryPrice") or 0.0)
            self.fee = float(detail.get("OpenFee") or 0.0)

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
            "collateral": to_str(self.collateral) if self.collateral is not None else "",
            "realized_pnl": self.realized_pnl if self.order_type == SHORT_CLOSE else "",
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
        """Sells and covers first, then buys and new shorts sized to the cash actually free."""
        results: List[OrderResult] = []
        sells = [t for t in trades if t.side == SELL]
        covers = [t for t in trades if t.side == COVER]
        buys = [t for t in trades if t.side == BUY]
        shorts = [t for t in trades if t.side == SHORT]
        if sells:
            results += self._run(self._size_sells(sells, quotes, self.client.balance()), quotes)
        if covers:
            results += self._cover(covers, quotes)
        if buys or shorts:
            scale = self._cash_scale(buys + shorts, quotes, self.client.balance())
            if buys:
                results += self._run(self._size_buys(buys, quotes, scale), quotes)
            if shorts:
                results += self._open_shorts(shorts, quotes, scale)
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

    def _cash_scale(self, trades: List[PlannedTrade], quotes: Dict[str, Dict[str, float]],
                    wallet: Dict[str, Dict[str, float]]) -> float:
        """Fraction of the planned buys and new shorts that free cash can pay for."""
        usable = [t for t in trades if t.pair in self.rules and t.pair in quotes]
        cash = float(wallet.get("USD", {}).get("Free") or 0.0) * (1.0 - self.cfg.cash_buffer)
        wanted = sum(t.usd * (1.0 + SHORT_FEE if t.side == SHORT else 1.0) for t in usable)
        scale = min(1.0, cash / wanted) if wanted > 0 else 0.0
        if scale < 1.0:
            log.warning("free cash %.2f covers %.0f%% of planned buys and shorts; scaling them down",
                        cash, scale * 100)
        return scale

    def _size_buys(self, trades: List[PlannedTrade], quotes: Dict[str, Dict[str, float]],
                   scale: float) -> List[Tuple[PlannedTrade, Decimal]]:
        usable = [t for t in trades if t.pair in self.rules and t.pair in quotes]
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

    def _open_shorts(self, trades: List[PlannedTrade], quotes: Dict[str, Dict[str, float]],
                     scale: float) -> List[OrderResult]:
        results = []
        for t in trades:
            rules, quote = self.rules.get(t.pair), quotes.get(t.pair)
            if rules is None or quote is None:
                log.warning("skipping SHORT %s: no trading rules or price", t.pair)
                continue
            bid = float(quote["MaxBid"])
            collateral = round_down(t.usd * scale, 2)
            quantity = round_down(float(collateral) / bid, rules.amount_decimals)
            if not self._large_enough(t, quantity, bid, rules):
                continue
            order = OrderResult(t, SHORT_OPEN, quantity, collateral=collateral)
            log.info("%s SHORT %s %s with %s USD collateral (%s)", "would open" if self.dry_run else "opening",
                     to_str(quantity), t.pair, to_str(collateral), t.reason)
            if self.dry_run:
                order.status = DRY_RUN
            else:
                try:
                    order.update_short(self.client.short_open(t.pair, to_str(collateral)))
                except RoostooError as exc:
                    order.status, order.error = ERROR, str(exc)
                    log.error("short failed: %s: %s", t.pair, exc)
            self.on_order(order)
            results.append(order)
        return results

    def _cover(self, trades: List[PlannedTrade], quotes: Dict[str, Dict[str, float]]) -> List[OrderResult]:
        try:
            positions = {p.get("Pair"): p for p in self.client.short_positions()}
        except RoostooError as exc:
            log.error("could not read short positions, skipping covers: %s", exc)
            return []
        results = []
        for t in trades:
            rules, quote, position = self.rules.get(t.pair), quotes.get(t.pair), positions.get(t.pair)
            if rules is None or quote is None or position is None:
                log.warning("skipping COVER %s: no trading rules, price or open short", t.pair)
                continue
            held = float(position.get("ShortQty") or 0.0)
            ask = float(quote["MinAsk"])
            wanted = held if t.close_position else min(held, t.usd / ask)
            quantity = round_down(wanted, rules.amount_decimals)
            if quantity <= 0 or float(quantity) * ask < rules.min_order_value:
                log.info("skipping COVER %s: %s units is below the exchange minimum", t.pair, to_str(quantity))
                continue
            order = OrderResult(t, SHORT_CLOSE, quantity)
            log.info("%s COVER %s %s (%s)", "would close" if self.dry_run else "closing",
                     "all" if t.close_position else to_str(quantity), t.pair, t.reason)
            if self.dry_run:
                order.status = DRY_RUN
            else:
                try:
                    order.update_short(self.client.short_close(
                        t.pair, None if t.close_position else to_str(quantity)))
                except RoostooError as exc:
                    order.status, order.error = ERROR, str(exc)
                    log.error("cover failed: %s: %s", t.pair, exc)
            self.on_order(order)
            results.append(order)
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
