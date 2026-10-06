"""Operational circuit breakers: the bot stands aside from bad data, a broken exchange
connection or its own runaway orders (RESEARCH_QUEUE.md, part A).

Each check acts on the trades the planner proposes, before any order is sent, or on the
orders as they go out, and every trip is written to the journal (breakers.csv: name, pair,
value, threshold, action), so the trade log shows why the bot stood aside. Exits, stops and
the activity trade always stay allowed, except under the commit-controlled `trading_halt`, so a
tripped breaker can neither trap a position nor break the competition's trading-days rule.

  A1  price divergence   Roostoo's mid against Binance's last close; the pair gets no orders,
                         and if too many pairs diverge the cycle is skipped
  A2  stale quote        Roostoo's bid, ask and last unchanged over several samples while
                         Binance moved: the pair is treated as halted
  A3  missing bars       Binance's newest bar too old, or its fetch failed: no new entries in
                         the pair (exits still run)
  A4  wide spread        the pair trades by limit orders only
  A5  order failures     several rejected orders in a row, or too large a share of the cycle's:
                         the rest of the cycle is skipped (in bot/execution.py)
  A6  fill slippage      a market fill far from the quote is logged; repeated, the pair trades
                         by limit orders only for a day
  A7  order sanity       an entry larger than a share of equity, or too many orders in a cycle,
                         is refused
  A8  fee budget         fees over the last 24 hours above a share of equity: only exits
  A9  state mismatch     holdings that changed between cycles without the bot trading: no new
                         entries this cycle (the strategy state is rebuilt from the wallet anyway)
  A10 halt flags         `trading_halt` (no orders at all) and `reduce_only` (exits only), set
                         in config/<account>.json and so changed only by a commit
"""
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

from bot.config import BreakerConfig
from bot.market_data import HOUR_MS, Bar
from bot.planner import ACTIVITY, BUY, COVER, SELL, SHORT, PlannedTrade

log = logging.getLogger(__name__)

ENTRIES = {BUY, SHORT}
EXITS = {SELL, COVER}
DAY_MS = 24 * HOUR_MS


@dataclass
class Trip:
    name: str
    pair: str
    value: float
    threshold: float
    action: str

    def as_row(self) -> Dict[str, object]:
        return {"name": self.name, "pair": self.pair, "value": round(self.value, 6),
                "threshold": self.threshold, "action": self.action}


class Breakers:
    """The checks that need memory between cycles (quote samples, slippage strikes, fees paid,
    the holdings the last cycle ended with) and the screen applied to each cycle's trades."""

    def __init__(self, cfg: BreakerConfig):
        self.cfg = cfg
        self.samples: Dict[str, Deque[Tuple[float, float, float]]] = {}
        self.strikes: Dict[str, Deque[int]] = {}
        self.limit_until: Dict[str, int] = {}
        self.fees: Deque[Tuple[int, float]] = deque()
        self.held: Optional[Dict[str, float]] = None

    # ---- memory ---------------------------------------------------------------

    def sample(self, quotes: Dict[str, Dict[str, float]]) -> None:
        """A Roostoo ticker sample, for the stale-quote check (A2)."""
        for pair, q in quotes.items():
            row = (float(q.get("MaxBid") or 0), float(q.get("MinAsk") or 0), float(q.get("LastPrice") or 0))
            self.samples.setdefault(pair, deque(maxlen=self.cfg.stale_samples)).append(row)

    def remember_holdings(self, quantities: Dict[str, float]) -> None:
        """The wallet's coin quantities as the cycle ends, for the next cycle's A9 check."""
        self.held = dict(quantities)

    # ---- before trading ------------------------------------------------------

    def screen(self, trades: List[PlannedTrade], now: int, equity: float,
               quotes: Dict[str, Dict[str, float]], bars: Dict[str, Bar], failed: Set[str],
               quantities: Dict[str, float]) -> Tuple[List[PlannedTrade], Set[str], List[Trip]]:
        """The trades allowed this cycle, the pairs to trade by limit orders only, and the trips."""
        c = self.cfg
        trips: List[Trip] = []
        if c.trading_halt:
            return [], set(), [Trip("A10 trading halt", "", 1, 1, "no orders")] if trades else []

        no_entries = False                       # for every pair (the activity trade excepted)
        if c.reduce_only:
            no_entries = True
            trips.append(Trip("A10 reduce only", "", 1, 1, "exits only"))
        mismatch = self._mismatch(quantities, quotes, equity)
        if mismatch > c.max_state_mismatch:
            no_entries = True
            trips.append(Trip("A9 state mismatch", "", mismatch, c.max_state_mismatch, "no new entries"))
        fees = self._fees_24h(now) / equity if equity > 0 else 0.0
        if fees > c.fee_budget:
            no_entries = True
            trips.append(Trip("A8 fee budget", "", fees, c.fee_budget, "exits only"))

        blocked: Set[str] = set()                # no orders at all
        stale_data: Set[str] = set()             # no new entries
        limit_only = {p for p, until in self.limit_until.items() if until > now}
        for pair, q in quotes.items():
            bid, ask = float(q.get("MaxBid") or 0), float(q.get("MinAsk") or 0)
            bar = bars.get(pair)
            if bid > 0 and ask > 0 and bar is not None and bar.close > 0:
                divergence = abs((bid + ask) / 2 / bar.close - 1)
                if divergence > c.max_divergence:
                    blocked.add(pair)
                    trips.append(Trip("A1 price divergence", pair, divergence, c.max_divergence, "no orders"))
                if self._stale(pair, bar):
                    blocked.add(pair)
                    trips.append(Trip("A2 stale quote", pair, abs(bar.close / bar.open - 1), c.stale_move,
                                      "treated as halted"))
            if bid > 0 and ask > 0 and (ask - bid) / ((ask + bid) / 2) > c.max_spread:
                limit_only.add(pair)
                trips.append(Trip("A4 wide spread", pair, (ask - bid) / ((ask + bid) / 2), c.max_spread,
                                  "limit orders only"))
        divergent = sum(1 for t in trips if t.name.startswith("A1"))
        if divergent > c.max_divergent_pairs:
            trips.append(Trip("A1 price divergence", "", divergent, c.max_divergent_pairs, "cycle skipped"))
            return [], limit_only, trips
        for pair in set(bars) | failed:
            bar = bars.get(pair)
            age_min = (now - (bar.ts + HOUR_MS)) / 60000 if bar is not None else float("inf")
            if pair in failed or age_min > c.max_bar_age_min:
                stale_data.add(pair)
                trips.append(Trip("A3 missing bars", pair, age_min, c.max_bar_age_min, "no new entries"))

        allowed: List[PlannedTrade] = []
        for t in trades:
            if t.pair in blocked:
                continue
            entry = t.side in ENTRIES
            if entry and t.reason != ACTIVITY and (no_entries or t.pair in stale_data):
                continue
            if entry and t.side == SHORT and t.pair in limit_only:
                continue                         # shorts trade at market only
            if entry and t.reason != ACTIVITY and equity > 0 and t.usd > c.max_order_share * equity:
                trips.append(Trip("A7 order sanity", t.pair, t.usd / equity, c.max_order_share, "order refused"))
                continue
            allowed.append(t)
        if len(allowed) > c.max_orders:
            allowed.sort(key=lambda t: t.side not in EXITS)        # exits first
            trips.append(Trip("A7 order sanity", "", len(allowed), c.max_orders, "orders beyond the cap refused"))
            allowed = allowed[:c.max_orders]
        return allowed, limit_only, trips

    # ---- after trading -------------------------------------------------------

    def after(self, results, quotes: Dict[str, Dict[str, float]], now: int) -> List[Trip]:
        """Records fees (A8) and checks market fills against the quote they were sized on (A6)."""
        c = self.cfg
        trips: List[Trip] = []
        for r in results:
            if r.filled <= 0:
                continue
            fee = r.fee if r.fee_coin in ("USD", "") else r.fee * r.avg_price
            self.fees.append((now, fee))
            q = quotes.get(r.trade.pair)
            if r.order_type != "MARKET" or not q or r.avg_price <= 0:
                continue
            ref = float(q["MinAsk"] if r.trade.side in (BUY, COVER) else q["MaxBid"])
            slip = abs(r.avg_price / ref - 1) if ref > 0 else 0.0
            if slip > c.max_slippage:
                strikes = self.strikes.setdefault(r.trade.pair, deque())
                strikes.append(now)
                while strikes and strikes[0] <= now - DAY_MS:
                    strikes.popleft()
                action = "logged"
                if len(strikes) >= c.slippage_strikes:
                    self.limit_until[r.trade.pair] = now + DAY_MS
                    action = "limit orders only for 24 hours"
                trips.append(Trip("A6 fill slippage", r.trade.pair, slip, c.max_slippage, action))
        return trips

    # ---- helpers -------------------------------------------------------------

    def _stale(self, pair: str, bar: Bar) -> bool:
        rows = self.samples.get(pair)
        return (rows is not None and len(rows) >= self.cfg.stale_samples and len(set(rows)) == 1
                and bar.open > 0 and abs(bar.close / bar.open - 1) > self.cfg.stale_move)

    def _mismatch(self, quantities: Dict[str, float], quotes: Dict[str, Dict[str, float]], equity: float) -> float:
        if self.held is None or equity <= 0:
            return 0.0
        moved = 0.0
        for pair in set(quantities) | set(self.held):
            q = quotes.get(pair)
            if q:
                moved += abs(quantities.get(pair, 0.0) - self.held.get(pair, 0.0)) * float(q["LastPrice"])
        return moved / equity

    def _fees_24h(self, now: int) -> float:
        while self.fees and self.fees[0][0] <= now - DAY_MS:
            self.fees.popleft()
        return sum(f for _, f in self.fees)
