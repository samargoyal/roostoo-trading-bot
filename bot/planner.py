"""Turns target weights into trades. Shared by the backtest and the live bot.

Targets are signed: positive for a long holding, negative for a short. Two rules keep
trading costs and compliance in balance:

  Threshold  a held position is resized only when it is off target by at least
             rebalance_threshold of equity; entries and full exits always go through.
  Activity   the competition requires trades on at least 8 days. If nothing has
             filled in the current UTC-aligned block of activity_block_hours and the
             block is nearly over, the long position furthest from its target is
             rebalanced, even below the threshold. The permanent PAXG core means
             there is always a position to adjust.
"""
from dataclasses import dataclass
from typing import AbstractSet, Dict, List, Optional

from bot.config import ExecutionConfig
from bot.market_data import HOUR_MS
from bot.strategy import (EXIT_REGIME, EXIT_SHORT, EXIT_SHORT_STOP, EXIT_STOP, EXIT_TREND,
                          HOLD_HALTED, SHORT_ENTRY, Decision)

BUY = "BUY"        # open or add to a long
SELL = "SELL"      # reduce or close a long
SHORT = "SHORT"    # open or add to a short
COVER = "COVER"    # reduce or close a short
REBALANCE = "rebalance"
ACTIVITY = "activity"
EXIT_UNIVERSE = "exit_universe"  # a holding in a pair the strategy no longer trades
EXITS = (EXIT_TREND, EXIT_STOP, EXIT_REGIME, EXIT_SHORT_STOP, EXIT_SHORT)


@dataclass
class PlannedTrade:
    pair: str
    side: str              # BUY, SELL, SHORT or COVER
    usd: float             # notional at the current price
    close_position: bool   # close the entire long (SELL) or short (COVER)
    current_weight: float
    target_weight: float
    reason: str


def plan_trades(decision: Decision, weights: Dict[str, float], equity: float, ts: int,
                last_fill_ts: int, cfg: ExecutionConfig, min_position_weight: float,
                frozen: AbstractSet[str] = frozenset()) -> List[PlannedTrade]:
    """Trades that move the portfolio towards the targets. Sells and covers come first, so
    their proceeds are free before buys and new shorts. Holdings outside the universe are closed.
    Pairs in `frozen` (halted on the exchange) get no trades, so the activity rule still finds a
    pair it can trade."""
    trades: List[PlannedTrade] = []
    pairs = list(decision.targets) + [p for p in weights if p not in decision.targets]
    for pair in [p for p in pairs if p not in frozen]:
        target = decision.targets.get(pair, 0.0)
        current = weights.get(pair, 0.0)
        if pair not in decision.targets:
            reason = EXIT_UNIVERSE
        else:
            reason = decision.reasons.get(pair, REBALANCE)
        if current * target < 0:
            # Switching sides: close the old position, then open the new one.
            trades.append(PlannedTrade(pair, SELL if current > 0 else COVER, abs(current) * equity,
                                       True, current, 0.0, reason))
            trades.append(PlannedTrade(pair, BUY if target > 0 else SHORT, abs(target) * equity,
                                       False, 0.0, target, reason))
            continue
        trade = _towards(pair, current, target, equity, reason, cfg, min_position_weight)
        if trade is not None:
            trades.append(trade)

    if not trades and activity_due(ts, last_fill_ts, cfg):
        trade = activity_trade(decision, weights, equity, cfg, frozen)
        if trade is not None:
            trades.append(trade)

    trades.sort(key=lambda t: t.side not in (SELL, COVER))
    return trades


def _towards(pair: str, current: float, target: float, equity: float, reason: str,
             cfg: ExecutionConfig, min_position_weight: float) -> Optional[PlannedTrade]:
    """One trade on the same side of zero (or none, if the gap is too small to bother)."""
    delta = target - current
    usd = abs(delta) * equity
    if usd < cfg.min_trade_usd:
        return None
    closing = target == 0.0 and current != 0.0
    opening = abs(current) < min_position_weight and target != 0.0
    if not (closing or opening or abs(delta) >= cfg.rebalance_threshold):
        return None
    if not (closing or opening) and reason not in EXITS:
        reason = REBALANCE
    if opening and target < 0:
        reason = SHORT_ENTRY
    short_side = target < 0 or current < 0
    if short_side:
        side = SHORT if delta < 0 else COVER
    else:
        side = BUY if delta > 0 else SELL
    return PlannedTrade(pair, side, usd, closing, current, target, reason)


def activity_due(ts: int, last_fill_ts: int, cfg: ExecutionConfig) -> bool:
    """True when nothing has filled in the current activity block and it is nearly over."""
    block_ms = cfg.activity_block_hours * HOUR_MS
    block_start = ts // block_ms * block_ms
    if last_fill_ts >= block_start:
        return False
    hours_left = (block_start + block_ms - ts) / HOUR_MS
    return hours_left <= cfg.activity_trigger_hours_left


def activity_trade(decision: Decision, weights: Dict[str, float], equity: float,
                   cfg: ExecutionConfig, frozen: AbstractSet[str] = frozenset()) -> Optional[PlannedTrade]:
    """Rebalance the long position furthest from its target, by at least activity_min_usd.
    Halted pairs are skipped: the exchange would refuse the order."""
    pairs = [p for p, target in decision.targets.items()
             if (target > 0 or weights.get(p, 0.0) > 0) and p not in frozen
             and decision.reasons.get(p) != HOLD_HALTED]
    if not pairs or equity <= 0:
        return None
    pair = max(pairs, key=lambda p: abs(decision.targets[p] - weights.get(p, 0.0)))
    current = weights.get(pair, 0.0)
    target = decision.targets[pair]
    side = BUY if target >= current else SELL
    usd = max(abs(target - current) * equity, cfg.activity_min_usd)
    if side == SELL:
        usd = min(usd, current * equity)
        if usd < cfg.min_trade_usd:
            side, usd = BUY, cfg.activity_min_usd
    return PlannedTrade(pair, side, usd, False, current, target, ACTIVITY)
