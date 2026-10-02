"""Turns target weights into trades. Shared by the backtest and the live bot.

Two rules keep trading costs and compliance in balance:

  Threshold  a held position is resized only when it is off target by at least
             rebalance_threshold of equity; entries and full exits always go through.
  Activity   the competition requires trades on at least 8 days. If nothing has
             filled in the current UTC-aligned block of activity_block_hours and the
             block is nearly over, the position furthest from its target is
             rebalanced, even below the threshold. The permanent PAXG core means
             there is always a position to adjust.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from bot.config import ExecutionConfig
from bot.market_data import HOUR_MS
from bot.strategy import EXIT_REGIME, EXIT_STOP, EXIT_TREND, Decision

BUY = "BUY"
SELL = "SELL"
REBALANCE = "rebalance"
ACTIVITY = "activity"


@dataclass
class PlannedTrade:
    pair: str
    side: str              # BUY or SELL
    usd: float             # notional at the current price
    close_position: bool   # sell the entire holding
    current_weight: float
    target_weight: float
    reason: str


def plan_trades(decision: Decision, weights: Dict[str, float], equity: float, ts: int,
                last_fill_ts: int, cfg: ExecutionConfig,
                min_position_weight: float) -> List[PlannedTrade]:
    """Trades that move the portfolio towards the targets, sells first to free cash."""
    trades = []
    for pair, target in decision.targets.items():
        current = weights.get(pair, 0.0)
        delta = target - current
        usd = abs(delta) * equity
        if usd < cfg.min_trade_usd:
            continue
        closing = target <= 0.0 and current > 0.0
        opening = current < min_position_weight and target > 0.0
        if not (closing or opening or abs(delta) >= cfg.rebalance_threshold):
            continue
        reason = decision.reasons.get(pair, REBALANCE)
        if not (closing or opening) and reason not in (EXIT_TREND, EXIT_STOP, EXIT_REGIME):
            reason = REBALANCE
        trades.append(PlannedTrade(pair, BUY if delta > 0 else SELL, usd, closing,
                                   current, target, reason))

    if not trades and activity_due(ts, last_fill_ts, cfg):
        trade = activity_trade(decision, weights, equity, cfg)
        if trade is not None:
            trades.append(trade)

    trades.sort(key=lambda t: t.side != SELL)
    return trades


def activity_due(ts: int, last_fill_ts: int, cfg: ExecutionConfig) -> bool:
    """True when nothing has filled in the current activity block and it is nearly over."""
    block_ms = cfg.activity_block_hours * HOUR_MS
    block_start = ts // block_ms * block_ms
    if last_fill_ts >= block_start:
        return False
    hours_left = (block_start + block_ms - ts) / HOUR_MS
    return hours_left <= cfg.activity_trigger_hours_left


def activity_trade(decision: Decision, weights: Dict[str, float], equity: float,
                   cfg: ExecutionConfig) -> Optional[PlannedTrade]:
    """Rebalance the position furthest from its target, by at least activity_min_usd."""
    pairs = [p for p, target in decision.targets.items() if target > 0 or weights.get(p, 0.0) > 0]
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
