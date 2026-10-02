"""The trading strategy: long-only trend following on hourly bars.

The same Strategy object drives the backtest and the live bot. It turns the latest
closed bars and the current portfolio into target weights; it never talks to the
exchange. Every number below is a field of StrategyConfig.

  Regime   BTC above its 200-hour EMA: up to 75% invested and 4 positions.
           Otherwise up to 25% and 2 positions, with PAXG first in line.
  Trend    a coin is eligible while EMA50 > EMA200 and its close is above EMA200.
  Ranking  0.5 x 72h return + 0.5 x 168h return, each divided by the volatility
           over that horizon. Free slots go to the best-ranked eligible coins.
  Entry    only into eligible coins with RSI(14) <= 70, not cooling down after a stop.
  Sizing   weights proportional to 1 / (ATR / price), so each position carries a
           similar amount of risk, scaled to the exposure limit and capped at 15%.
  Exits    EMA50 falls below EMA200, or the close drops 8 ATR below the highest
           close since entry (then no re-entry into that coin for 24 hours).
           A held coin is never sold just for ranking lower; only when risk-off
           cuts the number of positions are the lowest-ranked ones closed.
  Brake    once equity is 4% below its peak, trend positions are halved until the
           drawdown is back under 2%.
  Core     5% of equity stays in PAXG at all times, so there is always a position
           to rebalance on quiet days (see the activity rule in planner.py).

The baseline in the project brief used EMA 20/100, 24h/72h momentum and a 2.5 ATR
stop, and also re-ranked held coins every hour. Backtests on two separate years
showed heavy churn (positions held for a median of 6 hours) and losses after fees,
so the slower settings above replaced it. See the README for the comparison.
"""
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from bot.config import StrategyConfig
from bot.indicators import IndicatorSet, Signal
from bot.market_data import HOUR_MS, Bar

# Reasons attached to each target, recorded with every decision and order.
ENTRY = "entry"
HOLD = "hold"
CORE = "core"
EXIT_TREND = "exit_trend"
EXIT_STOP = "exit_stop"
EXIT_REGIME = "exit_regime"
HOLD_NO_DATA = "hold_no_data"


@dataclass
class PositionInfo:
    entry_ts: int         # ms; when the trend position was opened
    highest_close: float  # highest hourly close since entry, for the trailing stop


@dataclass
class StrategyState:
    """What the strategy remembers between hours. The live bot saves it to disk every cycle."""
    positions: Dict[str, PositionInfo] = field(default_factory=dict)  # open trend positions
    cooldown_until: Dict[str, int] = field(default_factory=dict)      # pair -> ms
    peak_equity: float = 0.0
    brake_on: bool = False
    last_fill_ts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyState":
        return cls(
            positions={p: PositionInfo(**v) for p, v in data.get("positions", {}).items()},
            cooldown_until={p: int(v) for p, v in data.get("cooldown_until", {}).items()},
            peak_equity=float(data.get("peak_equity", 0.0)),
            brake_on=bool(data.get("brake_on", False)),
            last_fill_ts=int(data.get("last_fill_ts", 0)),
        )


@dataclass
class Decision:
    ts: int
    risk_on: bool
    exposure_limit: float
    drawdown: float
    brake_on: bool
    targets: Dict[str, float]   # target weight (fraction of equity) for every pair in the universe
    reasons: Dict[str, str]     # why each pair's target is what it is
    scores: Dict[str, float]    # ranking score of every pair with data


class Strategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.indicators = {
            pair: IndicatorSet(cfg.fast_ema, cfg.slow_ema, cfg.regime_ema, cfg.atr_period,
                               cfg.rsi_period, cfg.momentum_short, cfg.momentum_long,
                               cfg.volatility_window)
            for pair in cfg.universe
        }

    def update(self, pair: str, bar: Bar) -> None:
        indicators = self.indicators.get(pair)
        if indicators is not None:
            indicators.update(bar)

    def signals(self) -> Dict[str, Signal]:
        out = {}
        for pair, indicators in self.indicators.items():
            signal = indicators.signal()
            if signal is not None:
                out[pair] = signal
        return out

    def score(self, s: Signal) -> float:
        """Momentum over two horizons, each divided by the volatility expected over that horizon."""
        c = self.cfg
        if s.volatility <= 0:
            return 0.0
        short = s.return_short / (s.volatility * math.sqrt(c.momentum_short))
        long = s.return_long / (s.volatility * math.sqrt(c.momentum_long))
        return c.momentum_short_weight * short + (1.0 - c.momentum_short_weight) * long

    def decide(self, ts: int, equity: float, weights: Dict[str, float],
               state: StrategyState) -> Decision:
        """Target weights for every pair, given current holdings as fractions of equity.

        Updates the drawdown brake and the trailing-stop highs in `state`. Entries and
        exits are recorded later by reconcile(), once we know what actually traded.
        """
        c = self.cfg
        signals = self.signals()
        drawdown = self._update_brake(equity, state)

        regime = signals.get(c.regime_pair)
        risk_on = regime is not None and regime.close > regime.ema_regime
        exposure = c.risk_on_exposure if risk_on else c.risk_off_exposure
        max_positions = c.max_positions_risk_on if risk_on else c.max_positions_risk_off

        reasons: Dict[str, str] = {}
        exiting = set()
        for pair, position in state.positions.items():
            s = signals.get(pair)
            if s is None:
                continue
            position.highest_close = max(position.highest_close, s.close)
            if s.ema_fast < s.ema_slow:
                reasons[pair] = EXIT_TREND
                exiting.add(pair)
            elif s.close < position.highest_close - c.stop_atr_multiple * s.atr:
                reasons[pair] = EXIT_STOP
                exiting.add(pair)

        scores = {pair: self.score(s) for pair, s in signals.items()}
        held = [p for p in state.positions if p not in exiting and p in signals]
        candidates = [p for p, s in signals.items()
                      if p not in state.positions and self._can_enter(p, s, ts, state)]

        ranked = sorted(held + candidates, key=lambda p: scores[p], reverse=True)
        if not risk_on and c.defensive_pair in ranked:
            ranked.remove(c.defensive_pair)
            ranked.insert(0, c.defensive_pair)
        rank = {p: i for i, p in enumerate(ranked)}

        # Held coins stay until an exit rule fires; only a smaller risk-off limit cuts them.
        keep = sorted(held, key=rank.get)[:max_positions]
        for pair in held:
            if pair not in keep:
                reasons[pair] = EXIT_REGIME
        slots = max_positions - len(keep)
        entries = [p for p in ranked if p in candidates and rank[p] < max_positions][:max(slots, 0)]

        targets = {pair: 0.0 for pair in c.universe}
        core = c.core_weight if c.defensive_pair in targets else 0.0
        sized = self._size(keep + entries, signals, max(exposure - core, 0.0))
        for pair, weight in sized.items():
            if state.brake_on:
                weight *= c.brake_factor
            targets[pair] = weight
            reasons[pair] = HOLD if pair in keep else ENTRY

        for pair in state.positions:
            if pair not in signals:
                # No data to judge it by: leave the position as it is (the core is added below).
                held_core = core if pair == c.defensive_pair else 0.0
                targets[pair] = max(weights.get(pair, 0.0) - held_core, 0.0)
                reasons[pair] = HOLD_NO_DATA

        if core > 0:
            targets[c.defensive_pair] += core
            reasons.setdefault(c.defensive_pair, CORE)

        return Decision(ts=ts, risk_on=risk_on, exposure_limit=exposure, drawdown=drawdown,
                        brake_on=state.brake_on, targets=targets, reasons=reasons, scores=scores)

    def reconcile(self, ts: int, weights: Dict[str, float], decision: Optional[Decision],
                  state: StrategyState) -> None:
        """Bring the remembered trend positions in line with what is actually held after trading.

        Works from real holdings rather than from what was ordered, so a failed order or a
        restart cannot leave the state out of step with the account.
        """
        c = self.cfg
        signals = self.signals()
        for pair in c.universe:
            weight = weights.get(pair, 0.0)
            if pair == c.defensive_pair:
                weight -= c.core_weight
            held = weight > c.min_position_weight
            if held and pair not in state.positions:
                # The core alone must not register as a trend position when PAXG drifts up.
                if pair == c.defensive_pair and (decision is None or decision.reasons.get(pair) != ENTRY):
                    continue
                close = signals[pair].close if pair in signals else 0.0
                state.positions[pair] = PositionInfo(entry_ts=ts, highest_close=close)
            elif not held and pair in state.positions:
                del state.positions[pair]
                if decision is not None and decision.reasons.get(pair) == EXIT_STOP:
                    state.cooldown_until[pair] = ts + c.stop_cooldown_hours * HOUR_MS
        for pair in [p for p, until in state.cooldown_until.items() if until <= ts]:
            del state.cooldown_until[pair]

    def _can_enter(self, pair: str, s: Signal, ts: int, state: StrategyState) -> bool:
        c = self.cfg
        return (s.ema_fast > s.ema_slow
                and s.close > s.ema_slow
                and s.rsi <= c.rsi_max_entry
                and state.cooldown_until.get(pair, 0) <= ts)

    def _size(self, pairs: List[str], signals: Dict[str, Signal], budget: float) -> Dict[str, float]:
        """Inverse-ATR weights scaled to the budget, each capped at max_weight."""
        c = self.cfg
        inverse_risk = {p: signals[p].close / signals[p].atr for p in pairs if signals[p].atr > 0}
        total = sum(inverse_risk.values())
        if total <= 0:
            return {}
        weights = {}
        for pair, x in inverse_risk.items():
            cap = c.max_weight - (c.core_weight if pair == c.defensive_pair else 0.0)
            weights[pair] = min(budget * x / total, max(cap, 0.0))
        return weights

    def _update_brake(self, equity: float, state: StrategyState) -> float:
        c = self.cfg
        state.peak_equity = max(state.peak_equity, equity)
        drawdown = 1.0 - equity / state.peak_equity if state.peak_equity > 0 else 0.0
        if state.brake_on and drawdown <= c.brake_release_drawdown:
            state.brake_on = False
        elif not state.brake_on and drawdown >= c.brake_drawdown:
            state.brake_on = True
        return drawdown
