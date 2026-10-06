"""The trading strategy, on hourly bars: a momentum rotation beside a trend book.

The same Strategy object drives the backtest and the live bot. It turns the latest closed bars
and the current portfolio into target weights; it never talks to the exchange. Every number
below is a field of StrategyConfig (bot/config.py).

Reading guide. decide() runs once an hour: _rotation() gives the rotation's weights, the book is
planned (the long-short trend book in _long_short_book(), or the defensive book in the body of
decide()), and _merge() combines the two. The calls named _research_*() and everything in
bot/research_rules.py are research options, all off by default: with the live settings they
return their input unchanged.

  Rotation  70% of equity (rotation_weight) holds the 2 coins with the strongest positive
            336h return, re-chosen daily at 00:00 UTC, while BTC's 168h EMA is above its 672h
            EMA, and leaves at once when it is not. An empty slot goes to PAXG while PAXG's own
            336h return is positive, else to cash.

The book, the other 30%, is one of:

  Long-short trend book (book_mode "long_short"; the competition account since 5 October 2026,
  through config/comp.json)
    Direction  long while the coin's 240h EMA is above its 960h EMA, short while below; PAXG
               is left out.
    Sizing     each coin's share of the book is its inverse volatility over the sum for all
               coins, and a coin not held leaves its share in cash. Shorts are 1x on Roostoo,
               so longs plus shorts never exceed the book's share.
    Stops      a short is covered 10 ATR above its lowest close since entry, with no new short
               in that coin for 24 hours.
    Crowding   no short in a coin whose perpetual funding rate averaged below zero over the
               last 3 days (crowded shorts); the live bot fetches the rates from Binance's
               futures API and passes them in as external_scores.

  Defensive trend book (book_mode "trend", the default)
    Regime   BTC above its 200-hour EMA: up to 75% invested and 8 positions. Otherwise up to
             25% and 3 positions, with PAXG first in line.
    Trend    a coin is eligible while EMA50 > EMA200 and its close is above EMA200.
    Ranking  the lowest volatility of hourly returns over the last 168 hours first.
    Entry    only into eligible coins with RSI(14) <= 70, not cooling down after a stop.
    Sizing   equal risk contribution from the last 336 hours' covariance (each position adds
             the same share of the book's variance), scaled to the exposure limit and capped
             at 15%.
    Exits    EMA50 falls below EMA200, or the close drops 8 ATR below the highest close since
             entry (then no re-entry for 24 hours). A held coin is never sold just for ranking
             lower; only a risk-off cut in positions closes the lowest-ranked.
    Brake    once the book is 4% below its peak, its positions are halved until it is back
             within 2%.
    Core     5% of the book stays in PAXG.

  Halts      a pair Roostoo will not trade (decide()'s `frozen`, with plan_around_halts) is held
             exactly as it is and never entered, and the rest is planned around it.

The project brief's baseline (EMA 20/100, 24h/72h momentum, 2.5 ATR stops, hourly re-ranking)
churned and lost money after fees, so slower settings replaced it, and research found that low
volatility, not momentum, predicted which coin would do better next. See the README.
"""
import math
from dataclasses import asdict, dataclass, field
from typing import AbstractSet, Dict, List, Optional, Tuple

from bot.config import StrategyConfig
from bot.indicators import IndicatorSet, Signal
from bot.optimize import covariance, erc_weights, erc_weights_fixed, min_variance
from bot.market_data import HOUR_MS, Bar

# Reasons attached to each target, recorded with every decision and order (bot/reasons.py).
from bot.reasons import (CORE, ENTRY, EXIT_REGIME, EXIT_SHORT, EXIT_SHORT_STOP, EXIT_STOP, EXIT_TREND,
                         HOLD, HOLD_HALTED, HOLD_NO_DATA, LS_LONG, LS_SHORT, ROTATION)
from bot.research_rules import ResearchRules, _normal_scores

@dataclass
class PositionInfo:
    entry_ts: int         # ms; when the trend position was opened
    highest_close: float  # highest hourly close since entry, for the trailing stop


@dataclass
class ShortInfo:
    entry_ts: int         # ms; when the short was opened
    lowest_close: float   # lowest hourly close since entry, for the trailing stop


@dataclass
class StrategyState:
    """What the strategy remembers between hours. The live bot saves it to disk every cycle."""
    positions: Dict[str, PositionInfo] = field(default_factory=dict)  # open trend positions
    cooldown_until: Dict[str, int] = field(default_factory=dict)      # pair -> ms
    peak_equity: float = 0.0
    brake_on: bool = False
    last_fill_ts: int = 0
    shorts: Dict[str, ShortInfo] = field(default_factory=dict)        # open short positions
    short_cooldown_until: Dict[str, int] = field(default_factory=dict)
    rotation_plan: Dict[str, float] = field(default_factory=dict)      # the sleeve's weights
    rotation_plan_ts: int = 0                                          # when they were chosen
    rotation_highs: Dict[str, float] = field(default_factory=dict)     # each pick's high since picked
    rotation_entry: Dict[str, int] = field(default_factory=dict)       # when each pick was first picked
    rotation_brake_on: bool = False
    rotation_on_since: int = 0                                          # when the sleeve's filter last turned on
    regime_on: int = -1                                                 # with hysteresis: -1 unknown, 0 off, 1 on
    rotation_plan_on: int = -1                                          # filter state when the plan was made
    rotation_filter_on: int = -1
    equity_history: List[float] = field(default_factory=list)          # hourly account values (recent)
    rotation_cooldown: Dict[str, int] = field(default_factory=dict)    # stopped picks barred until (ms)
    # The non-rotation book's own value, so its drawdown brake ignores the rotation sleeve.
    book_nav: float = 1.0
    book_peak: float = 1.0
    book_weights: Dict[str, float] = field(default_factory=dict)       # its last targets
    book_closes: Dict[str, float] = field(default_factory=dict)        # prices when they were set
    rotation_entry_close: Dict[str, float] = field(default_factory=dict)  # research: each pick's entry price
    rotation_trimmed: Dict[str, float] = field(default_factory=dict)      # research: share kept after profit
    window_index: int = -1              # research: the competition window of the profits lock
    window_equity: float = 0.0          # research: the account's value when that window began
    profit_locked: bool = False         # research: profits secured for the rest of the window
    risk_day: int = -1                  # research (round 75): the UTC day of risk_day_equity
    risk_day_equity: float = 0.0        # the account's value at that day's 00:00
    risk_until: Dict[str, int] = field(default_factory=dict)  # breaker -> ms it stays on
    dd_tier: int = 0                    # the drawdown ladder's step, and when it was reached
    dd_tier_since: int = 0
    coin_block_until: Dict[str, int] = field(default_factory=dict)  # B3: no adds until (ms)
    capitulation: Dict[str, List[float]] = field(default_factory=dict)  # E5: pair -> [entry ms, price, ATR]
    paper: Dict[str, List[float]] = field(default_factory=dict)  # E3: open paper trades, key -> [side, price]
    paper_last: Dict[str, float] = field(default_factory=dict)   # E3: key -> last closed paper return
    paper_skip: Dict[str, float] = field(default_factory=dict)   # E3: book legs skipped, key -> side

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
            shorts={p: ShortInfo(**v) for p, v in data.get("shorts", {}).items()},
            short_cooldown_until={p: int(v) for p, v in data.get("short_cooldown_until", {}).items()},
            rotation_plan={p: float(v) for p, v in data.get("rotation_plan", {}).items()},
            rotation_plan_ts=int(data.get("rotation_plan_ts", 0)),
            rotation_highs={p: float(v) for p, v in data.get("rotation_highs", {}).items()},
            rotation_entry={p: int(v) for p, v in data.get("rotation_entry", {}).items()},
            rotation_brake_on=bool(data.get("rotation_brake_on", False)),
            rotation_on_since=int(data.get("rotation_on_since", 0)),
            regime_on=int(data.get("regime_on", -1)),
            rotation_plan_on=int(data.get("rotation_plan_on", -1)),
            rotation_filter_on=int(data.get("rotation_filter_on", -1)),
            equity_history=[float(v) for v in data.get("equity_history", [])],
            rotation_cooldown={p: int(v) for p, v in data.get("rotation_cooldown", {}).items()},
            book_nav=float(data.get("book_nav", 1.0)),
            book_peak=float(data.get("book_peak", 1.0)),
            book_weights={p: float(v) for p, v in data.get("book_weights", {}).items()},
            book_closes={p: float(v) for p, v in data.get("book_closes", {}).items()},
            rotation_entry_close={p: float(v) for p, v in data.get("rotation_entry_close", {}).items()},
            rotation_trimmed={p: float(v) for p, v in data.get("rotation_trimmed", {}).items()},
            window_index=int(data.get("window_index", -1)),
            window_equity=float(data.get("window_equity", 0.0)),
            profit_locked=bool(data.get("profit_locked", False)),
            risk_day=int(data.get("risk_day", -1)),
            risk_day_equity=float(data.get("risk_day_equity", 0.0)),
            risk_until={k: int(v) for k, v in data.get("risk_until", {}).items()},
            dd_tier=int(data.get("dd_tier", 0)),
            dd_tier_since=int(data.get("dd_tier_since", 0)),
            coin_block_until={p: int(v) for p, v in data.get("coin_block_until", {}).items()},
            capitulation={p: [float(x) for x in v] for p, v in data.get("capitulation", {}).items()},
            paper={k: [float(x) for x in v] for k, v in data.get("paper", {}).items()},
            paper_last={k: float(v) for k, v in data.get("paper_last", {}).items()},
            paper_skip={k: float(v) for k, v in data.get("paper_skip", {}).items()},
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
    rotation: Dict[str, float] = field(default_factory=dict)  # the rotation sleeve's own weights


class Strategy(ResearchRules):
    def __init__(self, cfg: StrategyConfig, pairs: Optional[List[str]] = None,
                 external_scores: Optional[Dict[int, Dict[str, float]]] = None):
        """`pairs` (default: the universe) are the pairs whose indicators are kept. The backtest
        tracks every candidate so the universe can change month by month. `external_scores`
        (research only) maps each day's 00:00 bar time to model scores, for the "external"
        rankings."""
        self.cfg = cfg
        self.external_scores = external_scores or {}
        self.indicators = {
            pair: IndicatorSet(cfg.fast_ema, cfg.slow_ema, cfg.regime_ema, cfg.atr_period,
                               cfg.rsi_period, cfg.momentum_short, cfg.momentum_long,
                               cfg.volatility_window, cfg.rotation_lookback,
                               cfg.rotation_trend_fast, cfg.rotation_trend_slow,
                               tuple(cfg.slow_filter) if cfg.slow_filter else None,
                               tuple(cfg.ls_trend) if cfg.book_mode in ("long_short", "hybrid", "overlay") else None,
                               cfg.short_regime_hours or cfg.ls_absorb_sma_hours,
                               [tuple(p) for p in cfg.ls_ensemble] if cfg.book_mode != "trend" else None,
                               tuple(cfg.ls_short_trend) if cfg.ls_short_trend and cfg.book_mode != "trend" else None)
            for pair in (pairs if pairs is not None else cfg.universe)
        }

    def update(self, pair: str, bar: Bar) -> None:
        indicators = self.indicators.get(pair)
        if indicators is not None:
            indicators.update(bar)

    def signals(self) -> Dict[str, Signal]:
        """Signals for the pairs in the current universe that have finished warming up."""
        out = {}
        universe = set(self.cfg.universe)
        for pair, indicators in self.indicators.items():
            if pair not in universe:
                continue
            signal = indicators.signal()
            if signal is not None:
                out[pair] = signal
        return out

    def score(self, s: Signal) -> float:
        """Ranking score: higher is better."""
        c = self.cfg
        if c.ranking == "low_volatility":
            return -s.volatility
        # Momentum over two horizons, each divided by the volatility expected over that horizon.
        if s.volatility <= 0:
            return 0.0
        short = s.return_short / (s.volatility * math.sqrt(c.momentum_short))
        long = s.return_long / (s.volatility * math.sqrt(c.momentum_long))
        return c.momentum_short_weight * short + (1.0 - c.momentum_short_weight) * long

    def _external(self, ts: int) -> Dict[str, float]:
        """The model scores made at the latest 00:00 bar that has closed by ts."""
        day = (ts - HOUR_MS) // (24 * HOUR_MS) * (24 * HOUR_MS)
        return self.external_scores.get(day, {})

    def _scores(self, signals: Dict[str, Signal], ts: int = 0) -> Dict[str, float]:
        """Ranking score of every pair with data. The composite ranking is cross-sectional: the
        sum of normal scores of each coin's rank by low volatility, narrow spread and Kalman
        trend strength. The external ranking (research) orders the coins by model score and
        keeps the defensive pair first, as low volatility does."""
        if self.cfg.ranking == "external":
            table = self._external(ts)
            if table:
                top = max(table.values()) + 1.0
                low = min(table.values()) - 1.0
                return {p: top if p == self.cfg.defensive_pair else table.get(p, low) for p in signals}
            return {pair: self.score(s) for pair, s in signals.items()}
        if self.cfg.ranking != "composite" or len(signals) < 2:
            return {pair: self.score(s) for pair, s in signals.items()}
        total = {pair: 0.0 for pair in signals}
        for key in (lambda s: -s.volatility, lambda s: -s.spread, lambda s: s.trend_strength):
            for pair, z in _normal_scores({p: key(s) for p, s in signals.items()}).items():
                total[pair] += z
        return total

    def decide(self, ts: int, equity: float, weights: Dict[str, float],
               state: StrategyState, frozen: AbstractSet[str] = frozenset()) -> Decision:
        """Target weights for every pair, given current holdings as fractions of equity.

        Updates the drawdown brake and the trailing-stop highs in `state`. Entries and
        exits are recorded later by reconcile(), once we know what actually traded.
        `frozen` pairs cannot be traded now (halted on the exchange): they keep their
        current weight, and the rest of the portfolio is planned around them.
        """
        c = self.cfg
        account = weights
        signals = self.signals()
        drawdown = self._update_brake(equity, state, signals)
        rotation = self._rotation(ts, signals, state, frozen)
        rotation = self._research_rotation_overrides(rotation, account, equity, state, frozen)
        weights = self._defensive_weights(weights, rotation)

        regime = signals.get(c.regime_pair)
        risk_on = regime is not None and regime.close > regime.ema_regime
        risk_on = self._research_regime(risk_on, regime, signals, state)
        exposure = c.risk_on_exposure if risk_on else c.risk_off_exposure
        max_positions = c.max_positions_risk_on if risk_on else c.max_positions_risk_off
        if c.book_mode == "long_short" or (c.book_mode == "hybrid" and self._bear(regime)):
            # The long-short trend book in place of the defensive book (the competition account).
            sides = "short" if c.book_mode == "hybrid" else None
            rotation, share = self._research_absorb(rotation, regime, frozen)
            targets, reasons = self._long_short_book(signals, weights, frozen, regime, sides, state, ts)
            return self._merge(ts, risk_on, 1.0, drawdown, state, signals, targets, reasons, {},
                               rotation, account, frozen, share, equity)

        reasons: Dict[str, str] = {}
        exiting = set()
        for pair, position in state.positions.items():
            s = signals.get(pair)
            if s is None:
                continue
            position.highest_close = max(position.highest_close, s.close)
            if pair in frozen:
                continue                      # halted: it cannot be sold, whatever the signal
            if s.ema_fast < s.ema_slow * (1.0 - c.trend_exit_band):
                reasons[pair] = EXIT_TREND
                exiting.add(pair)
            elif (s.close < position.highest_close - c.stop_atr_multiple * s.atr
                  or (c.book_donchian_exit > 0 and s.close < self._channel(pair, c.book_donchian_exit, high=False))):
                reasons[pair] = EXIT_STOP
                exiting.add(pair)

        scores = self._scores(signals, ts)
        stuck = [p for p in state.positions if p in frozen]
        held = [p for p in state.positions if p not in exiting and p in signals and p not in frozen]
        candidates = [p for p, s in signals.items()
                      if p not in state.positions and p not in frozen and self._can_enter(p, s, ts, state)
                      and not (c.book_excludes_rotation and rotation.get(p, 0.0) > 0)]

        ranked = sorted(held + candidates, key=lambda p: scores[p], reverse=True)
        if not risk_on and c.defensive_pair in ranked:
            ranked.remove(c.defensive_pair)
            ranked.insert(0, c.defensive_pair)
        rank = {p: i for i, p in enumerate(ranked)}

        # Held coins stay until an exit rule fires; only a smaller risk-off limit cuts them.
        # Halted ones fill their slots first, since they cannot be sold.
        limit = max(max_positions - len(stuck), 0)
        keep = sorted(held, key=rank.get)[:limit]
        for pair in held:
            if pair not in keep:
                reasons[pair] = EXIT_REGIME
        slots = limit - len(keep)
        entries = [p for p in ranked if p in candidates and rank[p] < limit][:max(slots, 0)]

        targets = {pair: 0.0 for pair in c.universe}
        core = c.core_weight if c.defensive_pair in targets else 0.0
        fixed = {p: max(weights.get(p, 0.0) - (core if p == c.defensive_pair else 0.0), 0.0)
                 for p in stuck}
        sized = self._size(keep + entries, signals, max(exposure - core, 0.0), fixed)
        for pair, weight in sized.items():
            if state.brake_on:
                weight *= c.brake_factor
            targets[pair] = weight
            reasons[pair] = HOLD if pair in keep else ENTRY
        for pair, weight in fixed.items():
            targets[pair] = weight
            reasons[pair] = HOLD_HALTED

        for pair in state.positions:
            if pair not in signals and pair in targets and pair not in frozen:
                # No data to judge it by: leave the position as it is (the core is added below).
                held_core = core if pair == c.defensive_pair else 0.0
                targets[pair] = max(weights.get(pair, 0.0) - held_core, 0.0)
                reasons[pair] = HOLD_NO_DATA

        if core > 0:
            targets[c.defensive_pair] += core
            reasons.setdefault(c.defensive_pair, CORE)

        if c.book_mode == "overlay":
            self._overlay(ts, signals, weights, state, targets, reasons, regime, frozen)
        elif c.short_exposure > 0 or state.shorts:
            self._shorts(ts, signals, scores, weights, state, targets, reasons,
                         set(keep + entries) | set(rotation), frozen)
        return self._merge(ts, risk_on, exposure, drawdown, state, signals, targets, reasons, scores,
                           rotation, account, frozen, equity=equity)

    def _merge(self, ts: int, risk_on: bool, exposure: float, drawdown: float, state: StrategyState,
               signals: Dict[str, Signal], targets: Dict[str, float], reasons: Dict[str, str],
               scores: Dict[str, float], rotation: Dict[str, float], account: Dict[str, float],
               frozen: AbstractSet[str], share: Optional[float] = None, equity: float = 0.0) -> Decision:
        """The book's targets (fractions of its share, by default 1 - rotation_weight) and the
        rotation's, as fractions of equity."""
        c = self.cfg
        book = targets
        if c.rotation_weight > 0:
            state.book_weights = {p: t for p, t in targets.items() if t != 0.0}
            state.book_closes = {p: signals[p].close for p in state.book_weights if p in signals}
            share = 1.0 - c.rotation_weight if share is None else share
            targets = {p: share * t for p, t in targets.items()}
            for pair, weight in rotation.items():
                if targets.get(pair, 0.0) == 0.0:
                    reasons[pair] = ROTATION
                targets[pair] = targets.get(pair, 0.0) + c.rotation_weight * weight

        targets, exposure = self._research_window_lock(ts, equity, targets, book, rotation, signals, exposure,
                                                       state, frozen, reasons)
        targets = self._research_risk_breakers(ts, equity, targets, rotation, account, signals, state, frozen)
        if frozen:
            targets = _hold_frozen(targets, account, frozen, reasons)
        return Decision(ts=ts, risk_on=risk_on, exposure_limit=exposure, drawdown=drawdown,
                        brake_on=state.brake_on, targets=targets, reasons=reasons, scores=scores,
                        rotation=rotation)

    def reconcile(self, ts: int, weights: Dict[str, float], decision: Optional[Decision],
                  state: StrategyState) -> None:
        """Bring the remembered trend positions in line with what is actually held after trading.

        Works from real holdings rather than from what was ordered, so a failed order or a
        restart cannot leave the state out of step with the account.
        """
        c = self.cfg
        signals = self.signals()
        rotation = decision.rotation if decision is not None else state.rotation_plan
        weights = self._defensive_weights(weights, rotation)
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
        # Pairs that left the universe are sold by the planner; forget them once they are gone.
        universe = set(c.universe)
        for book in (state.positions, state.shorts):
            for pair in [p for p in book if p not in universe and abs(weights.get(p, 0.0)) <= c.min_position_weight]:
                del book[pair]

        for pair in c.universe:
            short = weights.get(pair, 0.0) < -c.min_position_weight
            if short and pair not in state.shorts:
                close = signals[pair].close if pair in signals else 0.0
                state.shorts[pair] = ShortInfo(entry_ts=ts, lowest_close=close if close > 0 else 1e18)
            elif not short and pair in state.shorts:
                del state.shorts[pair]
                if decision is not None and decision.reasons.get(pair) == EXIT_SHORT_STOP:
                    state.short_cooldown_until[pair] = ts + c.stop_cooldown_hours * HOUR_MS
        for pair in [p for p, until in state.short_cooldown_until.items() if until <= ts]:
            del state.short_cooldown_until[pair]

    def _rotation(self, ts: int, signals: Dict[str, Signal], state: StrategyState,
                  frozen: AbstractSet[str] = frozenset()) -> Dict[str, float]:
        """The rotation sleeve's weights (summing to at most 1), chosen afresh at each rebalance.
        Halted coins are not picked, and one already held keeps its weight and slot."""
        c = self.cfg
        if c.rotation_weight <= 0:
            return {}
        regime = signals.get(c.regime_pair)
        trend_on = regime is not None and regime.ema_trend_fast > regime.ema_trend_slow
        trend_on = self._research_trend_filter(trend_on, ts, signals, state, regime)
        hour = ts // HOUR_MS
        last = state.rotation_plan_ts // HOUR_MS
        due = (state.rotation_plan_ts == 0 or hour - last >= c.rotation_rebalance_hours
               or ((hour - c.rotation_rebalance_offset) % c.rotation_rebalance_hours == 0 and hour != last))
        if c.rotation_rebalance_at:                                       # research, round 79
            due = state.rotation_plan_ts == 0 or (hour % 24 in c.rotation_rebalance_at and hour != last)
        due = self._research_entry_due(due, trend_on, hour, last, state)
        if due:
            state.rotation_plan_on = int(trend_on)
        if due and (c.rotation_ensemble or c.rotation_adaptive_lookbacks):
            self._research_ensemble(ts, signals, state, frozen, trend_on)
        elif due:
            plan = {p: w for p, w in state.rotation_plan.items() if p in frozen}
            stuck = len([p for p in plan if p != c.defensive_pair])
            if trend_on:
                core = c.rotation_core_share if c.regime_pair in signals else 0.0
                if core > 0:
                    plan[c.regime_pair] = core
                rising = {p: s.return_rotation for p, s in signals.items()
                          if p != c.defensive_pair and s.return_rotation > 0 and p not in frozen
                          and not (core > 0 and p == c.regime_pair)
                          and state.rotation_cooldown.get(p, 0) <= ts}
                rising, alts = self._research_candidates(rising, signals, ts, regime)
                ranked = sorted(rising, key=rising.get, reverse=True)
                picks = ranked[:max(c.rotation_top - stuck, 0)]
                ranked, picks, concentrated = self._research_picks(
                    ranked, picks, rising, signals, state, ts, frozen, stuck)
                state.rotation_entry = {p: state.rotation_entry.get(p, ts) if p in state.rotation_plan else ts
                                        for p in picks}
                # The picks fill len(picks) of rotation_top slots; how they share it is the
                # weighting. Unfilled slots go to PAXG or cash below, whatever the weighting.
                filled = (1.0 - core) * (1.0 if concentrated else len(picks) / c.rotation_top)
                for pair, weight in self._rotation_weights(picks, signals).items():
                    plan[pair] = filled * weight
                self._research_reapply_trims(plan, state)
                self._research_btc_fill(plan, picks, alts, regime, frozen, stuck)
            elif c.rotation_shorts > 0 and self._shorts_allowed(regime):
                self._research_rotation_shorts(plan, signals, state, frozen, ts)
            empty = 1.0 - sum(abs(w) for w in plan.values())
            defensive = signals.get(c.defensive_pair)
            if (empty > 1e-9 and defensive is not None and defensive.return_rotation > 0
                    and c.defensive_pair not in frozen):
                plan[c.defensive_pair] = plan.get(c.defensive_pair, 0.0) + empty
            if c.rotation_cvar_limit > 0 and plan:
                plan = self._research_cvar(plan)
            state.rotation_plan = plan
            state.rotation_plan_ts = ts
        if c.rotation_stop_atr > 0 or c.rotation_stop_pct > 0 or c.rotation_donchian_exit > 0:
            self._rotation_stops(ts, signals, state, frozen)
        self._research_rotation_exits(ts, signals, state, frozen)
        plan = dict(state.rotation_plan)
        if not trend_on:
            # Leave at once when the trend filter fails; only the defensive part (and any
            # halted coin, which cannot be sold) stays, and any bear-market shorts.
            plan = {p: w for p, w in plan.items() if p == c.defensive_pair or p in frozen or w < 0}
        else:
            plan = {p: w for p, w in plan.items() if w > 0 or p in frozen}   # cover shorts at once
        return self._research_sleeve_scale(plan, regime, ts, frozen)

    def _rotation_weights(self, picks: List[str], signals: Dict[str, Signal]) -> Dict[str, float]:
        """Shares of the filled rotation sleeve (summing to 1) for the chosen coins."""
        c = self.cfg
        if not picks:
            return {}
        weights = {p: 1.0 / len(picks) for p in picks}
        if len(picks) > 1 and c.rotation_weighting == "inverse_vol":
            inverse = {p: 1.0 / signals[p].volatility for p in picks if signals[p].volatility > 0}
            if len(inverse) == len(picks):
                total = sum(inverse.values())
                weights = {p: v / total for p, v in inverse.items()}
        elif len(picks) > 1 and c.rotation_weighting in ("erc", "min_variance"):
            series = self._recent_returns(picks, c.rotation_cov_hours)
            if series is not None:
                cov = covariance(series)
                if c.rotation_weighting == "erc":
                    solved = erc_weights(cov)
                else:
                    solved = min_variance(cov, c.rotation_max_weight)
                weights = dict(zip(picks, solved))
        return _cap_weights(weights, c.rotation_max_weight)

    def _recent_returns(self, pairs: List[str], hours: int) -> Optional[List[List[float]]]:
        """The last `hours` hourly log returns of each pair, aligned at the latest bar."""
        out = []
        for pair in pairs:
            closes = list(self.indicators[pair].closes)[-(hours + 1):]
            out.append([math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0])
        length = min(len(r) for r in out)
        if length < 48:
            return None
        return [r[-length:] for r in out]

    def _defensive_weights(self, weights: Dict[str, float], rotation: Dict[str, float]) -> Dict[str, float]:
        """Holdings as fractions of the non-rotation book: the rotation sleeve's share is taken
        out, so its coins do not count as positions of the strategy above."""
        rw = self.cfg.rotation_weight
        if rw <= 0:
            return weights
        pairs = set(weights) | set(rotation)
        return {p: (weights.get(p, 0.0) - rw * rotation.get(p, 0.0)) / (1.0 - rw) for p in pairs}

    def _long_short_book(self, signals: Dict[str, Signal], weights: Dict[str, float],
                         frozen: AbstractSet[str], regime: Optional[Signal], sides: Optional[str] = None,
                         state: Optional[StrategyState] = None, ts: int = 0, slots: bool = False
                         ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """The long-short trend book (book_mode "long_short"), as fractions of the book: each
        coin long while its ls_trend fast EMA is above the slow one, short while below, weighted
        by inverse volatility to a gross of 1. Halted coins keep their weight and use up part
        of the budget; the defensive pair is left out."""
        c = self.cfg
        targets = {pair: 0.0 for pair in c.universe}
        reasons: Dict[str, str] = {}
        trend_on = regime is not None and regime.ema_trend_fast > regime.ema_trend_slow
        sides = sides or c.ls_sides
        shorts_ok = self._shorts_allowed(regime)
        fixed = {p: weights.get(p, 0.0) for p in frozen if p in targets and weights.get(p, 0.0) != 0.0}
        held_shorts = state.shorts if state is not None else {}
        liquid = None
        if c.short_top_volume > 0:
            ranked = sorted(signals, key=lambda p: self.indicators[p].dollar_sum, reverse=True)
            liquid = set(ranked[:c.short_top_volume])
        use_slots = slots
        slots = 0.0                         # gross if every eligible coin held its position
        raw = {}
        for pair, s in signals.items():
            if pair == c.defensive_pair or pair in frozen or pair not in targets:
                continue
            if c.ls_pairs == "btc" and pair != c.regime_pair:
                continue
            if s.ls_fast <= 0 or s.ls_slow <= 0 or s.volatility <= 0:
                continue
            slots += 1.0 / s.volatility
            if self._short_vetoed(pair, s, state, ts, held_shorts, liquid, reasons):
                continue
            side = 1.0 if s.ls_fast > s.ls_slow else -1.0     # long above the slow EMA, short below
            side = self._research_ls_side(pair, s, side, state, ts, reasons, trend_on, sides, shorts_ok)
            if side is None:
                continue
            raw[pair] = side / s.volatility
        raw = self._research_ls_weights(raw, ts)
        budget = max(1.0 - sum(abs(w) for w in fixed.values()), 0.0)
        gross = sum(abs(v) for v in raw.values())
        if c.short_entry_channel > 0 or c.short_stop_atr > 0 or use_slots:
            gross = slots                   # a short not timed in (or stopped out) leaves its slot in cash
        gross = self._research_ls_gross(gross, signals, frozen, targets)
        short_scale = (self._short_vol_scale() if c.short_vol_ratio else 1.0) * c.ls_short_scale  # research: 1
        for pair, v in raw.items():
            targets[pair] = v / gross * budget * (short_scale if v < 0 else 1.0)
            reasons[pair] = LS_LONG if v > 0 else LS_SHORT
        for pair, w in fixed.items():
            targets[pair] = w
            reasons[pair] = HOLD_HALTED
        self._research_idle_to_gold(targets, reasons, signals, frozen, sides)
        return targets, reasons

    def _short_vetoed(self, pair: str, s: Signal, state: Optional[StrategyState], ts: int,
                      held: Dict[str, "ShortInfo"], liquid: Optional[set], reasons: Dict[str, str]) -> bool:
        """True when the long-short book must not be short `pair` now, although its trend is down:
        outside the most traded coins, cooling down after a stop, stopped out, not yet broken
        below its entry channel, or broken above its exit channel. Only coins in a downtrend are
        vetoed; the caller skips the rest of its checks for them."""
        c = self.cfg
        if c.ls_ensemble:
            down = s.ls_vote < 0
        elif c.ls_short_trend:
            down = 0 < s.ls_short_fast < s.ls_short_slow
        else:
            down = 0 < s.ls_fast < s.ls_slow
        if not down:
            return False
        if liquid is not None and pair not in liquid:
            return True
        if c.short_exclude_external and self.external_scores and self._external(ts).get(pair, 0.0) < 0:
            return True
        if self._research_short_veto(pair, ts):
            return True
        if pair not in held:
            if (c.short_entry_min_funding > 0 and self.external_scores
                    and self._external(ts).get(pair, c.short_entry_min_funding) < c.short_entry_min_funding):
                return True
            if c.short_entry_rsi_min > 0 and s.rsi < c.short_entry_rsi_min:
                return True
        if c.short_min_funding_rank > 0 and self.external_scores:
            table = self._external(ts)
            rate = table.get(pair)
            if rate is not None and len(table) > 1:
                below = sum(1 for v in table.values() if v < rate)
                if below / (len(table) - 1) < c.short_min_funding_rank:
                    return True
        if state is not None and state.short_cooldown_until.get(pair, 0) > ts:
            return True
        if pair in held and c.short_stop_atr > 0:
            info = held[pair]
            info.lowest_close = min(info.lowest_close, s.close)
            if s.close > info.lowest_close + c.short_stop_atr * s.atr:
                reasons[pair] = EXIT_SHORT_STOP
                return True
        if c.short_entry_channel > 0:
            if pair in held:
                if s.close > self._channel(pair, c.short_exit_channel, high=True):
                    reasons[pair] = EXIT_SHORT
                    return True
            elif s.close >= self._channel(pair, c.short_entry_channel, high=False):
                return True
        return False

    def _can_enter(self, pair: str, s: Signal, ts: int, state: StrategyState) -> bool:
        c = self.cfg
        return (s.ema_fast > s.ema_slow
                and s.close > s.ema_slow
                and s.rsi <= c.rsi_max_entry
                and state.cooldown_until.get(pair, 0) <= ts)

    def _size(self, pairs: List[str], signals: Dict[str, Signal], budget: float,
              fixed: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Weights for the chosen coins scaled to the budget, each capped at max_weight: inverse
        ATR by default, or a minimum-variance or equal-risk-contribution portfolio.

        `fixed` holdings (halted coins, which cannot be resized) use up part of the budget.
        With ERC sizing the chosen coins are re-solved around them: equal risk contributions
        with the fixed weights in the covariance, so a coin that moves with a halted one gets
        less."""
        c = self.cfg
        fixed = {p: w for p, w in (fixed or {}).items() if w > 0}
        budget = max(budget - sum(fixed.values()), 0.0)
        inverse_risk = {p: signals[p].close / signals[p].atr for p in pairs if signals[p].atr > 0}
        total = sum(inverse_risk.values())
        if total <= 0:
            return {}
        shares = {p: x / total for p, x in inverse_risk.items()}
        if c.sizing in ("min_variance", "erc") and (len(shares) > 1 or fixed) and budget > 0:
            chosen = list(shares)
            stuck = [p for p in fixed if p in self.indicators] if c.sizing == "erc" else []
            series = self._recent_returns(chosen + stuck, c.rotation_cov_hours)
            if series is not None:
                cov = covariance(series)
                if stuck:
                    held = {len(chosen) + k: fixed[p] for k, p in enumerate(stuck)}
                    solved = erc_weights_fixed(cov, held, budget)[:len(chosen)]
                    shares = {p: w / budget for p, w in zip(chosen, solved)}
                elif len(chosen) > 1:
                    if c.sizing == "erc":
                        solved = erc_weights(cov)
                    else:
                        solved = min_variance(cov, min(1.0, c.max_weight / budget))
                    shares = dict(zip(chosen, solved))
        weights = {}
        for pair, share in shares.items():
            cap = c.max_weight - (c.core_weight if pair == c.defensive_pair else 0.0)
            weights[pair] = min(budget * share, max(cap, 0.0))
        return weights

    def _update_brake(self, equity: float, state: StrategyState,
                      signals: Optional[Dict[str, Signal]] = None) -> float:
        """Engage or release the brake; returns the account's drawdown for the records.

        With the rotation sleeve on, the brake follows the other book's own value, marked from
        its last targets and the price changes since, as it would on a separate account.
        """
        c = self.cfg
        state.peak_equity = max(state.peak_equity, equity)
        drawdown = 1.0 - equity / state.peak_equity if state.peak_equity > 0 else 0.0
        brake_drawdown = drawdown
        if c.rotation_weight > 0 and signals is not None:
            change = sum(w * (signals[p].close / state.book_closes[p] - 1.0)
                         for p, w in state.book_weights.items()
                         if p in signals and state.book_closes.get(p))
            state.book_nav *= 1.0 + change
            state.book_peak = max(state.book_peak, state.book_nav)
            brake_drawdown = 1.0 - state.book_nav / state.book_peak
        if state.brake_on and brake_drawdown <= c.brake_release_drawdown:
            state.brake_on = False
        elif not state.brake_on and brake_drawdown >= c.brake_drawdown:
            state.brake_on = True
        return drawdown


def _hold_frozen(targets: Dict[str, float], account: Dict[str, float], frozen: AbstractSet[str],
                 reasons: Dict[str, str]) -> Dict[str, float]:
    """Halted pairs keep exactly their current weight, so no order is sent for them. If that
    leaves more invested than planned, the other long targets give way in proportion."""
    planned = sum(t for t in targets.values() if t > 0)
    out = dict(targets)
    for pair in frozen:
        current = account.get(pair, 0.0)
        if pair in out or current != 0.0:
            if current != 0.0 or out.get(pair, 0.0) != current:
                reasons[pair] = HOLD_HALTED
            out[pair] = current
    stuck = sum(max(out[p], 0.0) for p in frozen if p in out)
    free = sum(t for p, t in out.items() if t > 0 and p not in frozen)
    if free > 0 and stuck + free > planned + 1e-12:
        scale = max(planned - stuck, 0.0) / free
        out = {p: t * scale if t > 0 and p not in frozen else t for p, t in out.items()}
    return out


def _cap_weights(weights: Dict[str, float], cap: float) -> Dict[str, float]:
    """Cap each weight, handing the excess to the uncapped ones in proportion (sum unchanged)."""
    if cap >= 1.0 or cap * len(weights) < 1.0:
        return weights
    w = dict(weights)
    for _ in range(len(w)):
        over = [p for p, v in w.items() if v > cap + 1e-12]
        if not over:
            break
        excess = sum(w[p] - cap for p in over)
        for p in over:
            w[p] = cap
        free = {p: v for p, v in w.items() if v < cap - 1e-12}
        total = sum(free.values())
        if total <= 0:
            break
        for p, v in free.items():
            w[p] = v + excess * v / total
    return w
