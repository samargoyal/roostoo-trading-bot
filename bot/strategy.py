"""The trading strategy: long-only trend following on hourly bars.

The same Strategy object drives the backtest and the live bot. It turns the latest
closed bars and the current portfolio into target weights; it never talks to the
exchange. Every number below is a field of StrategyConfig.

  Regime   BTC above its 200-hour EMA: up to 75% invested and 8 positions.
           Otherwise up to 25% and 3 positions, with PAXG first in line.
  Trend    a coin is eligible while EMA50 > EMA200 and its close is above EMA200.
  Ranking  the lowest volatility of hourly returns over the last 168 hours first.
           Free slots go to the best-ranked eligible coins. (Optionally: 0.5 x 72h
           + 0.5 x 168h volatility-adjusted momentum, the earlier default.)
  Entry    only into eligible coins with RSI(14) <= 70, not cooling down after a stop.
  Sizing   equal risk contribution from the last 336 hours' covariance (each position
           adds the same share of the book's variance), scaled to the exposure limit and
           capped at 15%. (Optionally 1 / (ATR / price), the earlier default.)
  Exits    EMA50 falls below EMA200, or the close drops 8 ATR below the highest
           close since entry (then no re-entry into that coin for 24 hours).
           A held coin is never sold just for ranking lower; only when risk-off
           cuts the number of positions are the lowest-ranked ones closed.
  Brake    once equity is 4% below its peak, trend positions are halved until the
           drawdown is back under 2%.
  Core     5% of equity stays in PAXG at all times, so there is always a position
           to rebalance on quiet days (see the activity rule in planner.py).
  Rotation 40% of equity (rotation_weight) rotates daily into the 2 coins with the
           strongest positive 336h return, while BTC's 168h EMA is above its 672h EMA, and
           leaves at once when it is not. Everything above runs on the other 60%. The two
           books' returns are barely correlated (0.18), so together they beat holding BTC
           in every test period with a smaller drawdown than BTC (research H15-H16).
  Shorts   optional (short_exposure, off by default): short the 3 most volatile coins
           whatever their trend, sized by inverse ATR, with a trailing stop 10 ATR above
           the lowest close since entry. Low-volatility coins have tended to beat
           high-volatility ones, so this sleeve earns the same effect from the other side
           and hedges the long book (research H13).
  Halts    a pair Roostoo will not trade (decide()'s `frozen`, when plan_around_halts is on)
           is held exactly as it is and never entered, and the book's other coins are
           re-solved around it: equal risk contributions with its weight fixed, counting its
           covariance with them. Either way the planner sends no orders in it.

The baseline in the project brief used EMA 20/100, 24h/72h momentum and a 2.5 ATR
stop, and also re-ranked held coins every hour. Backtests on two separate years
showed heavy churn (positions held for a median of 6 hours) and losses after fees,
so the slower settings above replaced it. Research in research/ then found that
momentum did not predict which coin would do better next, while low volatility did
(the low-volatility effect), so low volatility became the ranking. See the README.
"""
import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import AbstractSet, Dict, List, Optional

from bot.config import StrategyConfig
from bot.indicators import IndicatorSet, Signal
from bot.optimize import covariance, erc_weights, erc_weights_fixed, min_variance
from bot.market_data import HOUR_MS, Bar

# Reasons attached to each target, recorded with every decision and order.
ENTRY = "entry"
HOLD = "hold"
CORE = "core"
EXIT_TREND = "exit_trend"
EXIT_STOP = "exit_stop"
EXIT_REGIME = "exit_regime"
HOLD_NO_DATA = "hold_no_data"
HOLD_HALTED = "hold_halted"   # Roostoo is not trading the pair: hold it as it is
SHORT_ENTRY = "short_entry"
SHORT_HOLD = "short_hold"
EXIT_SHORT_STOP = "exit_short_stop"
EXIT_SHORT = "exit_short"
ROTATION = "rotation"


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


class Strategy:
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
                               cfg.rotation_trend_fast, cfg.rotation_trend_slow)
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
        if c.rotation_trim_ratio > 0 and c.rotation_weight > 0:
            rotation = {p: (max(w, min(account.get(p, 0.0) / c.rotation_weight, w * c.rotation_trim_ratio))
                            if w > 0 and p != c.defensive_pair else w) for p, w in rotation.items()}
        if c.rotation_brake_drawdown > 0 or c.rotation_equity_ma_hours > 0:
            rotation = self._rotation_risk(rotation, equity, state, frozen)
        weights = self._defensive_weights(weights, rotation)

        regime = signals.get(c.regime_pair)
        risk_on = regime is not None and regime.close > regime.ema_regime
        if c.regime_band > 0 and regime is not None:
            risk_on = _banded(regime.close, regime.ema_regime, c.regime_band, state.regime_on)
            state.regime_on = int(risk_on)
        if c.regime_breadth > 0:
            risk_on = _breadth(signals, c.defensive_pair) >= c.regime_breadth
        exposure = c.risk_on_exposure if risk_on else c.risk_off_exposure
        max_positions = c.max_positions_risk_on if risk_on else c.max_positions_risk_off

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

        if c.short_exposure > 0 or state.shorts:
            self._shorts(ts, signals, scores, weights, state, targets, reasons,
                         set(keep + entries) | set(rotation), frozen)

        if c.rotation_weight > 0:
            state.book_weights = {p: t for p, t in targets.items() if t != 0.0}
            state.book_closes = {p: signals[p].close for p in state.book_weights if p in signals}
            share = 1.0 - c.rotation_weight
            targets = {p: share * t for p, t in targets.items()}
            for pair, weight in rotation.items():
                if targets.get(pair, 0.0) == 0.0:
                    reasons[pair] = ROTATION
                targets[pair] = targets.get(pair, 0.0) + c.rotation_weight * weight

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
        if c.rotation_regime_pair:
            other = signals.get(c.rotation_regime_pair)
            trend_on = other is not None and other.ema_trend_fast > other.ema_trend_slow
        if c.rotation_filter_band > 0 and regime is not None:
            trend_on = _banded(regime.ema_trend_fast, regime.ema_trend_slow, c.rotation_filter_band,
                               state.rotation_filter_on)
            state.rotation_filter_on = int(trend_on)
        if c.rotation_breadth > 0:
            wide = _breadth(signals, c.defensive_pair) >= c.rotation_breadth
            trend_on = (trend_on and wide) if c.rotation_breadth_mode == "and" else wide
        if c.rotation_exit_sma > 0 and trend_on:
            trend_on = self._above_sma(c.regime_pair, c.rotation_exit_sma)
        if c.rotation_reentry_hours > 0:
            if not trend_on:
                state.rotation_on_since = 0
            elif state.rotation_on_since == 0:
                state.rotation_on_since = ts
            trend_on = trend_on and ts - state.rotation_on_since >= c.rotation_reentry_hours * HOUR_MS
        hour = ts // HOUR_MS
        last = state.rotation_plan_ts // HOUR_MS
        due = (state.rotation_plan_ts == 0 or hour - last >= c.rotation_rebalance_hours
               or (hour % c.rotation_rebalance_hours == 0 and hour != last))
        if (c.rotation_entry_every > 0 and trend_on and state.rotation_plan_on == 0
                and hour % c.rotation_entry_every == 0 and hour != last):
            due = True                          # the filter turned on: enter now, not at midnight
        if due:
            state.rotation_plan_on = int(trend_on)
        if due and (c.rotation_ensemble or c.rotation_adaptive_lookbacks):
            specs = c.rotation_ensemble
            if c.rotation_adaptive_lookbacks:
                high_vol, normal = c.rotation_adaptive_lookbacks
                specs = ["ret:%d" % (high_vol if self._vol_scale("realised") < 1.0 else normal)]
            plan = self._ensemble_plan(signals, frozen, specs) if trend_on else {}
            defensive = signals.get(c.defensive_pair)
            if (not trend_on and defensive is not None and defensive.return_rotation > 0
                    and c.defensive_pair not in frozen):
                plan = {c.defensive_pair: 1.0}
            state.rotation_plan = plan
            state.rotation_plan_ts = ts
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
                alts = None
                if c.rotation_vs_btc and regime is not None:
                    floor = regime.return_rotation + c.rotation_btc_margin
                    rising = {p: r for p, r in rising.items() if p != c.regime_pair and r > floor}
                    alts = True
                if c.rotation_donchian_entry > 0:
                    rising = {p: r for p, r in rising.items()
                              if signals[p].close >= self._channel(p, c.rotation_donchian_entry, high=True)}
                if c.rotation_min_age_hours > 0:
                    rising = {p: r for p, r in rising.items()
                              if self.indicators[p].bars_seen >= c.rotation_min_age_hours}
                if c.rotation_top_volume > 0:
                    liquid = sorted(signals, key=lambda p: self.indicators[p].dollar_sum, reverse=True)
                    allowed = set(liquid[:c.rotation_top_volume])
                    rising = {p: r for p, r in rising.items() if p in allowed}
                if c.rotation_min_tstat > 0:
                    rising = {p: r for p, r in rising.items()
                              if signals[p].volatility > 0 and math.log1p(signals[p].return_rotation)
                              / (signals[p].volatility * math.sqrt(c.rotation_lookback)) >= c.rotation_min_tstat}
                if c.rotation_pick_trend:
                    def own_trend(s: Signal) -> bool:
                        above = s.close > s.ema_slow
                        rising_ema = s.ema_fast > s.ema_slow
                        return {"close": above, "ema": rising_ema}.get(c.rotation_pick_trend, above and rising_ema)
                    rising = {p: r for p, r in rising.items() if own_trend(signals[p])}
                if c.rotation_exclude_external and self.external_scores:
                    table = self._external(ts)
                    rising = {p: r for p, r in rising.items() if table.get(p, 1.0) >= 0}
                if c.rotation_max_z > 0:
                    rising = {p: r for p, r in rising.items() if self._zscore(p) <= c.rotation_max_z}
                if c.rotation_ranking == "residual":
                    rising = {p: self._residual_return(p, r) for p, r in rising.items()}
                elif c.rotation_ranking == "kalman":
                    rising = {p: signals[p].trend_strength for p in rising}
                elif c.rotation_ranking == "multi" and len(rising) > 1:
                    rising = self._multi_horizon(list(rising))
                elif c.rotation_ranking == "external" and self._external(ts):
                    table = self._external(ts)
                    rising = {p: table.get(p, -1e9) for p in rising}
                ranked = sorted(rising, key=rising.get, reverse=True)
                if c.rotation_donchian_hold:
                    held = [p for p, w in state.rotation_plan.items()
                            if w > 0 and p != c.defensive_pair and p in signals and p not in frozen]
                    ranked = held + [p for p in ranked if p not in held]
                picks = ranked[:max(c.rotation_top - stuck, 0)]
                if c.rotation_corr_lambda > 0 and len(ranked) >= 3 and c.rotation_top == 2 and stuck == 0:
                    first, pool = ranked[0], ranked[1:5]
                    series = self._recent_returns([first] + pool, c.rotation_lookback)
                    if series is not None:
                        cov = covariance(series, shrink=0.0)
                        z = _normal_scores({p: rising[p] for p in pool})
                        corr = {p: cov[0][i + 1] / math.sqrt(cov[0][0] * cov[i + 1][i + 1])
                                for i, p in enumerate(pool) if cov[0][0] > 0 and cov[i + 1][i + 1] > 0}
                        second = max(corr, key=lambda p: z[p] - c.rotation_corr_lambda * corr[p])
                        ranked = [first, second] + [p for p in ranked if p not in (first, second)]
                        picks = ranked[:2]
                if (c.rotation_concentrate > 0 and len(ranked) >= 2 and stuck == 0
                        and signals[ranked[1]].return_rotation > 0
                        and signals[ranked[0]].return_rotation >= c.rotation_concentrate * signals[ranked[1]].return_rotation):
                    picks = ranked[:1]
                    concentrated = True
                else:
                    concentrated = False
                if c.rotation_buffer > c.rotation_top or c.rotation_min_hold_hours > 0:
                    held = [p for p, w in state.rotation_plan.items()
                            if w > 0 and p != c.defensive_pair and p in rising]
                    keep = [p for p in held
                            if (c.rotation_buffer > c.rotation_top and p in ranked[:c.rotation_buffer])
                            or (c.rotation_min_hold_hours > 0
                                and ts - state.rotation_entry.get(p, ts) < c.rotation_min_hold_hours * HOUR_MS)]
                    keep = sorted(keep, key=ranked.index)[:max(c.rotation_top - stuck, 0)]
                    picks = keep + [p for p in ranked if p not in keep][:max(c.rotation_top - stuck - len(keep), 0)]
                state.rotation_entry = {p: state.rotation_entry.get(p, ts) if p in state.rotation_plan else ts
                                        for p in picks}
                # The picks fill len(picks) of rotation_top slots; how they share it is the
                # weighting. Unfilled slots go to PAXG or cash below, whatever the weighting.
                filled = (1.0 - core) * (1.0 if concentrated else len(picks) / c.rotation_top)
                for pair, weight in self._rotation_weights(picks, signals).items():
                    plan[pair] = filled * weight
                slots = max(c.rotation_top - stuck, 0)
                if (alts and c.rotation_vs_btc == "btc" and len(picks) < slots
                        and regime.return_rotation > 0 and c.regime_pair not in frozen):
                    # No coin beat BTC: BTC itself fills the empty slots.
                    plan[c.regime_pair] = plan.get(c.regime_pair, 0.0) + (slots - len(picks)) / c.rotation_top
            elif c.rotation_shorts > 0:
                # Bear market: the sleeve's capital backs shorts on the weakest (or wildest) coins.
                pool = {p: s for p, s in signals.items()
                        if p != c.defensive_pair and p not in frozen and p not in state.positions}
                if c.rotation_short_ranking == "volatility":
                    order = sorted(pool, key=lambda p: pool[p].volatility, reverse=True)
                else:
                    order = sorted((p for p, s in pool.items() if s.return_rotation < 0),
                                   key=lambda p: pool[p].return_rotation)
                for pair in order[:c.rotation_shorts]:
                    plan[pair] = -1.0 / c.rotation_shorts
            empty = 1.0 - sum(abs(w) for w in plan.values())
            defensive = signals.get(c.defensive_pair)
            if (empty > 1e-9 and defensive is not None and defensive.return_rotation > 0
                    and c.defensive_pair not in frozen):
                plan[c.defensive_pair] = plan.get(c.defensive_pair, 0.0) + empty
            if c.rotation_cvar_limit > 0 and plan:
                # Tail cap: shrink the whole sleeve, leaving the difference in cash.
                cvar = self._daily_cvar(plan)
                if cvar * c.rotation_weight > c.rotation_cvar_limit:
                    scale = c.rotation_cvar_limit / (cvar * c.rotation_weight)
                    plan = {p: w * scale for p, w in plan.items()}
            state.rotation_plan = plan
            state.rotation_plan_ts = ts
        if c.rotation_stop_atr > 0 or c.rotation_stop_pct > 0 or c.rotation_donchian_exit > 0:
            self._rotation_stops(ts, signals, state, frozen)
        plan = dict(state.rotation_plan)
        if not trend_on:
            # Leave at once when the trend filter fails; only the defensive part (and any
            # halted coin, which cannot be sold) stays, and any bear-market shorts.
            plan = {p: w for p, w in plan.items() if p == c.defensive_pair or p in frozen or w < 0}
        else:
            plan = {p: w for p, w in plan.items() if w > 0 or p in frozen}   # cover shorts at once
        if c.rotation_external_scale and self.external_scores:
            scale = self._external(ts).get("__scale__", 1.0)
            plan = {p: w if p in frozen else w * scale for p, w in plan.items()}
        if c.rotation_euphoria > 0 and regime is not None and regime.return_rotation > c.rotation_euphoria:
            plan = {p: w if p in frozen else w * 0.5 for p, w in plan.items()}
        if c.rotation_vol_forecast and plan:
            scale = self._vol_scale(c.rotation_vol_forecast)
            plan = {p: w * scale for p, w in plan.items()}
        return plan

    def _channel(self, pair: str, hours: int, high: bool) -> float:
        """Highest high (or lowest low) of the `hours` bars before the latest one; a value that
        never triggers while there is too little history."""
        ind = self.indicators[pair]
        series = ind.highs if high else ind.lows
        if len(series) < hours + 1:
            return float("inf") if high else 0.0
        window = list(series)[-hours - 1:-1]
        return max(window) if high else min(window)

    def _horizon_return(self, pair: str, hours: int) -> Optional[float]:
        r = self.indicators[pair].returns
        if len(r) < hours:
            return None
        return math.exp(sum(list(r)[-hours:])) - 1.0

    def _ensemble_plan(self, signals: Dict[str, Signal], frozen: AbstractSet[str],
                       specs: Optional[List[str]] = None) -> Dict[str, float]:
        """Equal sub-sleeves, each holding its own top rotation_top coins; an empty slot in a
        sub-sleeve goes to the defensive pair if its 336h return is positive, else cash."""
        c = self.cfg
        specs = specs or c.rotation_ensemble
        share = 1.0 / len(specs)
        pool = [p for p in signals if p != c.defensive_pair and p not in frozen]
        defensive = signals.get(c.defensive_pair)
        gold_up = defensive is not None and defensive.return_rotation > 0 and c.defensive_pair not in frozen
        plan: Dict[str, float] = {}
        for spec in specs:
            kind, _, arg = spec.partition(":")
            horizons = [int(h) for h in arg.split(",")]
            if kind == "ret":
                rets = {p: self._horizon_return(p, horizons[0]) for p in pool}
                score = {p: r for p, r in rets.items() if r is not None and r > 0}
            else:
                eligible = [p for p in pool if signals[p].return_rotation > 0]
                score = {}
                if eligible:
                    total = {p: 0.0 for p in eligible}
                    for h in horizons:
                        vals = {p: (self._horizon_return(p, h) or 0.0) for p in eligible}
                        for p, z in _normal_scores(vals).items():
                            total[p] += z
                    score = total
            picks = sorted(score, key=score.get, reverse=True)[:c.rotation_top]
            for p in picks:
                plan[p] = plan.get(p, 0.0) + share / c.rotation_top
            empty = share * (c.rotation_top - len(picks)) / c.rotation_top
            if empty > 1e-12 and gold_up:
                plan[c.defensive_pair] = plan.get(c.defensive_pair, 0.0) + empty
        return plan

    def _rotation_risk(self, plan: Dict[str, float], equity: float, state: StrategyState,
                       frozen: AbstractSet[str]) -> Dict[str, float]:
        """Account-level controls on the sleeve: a brake on the account's drawdown, and a filter
        on the account's own equity curve (out while below its moving average)."""
        c = self.cfg
        if c.rotation_equity_ma_hours > 0:
            state.equity_history = (state.equity_history + [equity])[-c.rotation_equity_ma_hours:]
            if (len(state.equity_history) >= c.rotation_equity_ma_hours
                    and equity < sum(state.equity_history) / len(state.equity_history)):
                plan = {p: w for p, w in plan.items() if p == c.defensive_pair or p in frozen}
        if c.rotation_brake_drawdown > 0 and state.peak_equity > 0:
            drawdown = 1.0 - equity / state.peak_equity
            if state.rotation_brake_on and drawdown <= c.rotation_brake_release:
                state.rotation_brake_on = False
            elif not state.rotation_brake_on and drawdown >= c.rotation_brake_drawdown:
                state.rotation_brake_on = True
            if state.rotation_brake_on:
                plan = {p: w if p in frozen else w * 0.5 for p, w in plan.items()}
        return plan

    def _above_sma(self, pair: str, hours: int) -> bool:
        """True while the pair's close is at or above its simple average over `hours` closes
        (closes rebuilt from the hourly log returns)."""
        ind = self.indicators.get(pair)
        if ind is None or len(ind.returns) < hours:
            return True
        r = list(ind.returns)[-(hours - 1):] if hours > 1 else []
        level, total = 0.0, 1.0                 # closes relative to the latest one
        for x in reversed(r):
            level -= x
            total += math.exp(level)
        return total / hours <= 1.0

    def _multi_horizon(self, pairs: List[str]) -> Dict[str, float]:
        """Sum of the normal scores of each coin's rank by return over each horizon in
        rotation_horizons (from its hourly log returns)."""
        total = {p: 0.0 for p in pairs}
        for h in self.cfg.rotation_horizons:
            values = {}
            for p in pairs:
                r = self.indicators[p].returns
                values[p] = sum(list(r)[-h:]) if len(r) >= h else 0.0
            for p, z in _normal_scores(values).items():
                total[p] += z
        return total

    def _rotation_stops(self, ts: int, signals: Dict[str, Signal], state: StrategyState,
                        frozen: AbstractSet[str]) -> None:
        """Trailing stops on the rotation's picks: a pick that closes too far below its highest
        close since it was picked leaves the plan (its slot stays in cash until the next
        rebalance) and cannot be picked again for stop_cooldown_hours."""
        c = self.cfg
        for pair in [p for p in state.rotation_highs if state.rotation_plan.get(p, 0.0) <= 0]:
            del state.rotation_highs[pair]
        for pair, weight in list(state.rotation_plan.items()):
            s = signals.get(pair)
            if weight <= 0 or pair == c.defensive_pair or s is None or pair in frozen:
                continue
            high = max(state.rotation_highs.get(pair, s.close), s.close)
            state.rotation_highs[pair] = high
            hit = ((c.rotation_stop_atr > 0 and s.close < high - c.rotation_stop_atr * s.atr)
                   or (c.rotation_stop_pct > 0 and s.close < high * (1.0 - c.rotation_stop_pct))
                   or (c.rotation_donchian_exit > 0
                       and s.close < self._channel(pair, c.rotation_donchian_exit, high=False)))
            if hit:
                del state.rotation_plan[pair]
                del state.rotation_highs[pair]
                state.rotation_cooldown[pair] = ts + c.stop_cooldown_hours * HOUR_MS
        for pair in [p for p, until in state.rotation_cooldown.items() if until <= ts]:
            del state.rotation_cooldown[pair]

    def _vol_scale(self, method: str) -> float:
        """min(1, typical / forecast) for BTC's daily volatility: the forecast from the last 30
        days of hourly returns (HAR: the mean of the last day's, week's and month's realised
        variance; EWMA: RiskMetrics, lambda 0.94, on daily realised variance), the typical
        level the median of the same forecast at each of the last 60 day-ends."""
        ind = self.indicators.get(self.cfg.regime_pair)
        if ind is None or len(ind.returns) < 24 * 90:
            return 1.0
        r = list(ind.returns)
        days = [sum(x * x for x in r[len(r) - 24 * (i + 1):len(r) - 24 * i]) for i in range(90)][::-1]

        def forecast(k: int) -> float:          # using day blocks up to index k (inclusive)
            if method == "realised":
                return sum(days[k - 29:k + 1]) / 30
            if method == "ewma":
                var = days[k - 29]
                for d in days[k - 28:k + 1]:
                    var = 0.94 * var + 0.06 * d
                return var
            return (days[k] + sum(days[k - 6:k + 1]) / 7 + sum(days[k - 29:k + 1]) / 30) / 3

        now = forecast(89)
        history = sorted(forecast(k) for k in range(30, 90))
        typical = history[len(history) // 2]
        return min(1.0, math.sqrt(typical / now)) if now > 0 else 1.0

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

    def _residual_return(self, pair: str, raw: float) -> float:
        """The lookback return left after removing the part explained by the coin's beta to the
        regime pair: log(1 + r) - beta * log(1 + r_btc), with beta from the same hourly window."""
        c = self.cfg
        if pair == c.regime_pair or c.regime_pair not in self.indicators:
            return math.log1p(raw)
        series = self._recent_returns([pair, c.regime_pair], c.rotation_lookback)
        if series is None:
            return math.log1p(raw)
        coin, btc = series
        n = len(btc)
        mean_c, mean_b = sum(coin) / n, sum(btc) / n
        var_b = sum((x - mean_b) ** 2 for x in btc)
        beta = sum((x - mean_c) * (y - mean_b) for x, y in zip(coin, btc)) / var_b if var_b > 0 else 1.0
        return sum(coin) - beta * sum(btc)

    def _daily_cvar(self, weights: Dict[str, float], level: float = 0.95) -> float:
        """Historical 1-day CVaR of the given holdings: the average loss over the worst
        (1 - level) of overlapping 24-hour windows in the last rotation_cov_hours."""
        pairs = list(weights)
        series = self._recent_returns(pairs, self.cfg.rotation_cov_hours)
        if series is None:
            return 0.0
        hourly = [sum(weights[p] * series[i][t] for i, p in enumerate(pairs)) for t in range(len(series[0]))]
        daily = sorted(sum(hourly[t:t + 24]) for t in range(len(hourly) - 23))
        tail = daily[:max(1, int(len(daily) * (1 - level)))]
        return max(0.0, -sum(tail) / len(tail))

    def _zscore(self, pair: str) -> float:
        """How many standard deviations the latest close is above its mean over rotation_z_hours."""
        closes = list(self.indicators[pair].closes)[-self.cfg.rotation_z_hours:]
        n = len(closes)
        if n < 24:
            return 0.0
        mean = sum(closes) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in closes) / (n - 1))
        return (closes[-1] - mean) / sd if sd > 0 else 0.0

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

    def _shorts(self, ts: int, signals: Dict[str, Signal], scores: Dict[str, float],
                weights: Dict[str, float], state: StrategyState, targets: Dict[str, float],
                reasons: Dict[str, str], longs: set,
                frozen: AbstractSet[str] = frozenset()) -> None:
        """Negative targets for the short sleeve: the most volatile coins, held until their stop."""
        c = self.cfg
        stopped = set()
        for pair, info in state.shorts.items():
            s = signals.get(pair)
            if s is None:
                continue
            info.lowest_close = min(info.lowest_close, s.close)
            if pair in frozen:
                continue
            if s.close > info.lowest_close + c.short_stop_atr_multiple * s.atr:
                stopped.add(pair)
                reasons[pair] = EXIT_SHORT_STOP
        stuck = {p: max(-weights.get(p, 0.0), 0.0) for p in state.shorts if p in frozen}
        held = [p for p in state.shorts if p in signals and p not in stopped and p not in longs
                and p not in frozen]
        candidates = [p for p, s in signals.items()
                      if p not in state.shorts and p not in longs and p != c.defensive_pair
                      and p not in frozen
                      and s.rsi >= c.short_rsi_min and state.short_cooldown_until.get(p, 0) <= ts]
        order = sorted(held + candidates, key=lambda p: scores[p])  # lowest score: most volatile
        rank = {p: i for i, p in enumerate(order)}
        limit = max(c.max_shorts - len(stuck), 0) if c.short_exposure > 0 else 0
        keep = sorted(held, key=rank.get)[:limit]
        for pair in order:
            if len(keep) >= limit:
                break
            if pair not in keep:
                keep.append(pair)
        for pair in state.shorts:
            if pair not in targets:
                continue                                    # left the universe: the planner covers it
            if pair in stuck:
                targets[pair] = -stuck[pair]                # halted: hold it as it is
                reasons[pair] = HOLD_HALTED
            elif pair not in signals and pair not in longs:
                targets[pair] = min(weights.get(pair, 0.0), 0.0)   # no data: leave it as it is
                reasons[pair] = HOLD_NO_DATA
            elif pair not in keep and pair not in stopped:
                reasons[pair] = EXIT_SHORT
        for pair, weight in self._size(keep, signals, c.short_exposure, stuck).items():
            if state.brake_on:
                weight *= c.brake_factor
            targets[pair] = -weight
            reasons[pair] = SHORT_HOLD if pair in state.shorts else SHORT_ENTRY

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


def _banded(value: float, level: float, band: float, was_on: int) -> bool:
    """value > level, with hysteresis: once on, off only below level x (1 - band); once off, on
    only above level x (1 + band)."""
    if was_on == 1:
        return value >= level * (1.0 - band)
    if was_on == 0:
        return value > level * (1.0 + band)
    return value > level


def _breadth(signals: Dict[str, Signal], defensive_pair: str) -> float:
    """Share of the universe's coins (not the defensive pair) whose fast EMA is above the slow."""
    coins = [s for p, s in signals.items() if p != defensive_pair]
    return sum(1 for s in coins if s.ema_fast > s.ema_slow) / len(coins) if coins else 0.0


def _normal_scores(values: Dict[str, float]) -> Dict[str, float]:
    """Each value's rank turned into a standard normal score (ties share the average rank)."""
    n = len(values)
    order = sorted(values, key=values.get)
    ranks: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    dist = NormalDist()
    return {p: dist.inv_cdf((r - 0.5) / n) for p, r in ranks.items()}


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
