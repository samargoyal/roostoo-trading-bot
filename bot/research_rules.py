"""Research options for the strategy, all off by default (research/rounds.py, docs/research.md).

With the live settings none of this changes anything: bot/strategy.py calls the _research_*()
methods where the research options act, and each returns its input unchanged unless its option
is switched on. Strategy inherits this class, so these methods see the same indicators, state
and configuration. Each one is the code the research rounds ran, moved here unchanged so that
bot/strategy.py reads as the live strategy.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING, AbstractSet, Dict, List, Optional

from bot.indicators import Signal
from bot.market_data import HOUR_MS
from bot.optimize import covariance
from bot.reasons import (CORE, EXIT_SHORT, EXIT_SHORT_STOP, EXIT_STOP, HOLD_HALTED, HOLD_NO_DATA, LS_SHORT,
                         ROTATION, SHORT_ENTRY, SHORT_HOLD)

if TYPE_CHECKING:  # type hints only: bot.strategy imports this module
    from bot.strategy import StrategyState

DAY_MS = 24 * HOUR_MS


class ResearchRules:
    """The research-only rules, mixed into Strategy."""

    # ---- hooks called from Strategy where each research option acts ----

    def _research_rotation_overrides(self, rotation, account, equity, state, frozen):
        """Rounds 23 and 37: trimming held picks only past a ratio; the sleeve's own brake."""
        c = self.cfg
        if c.rotation_trim_ratio > 0 and c.rotation_weight > 0:
            rotation = {p: (max(w, min(account.get(p, 0.0) / c.rotation_weight, w * c.rotation_trim_ratio))
                            if w > 0 and p != c.defensive_pair else w) for p, w in rotation.items()}
        if c.rotation_brake_drawdown > 0 or c.rotation_equity_ma_hours > 0:
            rotation = self._rotation_risk(rotation, equity, state, frozen)
        if c.rotation_dd_scale:
            # Round 68: shrink the sleeve in proportion to the account's fall from its recent high.
            state.equity_history = (state.equity_history + [equity])[-c.rotation_dd_hours:]
            dd = 1.0 - equity / max(state.equity_history)
            span = max(c.rotation_dd_full - c.rotation_dd_start, 1e-9)
            scale = 1.0 - (1.0 - c.rotation_dd_min) * min(max((dd - c.rotation_dd_start) / span, 0.0), 1.0)
            rotation = {p: w if p in frozen else w * scale for p, w in rotation.items()}
        return rotation

    def _research_regime(self, risk_on, regime, signals, state):
        """Rounds 24, 33 and 49: other definitions of the defensive book's risk-on regime."""
        c = self.cfg
        if c.regime_slow_filter and c.slow_filter and regime is not None:
            risk_on = risk_on and 0 < regime.slow_slow < regime.slow_fast
        if c.regime_band > 0 and regime is not None:
            risk_on = _banded(regime.close, regime.ema_regime, c.regime_band, state.regime_on)
            state.regime_on = int(risk_on)
        if c.regime_breadth > 0:
            risk_on = _breadth(signals, c.defensive_pair) >= c.regime_breadth
        return risk_on

    def _research_absorb(self, rotation, regime, frozen):
        """Rounds 61 and 62: while the rotation is out, its share runs the long-short book."""
        c = self.cfg
        share = None
        if (c.ls_absorb_rotation > 0 and c.book_mode == "long_short" and regime is not None
                and regime.ema_trend_fast <= regime.ema_trend_slow
                and (c.ls_absorb_sma_hours <= 0 or 0 < regime.close < regime.sma_long)):
            # The rotation is out of the market: its share (bar halted coins) joins the book.
            rotation = {p: w for p, w in rotation.items() if p in frozen}
            free = 1.0 - sum(abs(w) for w in rotation.values())
            share = 1.0 - c.rotation_weight + c.rotation_weight * c.ls_absorb_rotation * free
        return rotation, share

    def _research_trend_filter(self, trend_on, ts, signals, state, regime):
        """Rounds 24, 25, 33, 38 and 49: other trend filters for the rotation."""
        c = self.cfg
        if c.slow_filter and regime is not None:
            trend_on = trend_on and 0 < regime.slow_slow < regime.slow_fast
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
        if c.rotation_market_attention and self.external_scores:
            market = self._external(ts).get("ATT:MARKET")
            if market is not None and market < c.rotation_market_attention_floor:
                trend_on = False
        if c.rotation_reentry_hours > 0:
            if not trend_on:
                state.rotation_on_since = 0
            elif state.rotation_on_since == 0:
                state.rotation_on_since = ts
            trend_on = trend_on and ts - state.rotation_on_since >= c.rotation_reentry_hours * HOUR_MS
        return trend_on

    def _research_entry_due(self, due, trend_on, hour, last, state):
        """Round 35: re-plan soon after the filter turns on instead of at the next midnight."""
        c = self.cfg
        if (c.rotation_entry_every > 0 and trend_on and state.rotation_plan_on == 0
                and hour % c.rotation_entry_every == 0 and hour != last):
            due = True                          # the filter turned on: enter now, not at midnight
        return due

    def _research_ensemble(self, ts, signals, state, frozen, trend_on):
        """Rounds 31 and 44: sub-sleeves with their own rankings, or a lookback set by volatility."""
        c = self.cfg
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

    def _research_candidates(self, rising, signals, ts, regime):
        """Rounds 20-57: filters on the rotation's candidates and other rankings."""
        c = self.cfg
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
        if c.rotation_attention_filter and self.external_scores:
            table = self._external(ts)
            key = c.rotation_attention_key + ":"
            rising = {p: r for p, r in rising.items() if table.get(key + p, c.rotation_attention_floor)
                      >= c.rotation_attention_floor}
        if c.rotation_exhaustion:
            rising = {p: r for p, r in rising.items() if not self._exhausted(p, signals[p])}
        if c.rotation_attention_rank > 0 and self.external_scores and len(rising) > 1:
            table = self._external(ts)
            z_ret = _normal_scores(rising)
            att = {p: table["ATT:" + p] for p in rising if "ATT:" + p in table}
            z_att = _normal_scores(att) if len(att) > 1 else {}
            rising = {p: z_ret[p] + c.rotation_attention_rank * z_att.get(p, 0.0) for p in rising}
        if c.rotation_max_external > 0 and self.external_scores:
            table = self._external(ts)
            rising = {p: r for p, r in rising.items() if table.get(p, 0.0) <= c.rotation_max_external}
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
        return rising, alts

    def _research_picks(self, ranked, picks, rising, signals, state, ts, frozen, stuck):
        """Rounds 22, 27, 36 and 42: holding, buffering, concentrating or diversifying the picks."""
        c = self.cfg
        concentrated = False
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
        return ranked, picks, concentrated

    def _research_btc_fill(self, plan, picks, alts, regime, frozen, stuck):
        """Round 21: with rotation_vs_btc "btc", BTC fills the slots no coin beating BTC took."""
        c = self.cfg
        slots = max(c.rotation_top - stuck, 0)
        if (alts and c.rotation_vs_btc == "btc" and len(picks) < slots
                and regime.return_rotation > 0 and c.regime_pair not in frozen):
            # No coin beat BTC: BTC itself fills the empty slots.
            plan[c.regime_pair] = plan.get(c.regime_pair, 0.0) + (slots - len(picks)) / c.rotation_top

    def _research_rotation_shorts(self, plan, signals, state, frozen, ts):
        """Rounds 19 and 50: while BTC's filter is off, the sleeve's capital backs shorts."""
        c = self.cfg
        # Bear market: the sleeve's capital backs shorts on the weakest (or wildest) coins.
        pool = {p: s for p, s in signals.items()
                if p != c.defensive_pair and p not in frozen and p not in state.positions}
        if c.rotation_short_ranking == "trend_basket":
            # Every coin in its own downtrend, by inverse volatility: many small shorts,
            # so one squeeze costs little (research H50).
            crowded = (self._external(ts) if c.short_exclude_external and self.external_scores
                       else {})
            down = {p: 1.0 / s.volatility for p, s in pool.items()
                    if 0 < s.ema_trend_fast < s.ema_trend_slow and s.volatility > 0
                    and crowded.get(p, 0.0) >= 0}
            total = sum(down.values())
            for pair, v in down.items():
                plan[pair] = -min(v / total, c.rotation_short_cap)
            order = []
        elif c.rotation_short_ranking == "volatility":
            order = sorted(pool, key=lambda p: pool[p].volatility, reverse=True)
        else:
            order = sorted((p for p, s in pool.items() if s.return_rotation < 0),
                           key=lambda p: pool[p].return_rotation)
        for pair in order[:c.rotation_shorts]:
            plan[pair] = -1.0 / c.rotation_shorts

    def _research_cvar(self, plan):
        """A tail cap on the sleeve's 1-day CVaR."""
        c = self.cfg
        # Tail cap: shrink the whole sleeve, leaving the difference in cash.
        cvar = self._daily_cvar(plan)
        if cvar * c.rotation_weight > c.rotation_cvar_limit:
            scale = c.rotation_cvar_limit / (cvar * c.rotation_weight)
            plan = {p: w * scale for p, w in plan.items()}
        return plan

    def _research_sleeve_scale(self, plan, regime, ts, frozen):
        """Rounds 19, 28 and 41: scaling the sleeve by a model, by BTC's euphoria or by forecast volatility."""
        c = self.cfg
        if c.rotation_external_scale and self.external_scores:
            scale = self._external(ts).get("__scale__", 1.0)
            plan = {p: w if p in frozen else w * scale for p, w in plan.items()}
        if c.rotation_euphoria > 0 and regime is not None and regime.return_rotation > c.rotation_euphoria:
            plan = {p: w if p in frozen else w * 0.5 for p, w in plan.items()}
        if c.rotation_vol_forecast and plan:
            scale = self._vol_scale(c.rotation_vol_forecast)
            plan = {p: w * scale for p, w in plan.items()}
        if c.rotation_btc_dip_hours > 0 and plan and self._btc_dipping(ts):
            plan = {p: w if p in frozen or p == c.defensive_pair else w * c.rotation_btc_dip_share
                    for p, w in plan.items()}
        return plan

    def _research_ls_side(self, pair, s, side, state, ts, reasons, trend_on, sides, shorts_ok):
        """Rounds 50-64: variants of the long-short book's position in one coin; the side itself (or
        the side times a trend strength), or None for no position. Unchanged with the live settings."""
        c = self.cfg
        if (c.long_max_external > 0 and s.ls_fast > s.ls_slow and not c.ls_ensemble and self.external_scores
                and self._external(ts).get(pair, 0.0) > c.long_max_external):
            return None
        if c.ls_short_trend:
            short_down = 0 < s.ls_short_fast < s.ls_short_slow
            if (side > 0) == short_down:
                return None                    # the long and short trends disagree: flat
            side = -1.0 if short_down else 1.0
        if side > 0 and self._long_stopped(pair, s, state, ts, reasons):
            return None
        if c.ls_ensemble:
            if s.ls_vote == 0:
                return None
            side = s.ls_vote
        gap = s.ls_fast / s.ls_slow - 1.0
        if c.ls_regime_aligned and (side > 0) != trend_on:
            return None
        if (sides == "long" and side < 0) or (sides == "short" and side > 0) or (side < 0 and not shorts_ok):
            return None
        if abs(gap) < c.ls_band:
            return None
        strength = min(abs(gap) / c.ls_full_gap, 1.0) if c.ls_full_gap > 0 else 1.0
        return side * strength

    def _research_ls_gross(self, gross, signals, frozen, targets):
        """Round 51: with ls_full_gap, full trends everywhere would fill the book."""
        c = self.cfg
        if c.ls_full_gap > 0:
            # Strength sizing: full trends everywhere would fill the book; weak ones leave cash.
            gross = sum(1.0 / s.volatility for p, s in signals.items()
                        if p != c.defensive_pair and p not in frozen and p in targets and s.volatility > 0
                        and (c.ls_pairs != "btc" or p == c.regime_pair))
        return gross

    def _research_idle_to_gold(self, targets, reasons, signals, frozen, sides):
        """Round 60: the long-short book's unused share in PAXG while gold rises."""
        c = self.cfg
        if (c.ls_idle_horizon > 0 and sides != "short" and c.defensive_pair in targets
                and c.defensive_pair in signals and c.defensive_pair not in frozen):
            gold = self._horizon_return(c.defensive_pair, c.ls_idle_horizon)
            idle = 1.0 - sum(abs(t) for t in targets.values())
            if gold is not None and gold > 0 and idle > 1e-9:
                targets[c.defensive_pair] = idle
                reasons[c.defensive_pair] = CORE

    def _research_rotation_exits(self, ts, signals, state, frozen):
        """Round 66: a held pick leaves when its attention collapses or spikes (a blow-off), and
        cools down for stop_cooldown_hours."""
        c = self.cfg
        if c.rotation_take_profit > 0 or c.rotation_profit_trail_after > 0:
            self._research_secure_profits(ts, signals, state, frozen)
        if not (c.rotation_exit_attention or c.rotation_exit_spike > 0) or not self.external_scores:
            return
        table = self._external(ts)
        for pair, w in list(state.rotation_plan.items()):
            if w <= 0 or pair == c.defensive_pair or pair in frozen:
                continue
            att, spike = table.get("ATT:" + pair), table.get("SPK:" + pair)
            if ((c.rotation_exit_attention and att is not None and att < c.rotation_exit_attention_floor)
                    or (c.rotation_exit_spike > 0 and spike is not None and spike > c.rotation_exit_spike)):
                del state.rotation_plan[pair]
                state.rotation_cooldown[pair] = ts + c.stop_cooldown_hours * HOUR_MS

    def _research_secure_profits(self, ts, signals, state, frozen):
        """Round 68: track each pick's entry price; once it is up rotation_take_profit keep only
        part of it (until it leaves the picks), and once up rotation_profit_trail_after trail it."""
        c = self.cfg
        held = {p for p, w in state.rotation_plan.items() if w > 0 and p != c.defensive_pair}
        for book in (state.rotation_entry_close, state.rotation_trimmed):
            for pair in [p for p in book if p not in held]:
                del book[pair]
        for pair in sorted(held):
            s = signals.get(pair)
            if s is None or pair in frozen:
                continue
            entry = state.rotation_entry_close.setdefault(pair, s.close)
            if c.rotation_take_profit > 0 and pair not in state.rotation_trimmed and s.close >= entry * (1 + c.rotation_take_profit):
                state.rotation_plan[pair] *= c.rotation_take_profit_keep
                state.rotation_trimmed[pair] = c.rotation_take_profit_keep
            if c.rotation_profit_trail_after > 0:
                high = max(state.rotation_highs.get(pair, s.close), s.close)
                state.rotation_highs[pair] = high
                if high >= entry * (1 + c.rotation_profit_trail_after) and s.close <= high * (1 - c.rotation_profit_trail):
                    del state.rotation_plan[pair]
                    state.rotation_highs.pop(pair, None)
                    state.rotation_cooldown[pair] = ts + c.stop_cooldown_hours * HOUR_MS

    def _research_reapply_trims(self, plan, state):
        """Round 68: a pick whose profit was taken keeps its reduced share at the daily re-pick."""
        for pair, keep in state.rotation_trimmed.items():
            if pair in plan:
                plan[pair] *= keep

    def _research_short_veto(self, pair, ts):
        """Round 66: no short into rising retail attention."""
        c = self.cfg
        if c.short_attention_max >= 99 or not self.external_scores:
            return False
        att = self._external(ts).get("ATT:" + pair)
        return att is not None and att > c.short_attention_max

    def _research_window_lock(self, ts, equity, targets, book, rotation, signals, exposure, state, frozen, reasons):
        """H70 (research/h70_secure_profits.py): securing profits within the competition window,
        as secure_mode says; the targets and the exposure limit."""
        c = self.cfg
        was_locked = state.profit_locked
        if not self._window_lock_on(ts, equity, targets, state):
            return targets, exposure
        if c.secure_mode == "refresh":
            if not was_locked and c.rotation_weight > 0:
                targets = self._refresh_rotation(targets, rotation, state, frozen, reasons)
        elif c.secure_mode == "half":
            targets = {p: w * c.secure_scale for p, w in targets.items()}
            exposure *= c.secure_scale
        elif c.secure_mode == "half_gold":
            gross = sum(abs(w) for p, w in targets.items() if p not in frozen)
            targets = {p: w if p in frozen else w * c.secure_scale for p, w in targets.items()}
            if c.defensive_pair not in frozen:
                targets[c.defensive_pair] = targets.get(c.defensive_pair, 0.0) + (1.0 - c.secure_scale) * gross
                reasons.setdefault(c.defensive_pair, ROTATION)
        elif c.rotation_weight > 0:
            targets = self._locked_targets(targets, book, rotation, signals, frozen, reasons)
        return targets, exposure

    # ---- helpers of the research options ----

    def _btc_dipping(self, ts):
        """Round 70: BTC's return over rotation_btc_dip_hours to the last 00:00 UTC close is below
        zero, so the sleeve changes size only at the daily close (research/round70_btc_dip.py)."""
        c = self.cfg
        ind = self.indicators.get(c.regime_pair)
        back = (ts // HOUR_MS) % 24                 # bars since the one that closed at 00:00 UTC
        hours = c.rotation_btc_dip_hours
        if ind is None or len(ind.closes) < back + hours + 1:
            return False
        closes = list(ind.closes)
        return closes[-1 - back] < closes[-1 - back - hours]

    def _exhausted(self, pair, s):
        """Round 69 (research/h69_trend_exit.py): weak highs or volume divergence, the two warnings
        that predicted a lower next 3 days in all six folds."""
        c = self.cfg
        ind = self.indicators[pair]
        closes = list(ind.closes)
        dollar = list(ind.dollar)
        if len(closes) < 169 or len(dollar) < 144:
            return False
        weak_high = s.close >= c.exhaustion_near_high * max(closes[-168:]) and s.rsi < c.exhaustion_rsi
        rising = s.close > closes[-73]
        thin = sum(dollar[-72:]) < c.exhaustion_volume * sum(dollar[-144:-72])
        return weak_high or (rising and thin)

    def _locked_targets(self, targets, book, rotation, signals, frozen, reasons):
        """The capital stays invested while profits are secured: the rotation's coins move into
        the book, BTC, the defensive pair or the rotation's top secure_top coins (secure_mode)."""
        c = self.cfg
        coins = {p: w for p, w in rotation.items() if w > 0 and p != c.defensive_pair and p not in frozen}
        freed = c.rotation_weight * sum(coins.values())
        if freed <= 0:
            return targets                    # the rotation holds no coins (its filter is off)
        out = dict(targets)
        for pair, weight in coins.items():
            out[pair] -= c.rotation_weight * weight
        if c.secure_mode == "book":
            into = dict(book)
        elif c.secure_mode == "btc":
            into = {c.regime_pair: 1.0}
        elif c.secure_mode == "gold":
            into = {c.defensive_pair: 1.0}
        else:
            ranked = sorted((p for p, s in signals.items() if p != c.defensive_pair and p not in frozen
                             and s.return_rotation > 0), key=lambda p: signals[p].return_rotation, reverse=True)
            into = self._rotation_weights(ranked[:c.secure_top], signals)
        for pair, weight in into.items():
            out[pair] = out.get(pair, 0.0) + freed * weight
            reasons.setdefault(pair, ROTATION)
        return out

    def _refresh_rotation(self, targets, rotation, state, frozen, reasons):
        """Profits secured by rotating (secure_mode "refresh"): the rotation's coins are sold into
        the defensive pair and barred until the window ends, so its next daily pick puts the money
        into other coins."""
        c = self.cfg
        coins = {p: w for p, w in rotation.items() if w > 0 and p != c.defensive_pair and p not in frozen}
        if not coins:
            return targets
        end = c.window_start_ms + (state.window_index + 1) * c.window_days * DAY_MS
        out = dict(targets)
        for pair, weight in coins.items():
            out[pair] -= c.rotation_weight * weight
            state.rotation_plan.pop(pair, None)
            state.rotation_cooldown[pair] = end
        freed = sum(coins.values())
        state.rotation_plan[c.defensive_pair] = state.rotation_plan.get(c.defensive_pair, 0.0) + freed
        out[c.defensive_pair] = out.get(c.defensive_pair, 0.0) + c.rotation_weight * freed
        reasons.setdefault(c.defensive_pair, ROTATION)
        return out

    def _window_lock_on(self, ts, equity, targets, state):
        """Whether profits are secured: once the gain in the current window exceeds secure_k
        daily volatilities times the square root of the days left, until the window ends."""
        c = self.cfg
        if c.secure_k <= 0 or c.window_start_ms <= 0 or ts < c.window_start_ms or equity <= 0:
            return False
        length = c.window_days * DAY_MS
        index = (ts - c.window_start_ms) // length
        if index != state.window_index:
            state.window_index = index
            first = index == 0 and c.window_start_equity > 0
            state.window_equity = c.window_start_equity if first else equity
            state.profit_locked = False
        gain = equity / state.window_equity - 1.0 if state.window_equity > 0 else 0.0
        if not state.profit_locked and gain > 0:
            days_left = (c.window_start_ms + (index + 1) * length - ts) / DAY_MS
            sigma = self._portfolio_volatility(ts, targets)
            state.profit_locked = sigma > 0 and gain >= c.secure_k * sigma * math.sqrt(days_left)
        return state.profit_locked

    def _portfolio_volatility(self, ts, targets):
        """Daily volatility of the target portfolio from its positions' last secure_vol_hours of
        hourly log returns (a short counts against it), measured once a day; 0 without data."""
        day = ts // DAY_MS
        if getattr(self, "_volatility_day", None) == day:
            return self._volatility
        series = {p: list(self.indicators[p].returns)[-self.cfg.secure_vol_hours:]
                  for p, w in targets.items() if w != 0.0 and p in self.indicators}
        series = {p: r for p, r in series.items() if len(r) >= 48}
        sigma = 0.0
        if series:
            length = min(len(r) for r in series.values())
            portfolio = [sum(targets[p] * r[len(r) - length + i] for p, r in series.items()) for i in range(length)]
            mean = sum(portfolio) / length
            sigma = math.sqrt(sum((x - mean) ** 2 for x in portfolio) / (length - 1) * 24)
        self._volatility_day, self._volatility = day, sigma
        return sigma

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
        weights = self.cfg.rotation_horizon_weights or [1.0] * len(self.cfg.rotation_horizons)
        for h, w in zip(self.cfg.rotation_horizons, weights):
            values = {}
            for p in pairs:
                r = self.indicators[p].returns
                values[p] = sum(list(r)[-h:]) if len(r) >= h else 0.0
            for p, z in _normal_scores(values).items():
                total[p] += w * z
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

    def _overlay(self, ts: int, signals: Dict[str, Signal], weights: Dict[str, float], state: StrategyState,
                 targets: Dict[str, float], reasons: Dict[str, str], regime: Optional[Signal],
                 frozen: AbstractSet[str]) -> None:
        """book_mode "overlay": the cash the defensive book leaves idle shorts every coin in its
        own downtrend that the book does not hold, each its inverse-volatility share of all coins
        (so a few laggards in a rising market get small shorts, a broad decline large ones)."""
        idle = max(1.0 - sum(t for t in targets.values() if t > 0), 0.0)
        shorts, why = self._long_short_book(signals, weights, frozen, regime, "short", state, ts, slots=True)
        for pair, weight in shorts.items():
            if pair in frozen:
                continue
            if weight < 0 and targets.get(pair, 0.0) <= 0:
                targets[pair] = weight * idle
                reasons[pair] = why.get(pair, LS_SHORT)
            elif why.get(pair) in (EXIT_SHORT_STOP, EXIT_SHORT) and targets.get(pair, 0.0) == 0:
                reasons[pair] = why[pair]

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

    def _bear(self, regime: Optional[Signal]) -> bool:
        """A confirmed bear market: the rotation's BTC filter off and, with short_regime_hours,
        BTC below its long simple average (never before that average is ready)."""
        c = self.cfg
        if regime is None or regime.ema_trend_fast >= regime.ema_trend_slow:
            return False
        return c.short_regime_hours <= 0 or 0 < regime.close < regime.sma_long

    def _shorts_allowed(self, regime: Optional[Signal]) -> bool:
        return self.cfg.short_regime_hours <= 0 or self._bear(regime)

    def _long_stopped(self, pair: str, s: Signal, state: Optional[StrategyState], ts: int,
                      reasons: Dict[str, str]) -> bool:
        """ls_long_stop_atr: a held long below its highest close since entry less that many ATRs
        is sold (and cools down); a coin cooling down is not bought again yet."""
        c = self.cfg
        if c.ls_long_stop_atr <= 0 or state is None:
            return False
        if state.cooldown_until.get(pair, 0) > ts:
            return True
        info = state.positions.get(pair)
        if info is None:
            return False
        info.highest_close = max(info.highest_close, s.close)
        if s.close < info.highest_close - c.ls_long_stop_atr * s.atr:
            reasons[pair] = EXIT_STOP
            return True
        return False

    def _short_vol_scale(self) -> float:
        """base / recent volatility of the regime pair's hourly returns, at most 1."""
        recent_h, base_h = self.cfg.short_vol_ratio
        ind = self.indicators.get(self.cfg.regime_pair)
        r = list(ind.returns) if ind is not None else []
        if len(r) < max(recent_h, base_h):
            return 1.0

        def stdev(xs):
            m = sum(xs) / len(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

        recent, base = stdev(r[-recent_h:]), stdev(r[-base_h:])
        return min(1.0, base / recent) if recent > 0 else 1.0


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
