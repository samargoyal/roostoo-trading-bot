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
from bot.optimize import covariance, erc_weights, hrp_weights, mean_variance, min_variance, risk_budget_weights
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
        bear = (not getattr(self, "_combined_trend_on", True)) if c.regime_index else (
            regime is not None and regime.ema_trend_fast <= regime.ema_trend_slow)
        if (c.ls_absorb_rotation > 0 and c.book_mode == "long_short" and regime is not None and bear
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
        if c.regime_index:                                                # round 109
            alt_on = self._alt_index_on(signals, ts)
            trend_on = alt_on if c.regime_index == "alts" else (trend_on and alt_on)
            self._combined_trend_on = trend_on
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
        if c.rotation_wr_hours > 0 and c.rotation_wr_entry > -100:           # round 86
            rising = {p: r for p, r in rising.items()
                      if self._williams_r(p, signals[p].close, c.rotation_wr_hours) >= c.rotation_wr_entry}
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
        if c.long_max_funding_z > 0 and self.external_scores:            # E7
            table = self._external(ts)
            rising = {p: r for p, r in rising.items() if table.get("FZ:" + p, 0.0) < c.long_max_funding_z}
        if c.rotation_entropy_order > 0 and len(rising) > 2:              # E4
            pe = {p: self._entropy(p, c.rotation_entropy_order) for p in rising}
            mid = sorted(pe.values())[len(pe) // 2]
            rising = {p: r for p, r in rising.items() if (pe[p] < mid) == c.rotation_entropy_low}
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
        if c.rank_skip_days and len(rising) > 1:                         # C3c
            if c.rotation_ranking == "multi":
                rising = self._multi_horizon(list(rising))
            else:
                rising = {p: self._hours_return(p, c.rotation_lookback, skip_days=c.rank_skip_days) for p in rising}
        self._rank_scores = dict(rising)                                 # round 83: the "mv" alphas
        if c.session_tilt and len(rising) > 1:                            # C4
            n = c.session_tilt_days * 24
            feature = {p: self._hours_return(p, n, hours_of_day=range(13, 21)) - self._hours_return(p, n, hours_of_day=range(0, 8))
                       for p in rising}
            z_rank, z_feature = _normal_scores(rising), _normal_scores(feature)
            rising = {p: z_rank[p] + c.session_tilt * z_feature[p] for p in rising}
        return rising, alts

    def _research_picks(self, ranked, picks, rising, signals, state, ts, frozen, stuck):
        """Rounds 22, 27, 36 and 42: holding, buffering, concentrating or diversifying the picks."""
        c = self.cfg
        concentrated = False
        if c.trade_dependence and c.trade_dependence_scope != "book":     # E3, the rotation
            unfiltered = set(ranked[:c.rotation_top])
            for pair in [k[4:] for k in state.paper if k.startswith("rot:")]:
                if pair not in unfiltered and pair in signals:
                    side, price = state.paper.pop("rot:" + pair)
                    state.paper_last["rot:" + pair] = signals[pair].close / price - 1
            for pair in unfiltered:
                if "rot:" + pair not in state.paper and pair in signals:
                    state.paper["rot:" + pair] = [1.0, signals[pair].close]
            held = {p for p, w in state.rotation_plan.items() if w > 0}
            ranked = [p for p in ranked if p in held or self._take(state.paper_last.get("rot:" + p))]
            picks = ranked[:max(c.rotation_top - stuck, 0)]
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

    def _research_short_losers(self, signals, frozen, trend_on, ts):
        """H123: the sleeve short its weakest coins on the ranking, equally."""
        c = self.cfg
        if c.rotation_short_losers == "bear" and trend_on:
            return {}
        crowded = self._external(ts) if c.short_exclude_external and self.external_scores else {}
        pool = [p for p, s in signals.items()
                if p != c.defensive_pair and p not in frozen and s.return_rotation < 0
                and crowded.get(p, 0.0) >= 0
                and (not c.rotation_short_trend or 0 < s.ema_trend_fast < s.ema_trend_slow)]
        if not pool:
            return {}
        score = self._multi_horizon(pool) if c.rotation_ranking == "multi" and len(pool) > 1 else             {p: signals[p].return_rotation for p in pool}
        weak = sorted(pool, key=score.get)[:c.rotation_top]
        return {p: -1.0 / c.rotation_top for p in weak}

    def _research_short_leg(self, plan, picks, signals, state, frozen, ts):
        """Round 84: while the filter is on, part of the sleeve shorts the weakest coins."""
        c = self.cfg
        if c.rotation_short_share <= 0:
            return
        crowded = self._external(ts) if c.short_exclude_external and self.external_scores else {}
        pool = [p for p, s in signals.items()
                if p != c.defensive_pair and p not in frozen and p not in picks
                and s.return_rotation < 0 and crowded.get(p, 0.0) >= 0
                and (not c.rotation_short_trend or 0 < s.ema_trend_fast < s.ema_trend_slow)]
        if c.rotation_short_by == "multi" and len(pool) > 1 and c.rotation_horizons:
            score = self._multi_horizon(pool)
        else:
            score = {p: signals[p].return_rotation for p in pool}
        weak = sorted(pool, key=score.get)[:c.rotation_short_count]
        if not weak:
            return
        for pair in list(plan):
            if plan[pair] > 0:
                plan[pair] *= 1.0 - c.rotation_short_share
        for pair in weak:
            plan[pair] = -c.rotation_short_share / len(weak)

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
        if (c.long_max_funding_z > 0 and side > 0 and self.external_scores
                and self._external(ts).get("FZ:" + pair, 0.0) >= c.long_max_funding_z):
            return None                                                   # E7
        if c.trade_dependence and c.trade_dependence_scope != "rotation" and state is not None:  # E3
            key = "ls:" + pair
            open_trade = state.paper.get(key)
            if open_trade is None or open_trade[0] != side:
                if open_trade is not None:
                    state.paper_last[key] = open_trade[0] * (s.close / open_trade[1] - 1)
                state.paper[key] = [side, s.close]
                if open_trade is not None and not self._take(state.paper_last.get(key)):
                    state.paper_skip[key] = side
                else:
                    state.paper_skip.pop(key, None)
            if state.paper_skip.get(key) == side:
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
        if c.ls_xs_mode:                                                  # round 105
            z = self._ls_xs_z.get(pair)
            if z is None:
                return None
            crowded = c.short_exclude_external and self.external_scores and self._external(ts).get(pair, 0.0) < 0
            if c.ls_xs_mode == "weighted":
                if z < 0 and crowded:
                    return None
                side = (1.0 if z > 0 else -1.0) * abs(z)               # sized by |z| (the later steps keep it)
            else:
                held = self._ls_xs_held.get(pair)
                if held is None or (held < 0 and crowded):
                    return None
                side = float(held)
        if c.ls_xs_n > 0:                                                 # round 104
            if pair in self._ls_xs_long:
                side = 1.0
            elif pair in self._ls_xs_short:
                if ((c.ls_xs_short == "bear" and trend_on) or (c.ls_xs_short == "trend" and s.ls_fast >= s.ls_slow)
                        or (c.short_exclude_external and self.external_scores
                            and self._external(ts).get(pair, 0.0) < 0)):
                    return None
                side = -1.0
            else:
                return None
        if c.ls_neutral == "rank":                                        # round 103
            z = getattr(self, "_ls_rank_z", {})
            if pair in z:
                side = 1.0 if z[pair] >= self._ls_rank_median else -1.0
                if side < 0 and c.short_exclude_external and self.external_scores                         and self._external(ts).get(pair, 0.0) < 0:
                    return None                                           # crowded shorts, as for the book
        if c.ls_breadth_align > 0 and (side > 0) != (self._ls_breadth >= c.ls_breadth_align):
            return None                                                   # round 89
        if c.short_btc_crash > 0 and side < 0 and self._btc_crashed:
            return None
        if c.ls_corr_cut > 0 and c.ls_corr_mode == "shorts" and side < 0 and self._ls_corr_high:
            return None                                                   # round 90
        if c.ls_vol_power != 1.0:                                         # H112
            self.__dict__.setdefault("_ls_sigma", {})[pair] = s.volatility
        if (c.ls_pyramid_up != 1.0 or c.ls_pyramid_down != 1.0) and state is not None:   # H120
            info = state.positions.get(pair) if side > 0 else state.shorts.get(pair)
            entry = getattr(info, "entry_close", 0.0) if info is not None else 0.0
            mult = 1.0
            if entry > 0:
                profit = (s.close / entry - 1.0) * (1.0 if side > 0 else -1.0)
                move = self._hours_return(pair, c.ls_pyramid_hours) * (1.0 if side > 0 else -1.0)
                if move < 0:
                    mult = c.ls_pyramid_down
                elif profit > 0:
                    mult = c.ls_pyramid_up
            self.__dict__.setdefault("_ls_pyr", {})[pair] = mult
        if c.ls_gap_slope_hours > 0:                                      # H122: the gap's slope
            hist = self.__dict__.setdefault("_gap_hist", {})
            gap_now = s.ls_fast / s.ls_slow - 1.0
            past = hist.setdefault(pair, {})
            past[ts] = gap_now
            then = past.get(ts - c.ls_gap_slope_hours * HOUR_MS)
            for t_old in [t for t in past if t < ts - 48 * HOUR_MS]:
                del past[t_old]
            if then is not None:
                widening = (gap_now - then) * (1.0 if side > 0 else -1.0) > 0
                held = state is not None and (pair in state.positions if side > 0 else pair in state.shorts)
                if not widening and (c.ls_gap_slope_mode == "hold" or not held):
                    return None
        if c.ls_fill_gap:                                                 # H105: each coin's EMA gap
            self.__dict__.setdefault("_ls_gap", {})[pair] = abs(math.log(s.ls_fast / s.ls_slow))
        if c.ls_hysteresis > 0:                                           # round 94
            prev = self.__dict__.setdefault("_ls_prev_side", {})
            g = s.ls_fast / s.ls_slow - 1.0
            if pair in prev and prev[pair] != side and abs(g) < c.ls_hysteresis:
                side = prev[pair]
            prev[pair] = side
        r2_cache = self.__dict__.setdefault("_r2_cache", {})
        if c.ls_r2_hours > 0 and (pair, ts // (6 * HOUR_MS)) in r2_cache:   # round 96, every 6 hours
            self.__dict__.setdefault("_ls_er", {})[pair] = r2_cache[(pair, ts // (6 * HOUR_MS))]
        elif c.ls_r2_hours > 0:
            r = list(self.indicators[pair].returns)[-c.ls_r2_hours:]
            n = len(r)
            y, acc = [], 0.0
            for x in r:
                acc += x
                y.append(acc)
            if n > 2:
                tm, ym = (n - 1) / 2.0, sum(y) / n
                sty = sum((t - tm) * (v - ym) for t, v in enumerate(y))
                stt = sum((t - tm) ** 2 for t in range(n))
                syy = sum((v - ym) ** 2 for v in y)
                r2 = sty * sty / (stt * syy) if stt > 0 and syy > 0 else 0.0
            else:
                r2 = 0.0
            self.__dict__.setdefault("_ls_er", {})[pair] = r2
            if len(r2_cache) > 20000:
                r2_cache.clear()
            r2_cache[(pair, ts // (6 * HOUR_MS))] = r2
        if c.ls_regime_tilt > 0 and (side > 0) != trend_on:
            side *= c.ls_regime_tilt
        if c.ls_er_hours > 0:
            r = list(self.indicators[pair].returns)[-c.ls_er_hours:]
            path = sum(abs(x) for x in r)
            self.__dict__.setdefault("_ls_er", {})[pair] = abs(sum(r)) / path if path > 0 else 0.0
        if c.ls_rel_hours > 0 and pair != c.regime_pair:                  # round 93
            mine = list(self.indicators[pair].returns)[-c.ls_rel_hours:]
            btc = list(self.indicators[c.regime_pair].returns)[-c.ls_rel_hours:] if c.regime_pair in self.indicators else []
            if len(mine) == c.ls_rel_hours and len(btc) == c.ls_rel_hours and (sum(mine) > sum(btc)) != (side > 0):
                return None
        if c.ls_top_n > 0 or c.ls_fresh_days > 0:                        # round 92
            self.__dict__.setdefault("_ls_z", {})[pair] = abs(math.log(s.ls_fast / s.ls_slow)) / max(s.volatility, 1e-12)
            ages = self.__dict__.setdefault("_ls_age", {})
            if pair not in ages or ages[pair][0] != side:
                ages[pair] = (side, ts)
        if c.ls_min_variance_ratio > 0 and self._variance_ratio(pair, ts) < c.ls_min_variance_ratio:
            return None                                                   # round 87
        if c.short_max_jump_share > 0 and side < 0 and self._jump_share(pair, ts) > c.short_max_jump_share:
            return None
        if c.ls_sizing != "inverse_vol" and not hasattr(self, "_ls_mult"):
            self._ls_mult = {}
        if c.ls_sizing == "merton":
            # dS/S = mu dt + sigma dW: growth-optimal weight mu / sigma^2. An EMA of N hours lags a
            # steady trend by (N - 1) / 2 hours, so the gap between the two EMAs reads the drift.
            lag = (c.ls_trend[1] - c.ls_trend[0]) / 2.0
            self._ls_mult[pair] = abs(math.log(s.ls_fast / s.ls_slow)) / lag / max(s.volatility, 1e-12)
        elif c.ls_sizing == "har":                                        # round 88
            forecast = self._har_vol(pair, ts)
            self._ls_mult[pair] = s.volatility / forecast if forecast > 0 else 1.0
        elif c.ls_sizing == "kalman":
            t = s.trend_strength
            self._ls_mult[pair] = min(abs(t) / c.ls_kalman_t, 1.0) if (t > 0) == (side > 0) else 0.0
        strength = min(abs(gap) / c.ls_full_gap, 1.0) if c.ls_full_gap > 0 else 1.0
        return side * strength

    def _research_ls_gross(self, gross, signals, frozen, targets):
        """Round 51: with ls_full_gap, full trends everywhere would fill the book."""
        c = self.cfg
        if getattr(self, "_ls_fill", False):                              # round 93
            self._ls_fill = False
            return None
        if c.ls_corr_cut > 0 and c.ls_corr_mode == "book" and self._ls_corr_high:
            gross *= 2.0                                                  # round 90: the book halved
        if c.ls_vol_manage:                                               # round 87
            gross /= max(self._vol_scale(c.ls_vol_manage), 1e-6)
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
        if c.rotation_wr_hours > 0 and c.rotation_wr_exit > -100:            # round 86
            for pair, w in list(state.rotation_plan.items()):
                s = signals.get(pair)
                if w <= 0 or pair == c.defensive_pair or pair in frozen or s is None:
                    continue
                if self._williams_r(pair, s.close, c.rotation_wr_hours) < c.rotation_wr_exit:
                    del state.rotation_plan[pair]
                    state.rotation_cooldown[pair] = ts + c.stop_cooldown_hours * HOUR_MS
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

    def _research_ls_weights(self, raw, ts):
        """Round 74: the long-short book's positions weighted by a convex optimiser (ls_weighting)
        instead of inverse volatility, with the same gross; re-solved once a day or when the book's
        coins or sides change."""
        c = self.cfg
        if (c.ls_er_hours > 0 or c.ls_r2_hours > 0) and raw:              # rounds 94, 96
            er = getattr(self, "_ls_er", {})
            if c.ls_er_keep > 0:                                          # round 98
                ranked = sorted(raw, key=lambda p: er.get(p, 0.0), reverse=True)
                kept = ranked[:max(1, round(len(ranked) * c.ls_er_keep))]
                er = {p: (1.0 if p in kept else 0.0) for p in raw}
            elif c.ls_er_power != 1.0:
                er = {p: er.get(p, 0.0) ** c.ls_er_power for p in raw}
            mean = sum(er.get(p, 0.0) for p in raw) / len(raw)
            if mean > 0:
                before = sum(abs(v) for v in raw.values())
                raw = {p: v * min(er.get(p, 0.0) / mean, c.ls_sizing_cap) for p, v in raw.items()}
                after = sum(abs(v) for v in raw.values())
                raw = {p: v * before / after for p, v in raw.items()} if after > 0 else raw
        if c.ls_neutral and raw:                                          # round 103: dollar neutral
            gross = sum(abs(v) for v in raw.values())
            longs = sum(v for v in raw.values() if v > 0)
            shorts = -sum(v for v in raw.values() if v < 0)
            raw = {p: v * (gross / 2 / longs if v > 0 else gross / 2 / shorts)
                   for p, v in raw.items() if (longs if v > 0 else shorts) > 0}
        if c.ls_vol_power != 1.0 and raw:                                 # H112: another volatility power
            sig = getattr(self, "_ls_sigma", {})
            before = sum(abs(v) for v in raw.values())
            new = {p: v * sig.get(p, 1.0) ** (1.0 - c.ls_vol_power) for p, v in raw.items()}
            after = sum(abs(v) for v in new.values())
            raw = {p: v * before / after for p, v in new.items()} if after > 0 else raw
        if (c.ls_pyramid_up != 1.0 or c.ls_pyramid_down != 1.0) and raw:   # H120: pyramid winners
            pyr = getattr(self, "_ls_pyr", {})
            new = {p: v * pyr.get(p, 1.0) for p, v in raw.items()}
            room = getattr(self, "_ls_slots", 0.0)
            total = sum(abs(v) for v in new.values())
            raw = {p: v * room / total for p, v in new.items()} if room > 0 and total > room else new
        if c.ls_fill_gap and raw:                                         # H105: fill the idle share by gap
            gaps = getattr(self, "_ls_gap", {})
            left = getattr(self, "_ls_slots", 0.0) - sum(abs(v) for v in raw.values())
            total = sum(gaps.get(p, 0.0) for p in raw)
            if left > 0 and total > 0:
                raw = {p: v + (1.0 if v > 0 else -1.0) * left * gaps.get(p, 0.0) / total for p, v in raw.items()}
            self._ls_fill = True
        if c.ls_rel_fill and raw:                                         # round 93: fill the left-out share
            self._ls_fill = True
        if (c.ls_top_n > 0 or c.ls_fresh_days > 0) and raw:              # round 92
            gross = sum(abs(v) for v in raw.values())
            new = dict(raw)
            if c.ls_top_n > 0:
                z = getattr(self, "_ls_z", {})
                keep = sorted(new, key=lambda p: z.get(p, 0.0), reverse=True)[:c.ls_top_n]
                new = {p: new[p] for p in keep}
            if c.ls_fresh_days > 0:
                ages = getattr(self, "_ls_age", {})
                new = {p: v * (c.ls_fresh_boost if ts - ages.get(p, (0, -10 ** 15))[1] < c.ls_fresh_days * DAY_MS
                               else 1.0) for p, v in new.items()}
            total = sum(abs(v) for v in new.values())
            raw = {p: v * gross / total for p, v in new.items()} if total > 0 else raw
        if c.ls_sizing != "inverse_vol" and raw:                          # round 87
            mult, self._ls_mult = getattr(self, "_ls_mult", {}), {}
            new = {p: v * mult.get(p, 1.0) for p, v in raw.items()}
            total = sum(abs(v) for v in new.values())
            if total <= 0:
                return {}
            scale = sum(abs(v) for v in raw.values()) / total
            return {p: max(-c.ls_sizing_cap * abs(raw[p]), min(c.ls_sizing_cap * abs(raw[p]), v * scale))
                    for p, v in new.items() if v != 0}
        if c.ls_weighting == "inverse_vol" or len(raw) < 2:
            return raw
        key = (ts // DAY_MS, tuple(sorted((p, v > 0) for p, v in raw.items())))
        if getattr(self, "_ls_key", None) != key:
            pairs = sorted(raw)
            series = [list(self.indicators[p].returns)[-c.ls_cov_hours:] for p in pairs]
            length = min(len(r) for r in series)
            if length < 48:
                return raw
            signed = [[(1.0 if raw[p] > 0 else -1.0) * x for x in r[-length:]] for p, r in zip(pairs, series)]
            cov = covariance(signed, shrink=0.1)
            er = getattr(self, "_ls_er", {})
            if c.ls_weighting == "risk_budget_er":                       # round 100
                w = risk_budget_weights(cov, [max(er.get(p, 0.0), 1e-3) for p in pairs])
            elif c.ls_weighting == "mv_er":
                z = _normal_scores({p: er.get(p, 0.0) for p in pairs})
                w = mean_variance([z[p] for p in pairs], cov, c.ls_mv_risk_aversion, c.ls_max_weight)
            else:
                w = (erc_weights(cov) if c.ls_weighting == "erc" else hrp_weights(cov) if c.ls_weighting == "hrp"
                     else min_variance(cov, cap=c.ls_max_weight))
            self._ls_key, self._ls_w = key, dict(zip(pairs, w))
        gross = sum(abs(v) for v in raw.values())
        return {p: (1.0 if v > 0 else -1.0) * self._ls_w[p] * gross for p, v in raw.items()}

    def _research_risk_breakers(self, ts, equity, targets, rotation, account, signals, state, frozen):
        """Round 75 (RESEARCH_QUEUE.md part B): risk circuit breakers on the final targets."""
        c = self.cfg
        if not (c.day_loss_stop or c.dd_ladder or c.coin_loss_cap or c.squeeze_rise or c.btc_shock_1h
                or c.vol_regime or c.entry_hours or c.weekend_no_entries or c.weekend_scale != 1.0
                or c.beta_hedge or c.macro_rho or c.capitulation_size or c.volume_accel
                or c.rotation_meta_sizing) or equity <= 0:
            return targets
        out = dict(targets)
        until = state.risk_until
        rot = {p: c.rotation_weight * w for p, w in rotation.items() if p not in frozen}

        def hold_back(pairs=None, longs_only=False):
            for p, t in out.items():
                if p in frozen or (pairs is not None and p not in pairs):
                    continue
                w = account.get(p, 0.0)
                if t > 0:
                    out[p] = min(t, max(w, 0.0))
                elif t < 0 and not longs_only:
                    out[p] = max(t, min(w, 0.0))

        def scale(factor, only=None):
            for p in out:
                if p not in frozen:
                    part = out[p] if only is None else only.get(p, 0.0)
                    out[p] -= part * (1.0 - factor)

        if c.squeeze_rise > 0:                                   # B5
            for p, w in account.items():
                s, ind = signals.get(p), self.indicators.get(p)
                if w >= 0 or s is None or ind is None or len(ind.closes) < 25 or p in frozen:
                    continue
                closes = ind.closes
                if (s.close / closes[-25] - 1 >= c.squeeze_rise
                        or (s.atr > 0 and s.close - closes[-2] >= c.squeeze_atr * s.atr)):
                    out[p] = 0.0
                    state.short_cooldown_until[p] = ts + c.squeeze_block_hours * HOUR_MS
            shorts = -sum(t for p, t in out.items() if t < 0 and p not in frozen)
            if shorts > c.short_collateral_cap:
                for p, t in out.items():
                    if t < 0 and p not in frozen:
                        out[p] = t * c.short_collateral_cap / shorts
        if c.coin_loss_cap > 0:                                  # B3
            for p, w in account.items():
                ind = self.indicators.get(p)
                if w == 0 or p in frozen or ind is None or len(ind.closes) < 25:
                    continue
                if w * (ind.closes[-1] / ind.closes[-25] - 1) <= -c.coin_loss_cap:
                    state.coin_block_until[p] = ts + 24 * HOUR_MS
                    half = w / 2
                    out[p] = min(out.get(p, 0.0), half) if w > 0 else max(out.get(p, 0.0), half)
            blocked = {p for p, t in state.coin_block_until.items() if t > ts}
            if blocked:
                hold_back(blocked)
        if c.day_loss_stop > 0:                                  # B1
            day = ts // DAY_MS
            if day != state.risk_day:
                state.risk_day, state.risk_day_equity = day, equity
            loss = equity / state.risk_day_equity - 1 if state.risk_day_equity > 0 else 0.0
            if loss <= -c.day_loss_stop:
                until["day_stop"] = (day + 1) * DAY_MS
            if c.day_loss_cut > 0 and loss <= -c.day_loss_cut:
                until["day_cut"] = (day + 1) * DAY_MS
            if until.get("day_stop", 0) > ts:
                hold_back()
            if until.get("day_cut", 0) > ts:
                scale(0.5)
        if c.dd_ladder:                                          # B2
            dd = 1.0 - equity / state.peak_equity if state.peak_equity > 0 else 0.0
            reached = sum(1 for level in c.dd_ladder if dd >= level)
            if reached > state.dd_tier:
                state.dd_tier, state.dd_tier_since = reached, ts
                if reached >= 3:
                    until["dd_rotation"] = ts + 24 * HOUR_MS
            while (state.dd_tier > 0 and dd < c.dd_ladder[state.dd_tier - 1] / 2
                   and ts - state.dd_tier_since >= 12 * HOUR_MS):
                state.dd_tier -= 1
                state.dd_tier_since = ts
            if until.get("dd_rotation", 0) > ts:
                scale(0.0, rot)
            if state.dd_tier >= 1:
                scale(0.5 if state.dd_tier == 1 else 0.25)
        if c.btc_shock_1h > 0:                                   # B6
            closes = self.indicators[c.regime_pair].closes if c.regime_pair in self.indicators else []
            if len(closes) >= 5 and (closes[-1] / closes[-2] - 1 <= -c.btc_shock_1h
                                     or closes[-1] / closes[-5] - 1 <= -c.btc_shock_4h):
                until["btc_shock"] = ts + 6 * HOUR_MS
            if until.get("btc_shock", 0) > ts:
                hold_back(longs_only=True)
                scale(0.5, rot)
        if c.vol_regime > 0:                                     # B7
            ratio = self._vol_ratio()
            if ratio >= c.vol_regime:
                scale(max(c.vol_regime_floor, c.vol_regime / ratio))
        if c.beta_hedge > 0 and rot:                             # E1
            coins = {p: w for p, w in rot.items() if w > 0 and p != c.defensive_pair and p != c.regime_pair}
            hedge = c.beta_hedge * sum(w * self._beta(p) for p, w in coins.items())
            total = sum(coins.values())
            if hedge > 0 and total > 0:
                hedge = min(hedge, total / 2)
                for p, w in coins.items():
                    out[p] -= w * hedge / total
                out[c.regime_pair] = out.get(c.regime_pair, 0.0) - hedge
        if c.rotation_meta_sizing and self.external_scores:      # E6
            table = self._external(ts)
            for p, w in rot.items():
                prob = table.get("META:" + p)
                if w > 0 and prob is not None and p != c.defensive_pair:
                    out[p] -= w * (1.0 - min(max(2 * prob, 0.5), 1.0))
        if c.macro_rho > 0 and self.external_scores:             # E2
            table = self._external(ts)
            if table.get("MACRO:rho", 0.0) > c.macro_rho and table.get("MACRO:below", 0.0) > 0:
                scale(c.macro_scale)
        if c.volume_accel:                                       # E8
            hold_back({p for p in out if not self._volume_accelerating(p)}, longs_only=True)
        if c.capitulation_size > 0:                              # E5
            self._capitulation(ts, out, signals, state, frozen)
        weekend = (ts // DAY_MS + 3) % 7 >= 5                   # the epoch was a Thursday
        if c.entry_hours and (ts // HOUR_MS) % 24 not in c.entry_hours:   # C2, C6
            hold_back()
        if weekend and c.weekend_no_entries:                     # C3a
            hold_back()
        if weekend and c.weekend_scale != 1.0:                   # C3b
            scale(c.weekend_scale)
        return out

    def _take(self, last):
        """E3: whether a signal is taken after a paper trade that returned `last` (None: none yet)."""
        if last is None:
            return True
        return last < 0 if self.cfg.trade_dependence == "after_loss" else last > 0

    def _hours_return(self, pair, count, skip_days=(), hours_of_day=None):
        """The pair's log return over its last `count` hourly bars, counting only bars outside
        `skip_days` (0 Monday .. 6 Sunday) and, if given, inside the UTC hours `hours_of_day`."""
        ind = self.indicators[pair]
        r = list(ind.returns)[-count:]
        if ind.last_ts is None:
            return 0.0
        wanted = set(hours_of_day) if hours_of_day is not None else None
        total = 0.0
        for i, x in enumerate(r):
            ts = ind.last_ts - (len(r) - 1 - i) * HOUR_MS                  # the bar's open time
            if skip_days and (ts // DAY_MS + 3) % 7 in skip_days:
                continue
            if wanted is not None and (ts // HOUR_MS) % 24 not in wanted:
                continue
            total += x
        return total

    def _beta(self, pair):
        """E1: the pair's beta to the regime pair over rotation_cov_hours of hourly returns."""
        c = self.cfg
        a = list(self.indicators[pair].returns)[-c.rotation_cov_hours:]
        b = list(self.indicators[c.regime_pair].returns)[-c.rotation_cov_hours:]
        n = min(len(a), len(b))
        if n < 48:
            return 1.0
        a, b = a[-n:], b[-n:]
        ma, mb = sum(a) / n, sum(b) / n
        var = sum((y - mb) ** 2 for y in b)
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / var if var > 0 else 1.0

    def _entropy(self, pair, order):
        """E4: permutation entropy (0 to 1) of the last 168 hourly closes' ordinal patterns."""
        closes = list(self.indicators[pair].closes)[-168:]
        counts = {}
        for i in range(len(closes) - order + 1):
            window = closes[i:i + order]
            key = tuple(sorted(range(order), key=window.__getitem__))
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return 1.0
        h = -sum(n / total * math.log(n / total) for n in counts.values())
        return h / math.log(math.factorial(order))

    def _volume_accelerating(self, pair):
        """E8: the second difference of the 24-hour average of dollar volume is positive."""
        n = self.cfg.volume_accel_hours
        d = list(self.indicators[pair].dollar)
        if len(d) < n + 2:
            return True
        m0, m1, m2 = (sum(d[len(d) - k - n:len(d) - k]) / n for k in (0, 1, 2))
        return m0 - 2 * m1 + m2 > 0

    def _capitulation(self, ts, out, signals, state, frozen):
        """E5: a small long after an hour of forced selling that closed off its low."""
        c = self.cfg
        regime = signals.get(c.regime_pair)
        on = regime is not None and regime.ema_trend_fast > regime.ema_trend_slow
        for pair, (entry_ts, price, atr) in list(state.capitulation.items()):
            s = signals.get(pair)
            if s is None or s.close >= price + 1.5 * atr or ts - entry_ts >= 24 * HOUR_MS:
                del state.capitulation[pair]
        if len(state.capitulation) < 3 and (c.capitulation_regime == "any" or (c.capitulation_regime == "on") == on):
            for pair, s in signals.items():
                if len(state.capitulation) >= 3:
                    break
                if pair in state.capitulation or pair in frozen or pair == c.defensive_pair:
                    continue
                ind = self.indicators[pair]
                r, d = list(ind.returns), list(ind.dollar)
                if len(r) < 169 or len(d) < 673 or not ind.highs:
                    continue
                past = r[-169:-1]
                mean = sum(past) / len(past)
                sd = (sum((x - mean) ** 2 for x in past) / (len(past) - 1)) ** 0.5
                week = sorted(d[-1 - 168 * k] for k in (1, 2, 3, 4))
                rvol = d[-1] / ((week[1] + week[2]) / 2) if week[1] + week[2] > 0 else 0.0
                rng = ind.highs[-1] - ind.lows[-1]
                wick = (s.close - ind.lows[-1]) / rng if rng > 0 else 0.0
                if sd > 0 and r[-1] / sd <= -3 and rvol >= 3 and wick >= 0.4:
                    state.capitulation[pair] = [float(ts), s.close, s.atr]
        for pair in state.capitulation:
            out[pair] = out.get(pair, 0.0) + c.capitulation_size

    def _vol_ratio(self):
        """B7: BTC's realised variance over the last 24 hours over its median over 30 days."""
        r = list(self.indicators[self.cfg.regime_pair].returns)[-(720 + 24):]
        if len(r) < 48:
            return 0.0
        sq = [x * x for x in r]
        sums, acc = [], sum(sq[:24])
        sums.append(acc)
        for i in range(24, len(sq)):
            acc += sq[i] - sq[i - 24]
            sums.append(acc)
        ordered = sorted(sums)
        median = ordered[len(ordered) // 2]
        return sums[-1] / median if median > 0 else 0.0

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

    def _intrabar_short_stop(self, pair, s, info, ref) -> bool:
        """H121: the lowest of the short's stop levels; True (and its fill price noted for the
        backtester) when the hour's high reached it."""
        c = self.cfg
        levels = []
        if c.short_stop_atr > 0:
            levels.append(ref + c.short_stop_atr * s.atr)
        if c.short_stop_pct > 0:
            levels.append(ref * (1.0 + c.short_stop_pct))
        if c.stop_cap_entry_atr >= 0 and info.entry_close > 0:
            levels.append(info.entry_close + c.stop_cap_entry_atr * s.atr)
        ind = self.indicators[pair]
        if not levels or not ind.highs or len(ind.closes) < 2:
            return False
        level = min(levels)
        if ind.highs[-1] < level:
            return False
        self.stop_fills[pair] = max(level, ind.closes[-2])           # a gap past the level fills at the open
        return True

    def _intrabar_long_stop(self, pair, s, info, ref) -> bool:
        """H121: as _intrabar_short_stop for a long, on the hour's low."""
        c = self.cfg
        levels = []
        if c.ls_long_stop_atr > 0:
            levels.append(ref - c.ls_long_stop_atr * s.atr)
        if c.ls_long_stop_pct > 0:
            levels.append(ref * (1.0 - c.ls_long_stop_pct))
        if c.stop_cap_entry_atr >= 0 and info.entry_close > 0:
            levels.append(info.entry_close - c.stop_cap_entry_atr * s.atr)
        ind = self.indicators[pair]
        if not levels or not ind.lows or len(ind.closes) < 2:
            return False
        level = max(levels)
        if ind.lows[-1] > level:
            return False
        self.stop_fills[pair] = min(level, ind.closes[-2])
        return True

    def _alt_index_on(self, signals, ts):
        """Round 109: an equal-weight index of the coins other than BTC and the defensive pair (the
        mean of their hourly log returns, summed), True while its 168h EMA is above its 672h EMA.
        Built from the indicators' return history on the first call, then updated hourly."""
        c = self.cfg
        pairs = [p for p in signals if p not in (c.regime_pair, c.defensive_pair) and p in self.indicators]
        a_f, a_s = 2.0 / 169, 2.0 / 673
        if getattr(self, "_alt_ts", None) is None:
            hist = [list(self.indicators[p].returns) for p in pairs]
            n = max((len(h) for h in hist), default=0)
            level = f = s = 0.0
            for k in range(n, 0, -1):
                rs = [h[-k] for h in hist if len(h) >= k]
                level += sum(rs) / len(rs) if rs else 0.0
                f, s = (level, level) if k == n else (f + a_f * (level - f), s + a_s * (level - s))
            self._alt_level, self._alt_f, self._alt_s, self._alt_ts = level, f, s, ts
        elif ts > self._alt_ts:
            rs = [self.indicators[p].returns[-1] for p in pairs if self.indicators[p].returns]
            self._alt_level += sum(rs) / len(rs) if rs else 0.0
            self._alt_f += a_f * (self._alt_level - self._alt_f)
            self._alt_s += a_s * (self._alt_level - self._alt_s)
            self._alt_ts = ts
        return self._alt_f > self._alt_s

    def _research_split(self, signals):
        """Round 106: the rotation's share of equity by a fast regime read each hour."""
        c = self.cfg
        if not c.split_regime:
            return
        btc = signals.get(c.regime_pair)
        votes = []
        if c.split_regime in ("btc", "both") and btc is not None:
            if btc.close > btc.ema_fast > btc.ema_slow:
                votes.append(1)
            elif btc.close < btc.ema_fast < btc.ema_slow:
                votes.append(-1)
            else:
                votes.append(0)
        if c.split_regime in ("breadth", "both"):
            ups = [sum(list(self.indicators[p].returns)[-72:]) > 0 for p in signals
                   if p != c.defensive_pair and len(self.indicators[p].returns) >= 72]
            share = sum(ups) / len(ups) if ups else 0.5
            votes.append(1 if share > 0.6 else -1 if share < 0.4 else 0)
        state = votes[0] if votes and all(v == votes[0] for v in votes) else 0
        bull, neutral, bear = c.split_shares
        c.rotation_weight = bull if state > 0 else bear if state < 0 else neutral

    def _research_ls_market(self, signals):
        """Round 89: the share of the book's coins in uptrends, and whether BTC fell more than
        short_btc_crash over the last 30 days."""
        c = self.cfg
        if c.ls_breadth_align > 0:
            trends = [s.ls_fast > s.ls_slow for p, s in signals.items()
                      if p != c.defensive_pair and s.ls_fast > 0 and s.ls_slow > 0]
            self._ls_breadth = sum(trends) / len(trends) if trends else 0.5
        if c.ls_corr_cut > 0:
            self._ls_corr_high = self._mean_correlation(signals) > c.ls_corr_cut
        if c.ls_xs_mode:                                                  # round 105
            pool = [p for p, s in signals.items() if p != c.defensive_pair and s.volatility > 0
                    and len(self.indicators[p].returns) >= max(c.rotation_horizons or [c.rotation_lookback])]
            day = max((s.ts for s in signals.values()), default=0) // DAY_MS
            if not (c.ls_xs_daily and getattr(self, "_ls_xs_day", None) == day):
                self._ls_xs_day = day
                score = self._multi_horizon(pool) if len(pool) > 2 else {}
                if score:
                    m = sum(score.values()) / len(score)
                    sd = math.sqrt(sum((v - m) ** 2 for v in score.values()) / (len(score) - 1)) or 1.0
                    self._ls_xs_z = {p: (v - m) / sd for p, v in score.items()}
                else:
                    self._ls_xs_z = {}
                held, new = getattr(self, "_ls_xs_held", {}), {}
                for p, z in self._ls_xs_z.items():
                    s = signals[p]
                    long_ok = c.ls_xs_mode != "dual" or s.return_rotation > 0
                    short_ok = c.ls_xs_mode != "dual" or s.ls_fast < s.ls_slow
                    if long_ok and (z > c.ls_xs_z or (held.get(p) == 1 and z > c.ls_xs_exit)):
                        new[p] = 1
                    elif short_ok and (z < -c.ls_xs_z or (held.get(p) == -1 and z < -c.ls_xs_exit)):
                        new[p] = -1
                self._ls_xs_held = new
        if c.ls_xs_n > 0:                                                 # round 104
            pool = [p for p, s in signals.items() if p != c.defensive_pair and s.volatility > 0
                    and len(self.indicators[p].returns) >= max(c.rotation_horizons or [c.rotation_lookback])]
            score = self._multi_horizon(pool) if len(pool) > 1 else {}
            ranked = sorted(score, key=score.get, reverse=True)
            n = min(c.ls_xs_n, len(ranked) // 2)
            self._ls_xs_long, self._ls_xs_short = set(ranked[:n]), set(ranked[len(ranked) - n:])
        if c.ls_neutral == "rank":                                        # round 103
            z = {p: math.log(s.ls_fast / s.ls_slow) / s.volatility for p, s in signals.items()
                 if p != c.defensive_pair and s.ls_fast > 0 and s.ls_slow > 0 and s.volatility > 0}
            self._ls_rank_z = z
            vals = sorted(z.values())
            self._ls_rank_median = vals[len(vals) // 2] if vals else 0.0
        if c.short_btc_crash > 0:
            r = self._horizon_return(c.regime_pair, 720) if c.regime_pair in self.indicators else None
            self._btc_crashed = r is not None and r < -c.short_btc_crash

    def _mean_correlation(self, signals, top: int = 10, hours: int = 72) -> float:
        """Round 90: the mean pairwise correlation of hourly returns over `hours` among the `top`
        most traded coins (by 30-day dollar volume); 0 with too little history."""
        c = self.cfg
        pairs = sorted((p for p in signals if p != c.defensive_pair and p in self.indicators),
                       key=lambda p: self.indicators[p].dollar_sum, reverse=True)[:top]
        series = [list(self.indicators[p].returns)[-hours:] for p in pairs]
        series = [r for r in series if len(r) == hours]
        if len(series) < 3:
            return 0.0
        z = []
        for r in series:
            m = sum(r) / hours
            sd = math.sqrt(sum((x - m) ** 2 for x in r) / (hours - 1))
            if sd <= 0:
                continue
            z.append([(x - m) / sd for x in r])
        total, n = 0.0, 0
        for i in range(len(z)):
            for j in range(i + 1, len(z)):
                total += sum(a * b for a, b in zip(z[i], z[j])) / (hours - 1)
                n += 1
        return total / n if n else 0.0

    def _har_vol(self, pair: str, ts: int) -> float:
        """The coin's hourly volatility forecast from a HAR model (Corsi, 2009) with equal weights:
        the mean of the last day's, the last week's mean and the last month's mean daily realised
        variance, per hour; 0 while there is under a month of history. Once a day (round 88)."""
        cache = self.__dict__.setdefault("_har_cache", {})
        key = (pair, ts // DAY_MS)
        if key not in cache:
            if len(cache) > 5000:
                cache.clear()
            r = list(self.indicators[pair].returns)[-720:]
            if len(r) < 720:
                cache[key] = 0.0
            else:
                days = [sum(x * x for x in r[i:i + 24]) for i in range(0, 720, 24)]
                cache[key] = math.sqrt((days[-1] + sum(days[-7:]) / 7 + sum(days) / 30) / 3 / 24)
        return cache[key]

    def _variance_ratio(self, pair: str, ts: int, q: int = 24, hours: int = 720) -> float:
        """Lo-MacKinlay variance ratio: the variance of overlapping q-hour returns over q times
        that of hourly returns; above 1 for trending (positively autocorrelated) prices, 1 for a
        random walk. 1 while there is too little history; computed once a day (round 87)."""
        cache = self.__dict__.setdefault("_vr_cache", {})
        key = (pair, ts // DAY_MS)
        if key not in cache:
            if len(cache) > 5000:
                cache.clear()
            cache[key] = self._variance_ratio_now(pair, q, hours)
        return cache[key]

    def _variance_ratio_now(self, pair: str, q: int, hours: int) -> float:
        r = list(self.indicators[pair].returns)[-hours:]
        if len(r) < hours // 2:
            return 1.0
        n = len(r)
        mean = sum(r) / n
        var1 = sum((x - mean) ** 2 for x in r) / (n - 1)
        if var1 <= 0:
            return 1.0
        sums, acc = [], sum(r[:q])
        sums.append(acc)
        for i in range(q, n):
            acc += r[i] - r[i - q]
            sums.append(acc)
        varq = sum((x - q * mean) ** 2 for x in sums) / max(len(sums) - 1, 1)
        return varq / (q * var1)

    def _jump_share(self, pair: str, ts: int, hours: int = 168) -> float:
        """The share of realised variance from jumps (Barndorff-Nielsen and Shephard): 1 - bipower
        variation (pi/2 x sum |r_t||r_t-1|) over realised variance, floored at 0; computed every 6
        hours (round 87)."""
        cache = self.__dict__.setdefault("_jump_cache", {})
        key = (pair, ts // (6 * HOUR_MS))
        if key in cache:
            return cache[key]
        if len(cache) > 5000:
            cache.clear()
        r = list(self.indicators[pair].returns)[-hours:]
        if len(r) < 24:
            return 0.0
        rv = sum(x * x for x in r)
        bv = math.pi / 2 * sum(abs(a) * abs(b) for a, b in zip(r[1:], r[:-1]))
        cache[key] = max(0.0, 1.0 - bv / rv) if rv > 0 else 0.0
        return cache[key]

    def _williams_r(self, pair: str, close: float, hours: int) -> float:
        """Williams %R over the last `hours` bars, the latest included: 0 at the range's top,
        -100 at its bottom; 0 (never filtered) while there is too little history."""
        ind = self.indicators[pair]
        if len(ind.highs) < hours:
            return 0.0
        hh, ll = max(list(ind.highs)[-hours:]), min(list(ind.lows)[-hours:])
        return -100.0 * (hh - close) / (hh - ll) if hh > ll else 0.0

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
                if self.cfg.rank_skip_days:                               # C3c
                    values[p] = self._hours_return(p, h, skip_days=self.cfg.rank_skip_days)
                else:
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
        if (c.ls_long_stop_atr <= 0 and c.ls_long_stop_pct <= 0) or state is None:
            return False
        if state.cooldown_until.get(pair, 0) > ts:
            return True
        info = state.positions.get(pair)
        if info is None:
            return False
        info.highest_close = max(info.highest_close, s.close)
        ref = info.entry_close if c.stop_from_entry and info.entry_close > 0 else info.highest_close
        if c.intrabar_stops and self._intrabar_long_stop(pair, s, info, ref):
            reasons[pair] = EXIT_STOP
            return True
        if ((c.ls_long_stop_atr > 0 and s.close < ref - c.ls_long_stop_atr * s.atr)
                or (c.ls_long_stop_pct > 0 and s.close < ref * (1.0 - c.ls_long_stop_pct))
                or (c.stop_cap_entry_atr >= 0 and info.entry_close > 0
                    and s.close < info.entry_close - c.stop_cap_entry_atr * s.atr)):
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
