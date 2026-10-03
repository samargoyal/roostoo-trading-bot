import unittest

from bot.config import StrategyConfig
from bot.indicators import Signal
from bot.market_data import HOUR_MS
from bot.strategy import (CORE, ENTRY, EXIT_REGIME, EXIT_STOP, EXIT_TREND, HOLD, PositionInfo,
                          Strategy, StrategyState)

UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "PAXG/USD"]


def sig(close=110.0, fast=105.0, slow=100.0, regime=90.0, atr=1.0, rsi=50.0,
        r_short=0.02, r_long=0.05, vol=0.01):
    """A signal that passes the trend filter by default."""
    return Signal(0, close, fast, slow, regime, atr, rsi, r_short, r_long, vol)


def make_strategy(signals, **overrides):
    """The strategy on its own: the rotation sleeve is off unless a test turns it on."""
    base = {"rotation_weight": 0.0, "max_positions_risk_on": 4, "sizing": "inverse_atr"}
    cfg = StrategyConfig(universe=list(UNIVERSE), **dict(base, **overrides))
    strategy = Strategy(cfg)
    strategy.signals = lambda: signals
    return strategy


class RegimeAndSizingTest(unittest.TestCase):
    def test_risk_on_caps_each_position_and_adds_core(self):
        signals = {p: sig() for p in UNIVERSE}
        d = make_strategy(signals).decide(0, 100000.0, {}, StrategyState())
        self.assertTrue(d.risk_on)
        self.assertEqual(d.exposure_limit, 0.75)
        entered = [p for p, r in d.reasons.items() if r == ENTRY]
        self.assertEqual(len(entered), 4)
        for pair in entered:
            # (0.75 - 0.05) / 4 = 0.175 would exceed the 15% cap
            expected = 0.15 - (0.05 if pair == "PAXG/USD" else 0.0)
            self.assertAlmostEqual(d.targets[pair] - (0.05 if pair == "PAXG/USD" else 0.0), expected)
        self.assertAlmostEqual(d.targets.get("PAXG/USD", 0.0) >= 0.05, True)

    def test_weights_are_inverse_to_atr_percent(self):
        signals = {"BTC/USD": sig(atr=1.0), "ETH/USD": sig(atr=2.0)}
        strategy = make_strategy(signals, max_weight=1.0, core_weight=0.0)
        d = strategy.decide(0, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["BTC/USD"], 2 * d.targets["ETH/USD"])
        self.assertAlmostEqual(d.targets["BTC/USD"] + d.targets["ETH/USD"], 0.75)

    def test_risk_off_puts_paxg_first_and_shrinks_exposure(self):
        signals = {p: sig(r_short=0.10 if p == "SOL/USD" else 0.0) for p in UNIVERSE}
        signals["BTC/USD"] = sig(close=110.0, regime=120.0)  # below its regime EMA
        d = make_strategy(signals, max_positions_risk_off=2, ranking="momentum").decide(0, 1.0, {}, StrategyState())
        self.assertFalse(d.risk_on)
        self.assertEqual(d.reasons["PAXG/USD"], ENTRY)
        self.assertEqual(d.reasons["SOL/USD"], ENTRY)
        self.assertEqual(sum(1 for r in d.reasons.values() if r == ENTRY), 2)
        self.assertLessEqual(sum(d.targets.values()), 0.25 + 1e-9)

    def test_risk_off_cuts_the_lowest_ranked_held_positions(self):
        signals = {p: sig(r_short=s) for p, s in
                   zip(UNIVERSE, [0.01, 0.05, 0.04, 0.03, 0.0])}
        signals["BTC/USD"] = sig(regime=200.0, r_short=0.01)
        state = StrategyState(positions={p: PositionInfo(0, 110.0)
                                         for p in ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]})
        d = make_strategy(signals, max_positions_risk_off=2, ranking="momentum").decide(0, 1.0, {}, state)
        self.assertEqual(d.reasons["ETH/USD"], HOLD)
        self.assertEqual(d.reasons["SOL/USD"], HOLD)
        self.assertEqual(d.reasons["DOGE/USD"], EXIT_REGIME)
        self.assertEqual(d.reasons["BTC/USD"], EXIT_REGIME)

    def test_core_is_held_even_with_no_signals(self):
        d = make_strategy({}).decide(0, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["PAXG/USD"], 0.05)
        self.assertEqual(d.reasons["PAXG/USD"], CORE)


class RankingTest(unittest.TestCase):
    def test_low_volatility_ranks_first_by_default(self):
        signals = {"BTC/USD": sig(vol=0.010), "ETH/USD": sig(vol=0.005, r_short=-0.05),
                   "SOL/USD": sig(vol=0.030, r_short=0.30)}
        strategy = make_strategy(signals, max_positions_risk_on=1, core_weight=0.0)
        d = strategy.decide(0, 1.0, {}, StrategyState())
        self.assertEqual([p for p, r in d.reasons.items() if r == ENTRY], ["ETH/USD"])

    def test_momentum_ranking_is_still_available(self):
        signals = {"BTC/USD": sig(vol=0.010), "ETH/USD": sig(vol=0.005, r_short=-0.05),
                   "SOL/USD": sig(vol=0.030, r_short=0.30)}
        strategy = make_strategy(signals, max_positions_risk_on=1, core_weight=0.0, ranking="momentum")
        d = strategy.decide(0, 1.0, {}, StrategyState())
        self.assertEqual([p for p, r in d.reasons.items() if r == ENTRY], ["SOL/USD"])


class EntryAndExitTest(unittest.TestCase):
    def test_overbought_coin_is_not_entered_but_is_kept_if_held(self):
        signals = {"BTC/USD": sig(rsi=80.0), "ETH/USD": sig(rsi=80.0)}
        state = StrategyState(positions={"ETH/USD": PositionInfo(0, 110.0)})
        d = make_strategy(signals).decide(0, 1.0, {}, state)
        self.assertEqual(d.targets["BTC/USD"], 0.0)
        self.assertEqual(d.reasons["ETH/USD"], HOLD)

    def test_trend_exit(self):
        signals = {"BTC/USD": sig(), "ETH/USD": sig(fast=95.0)}
        state = StrategyState(positions={"ETH/USD": PositionInfo(0, 110.0)})
        d = make_strategy(signals).decide(0, 1.0, {}, state)
        self.assertEqual(d.reasons["ETH/USD"], EXIT_TREND)
        self.assertEqual(d.targets["ETH/USD"], 0.0)

    def test_trailing_stop_uses_highest_close_since_entry(self):
        signals = {"BTC/USD": sig(), "ETH/USD": sig(close=110.0, atr=1.0)}
        state = StrategyState(positions={"ETH/USD": PositionInfo(0, 118.5)})
        d = make_strategy(signals).decide(0, 1.0, {}, state)
        self.assertEqual(d.reasons["ETH/USD"], EXIT_STOP)  # 110 < 118.5 - 8 * 1.0

        state = StrategyState(positions={"ETH/USD": PositionInfo(0, 117.5)})
        d = make_strategy(signals).decide(0, 1.0, {}, state)
        self.assertEqual(d.reasons["ETH/USD"], HOLD)

    def test_highest_close_is_tracked(self):
        signals = {"BTC/USD": sig(), "ETH/USD": sig(close=130.0)}
        state = StrategyState(positions={"ETH/USD": PositionInfo(0, 120.0)})
        make_strategy(signals).decide(0, 1.0, {}, state)
        self.assertEqual(state.positions["ETH/USD"].highest_close, 130.0)

    def test_cooldown_blocks_re_entry(self):
        signals = {"BTC/USD": sig(), "ETH/USD": sig()}
        state = StrategyState(cooldown_until={"ETH/USD": 10 * HOUR_MS})
        d = make_strategy(signals).decide(5 * HOUR_MS, 1.0, {}, state)
        self.assertEqual(d.targets["ETH/USD"], 0.0)
        d = make_strategy(signals).decide(10 * HOUR_MS, 1.0, {}, state)
        self.assertGreater(d.targets["ETH/USD"], 0.0)


class BrakeTest(unittest.TestCase):
    def test_brake_engages_at_threshold_and_releases_with_hysteresis(self):
        signals = {"BTC/USD": sig()}
        strategy = make_strategy(signals, core_weight=0.0)
        state = StrategyState()
        normal = strategy.decide(0, 100.0, {}, state).targets["BTC/USD"]
        self.assertFalse(state.brake_on)

        strategy.decide(0, 96.1, {}, state)           # 3.9% below peak: not yet
        self.assertFalse(state.brake_on)
        d = strategy.decide(0, 95.9, {}, state)       # 4.1% below peak: engaged
        self.assertTrue(d.brake_on)
        self.assertAlmostEqual(d.targets["BTC/USD"], normal / 2)

        strategy.decide(0, 97.0, {}, state)           # 3% below peak: still on
        self.assertTrue(state.brake_on)
        strategy.decide(0, 98.1, {}, state)           # 1.9% below peak: released
        self.assertFalse(state.brake_on)

    def test_brake_does_not_touch_the_core(self):
        strategy = make_strategy({})
        state = StrategyState(peak_equity=100.0)
        d = strategy.decide(0, 90.0, {}, state)
        self.assertTrue(d.brake_on)
        self.assertAlmostEqual(d.targets["PAXG/USD"], 0.05)


class ReconcileTest(unittest.TestCase):
    def test_tracks_real_holdings_and_sets_cooldown_after_stop(self):
        signals = {"BTC/USD": sig(close=110.0), "ETH/USD": sig()}
        strategy = make_strategy(signals)
        state = StrategyState(positions={"ETH/USD": PositionInfo(0, 120.0)})
        decision = strategy.decide(0, 1.0, {}, state)
        decision.reasons["ETH/USD"] = EXIT_STOP

        now = 5 * HOUR_MS
        strategy.reconcile(now, {"BTC/USD": 0.15, "PAXG/USD": 0.05}, decision, state)
        self.assertIn("BTC/USD", state.positions)
        self.assertEqual(state.positions["BTC/USD"].entry_ts, now)
        self.assertEqual(state.positions["BTC/USD"].highest_close, 110.0)
        self.assertNotIn("ETH/USD", state.positions)
        self.assertEqual(state.cooldown_until["ETH/USD"], now + 24 * HOUR_MS)
        self.assertNotIn("PAXG/USD", state.positions)  # the core alone is not a trend position

    def test_paxg_above_core_registers_only_after_an_entry(self):
        strategy = make_strategy({"PAXG/USD": sig()})
        state = StrategyState()
        strategy.reconcile(0, {"PAXG/USD": 0.08}, None, state)
        self.assertNotIn("PAXG/USD", state.positions)
        decision = strategy.decide(0, 1.0, {}, state)
        self.assertEqual(decision.reasons["PAXG/USD"], ENTRY)
        strategy.reconcile(0, {"PAXG/USD": 0.15}, decision, state)
        self.assertIn("PAXG/USD", state.positions)

    def test_state_round_trips_through_dict(self):
        state = StrategyState(positions={"BTC/USD": PositionInfo(1, 2.0)},
                              cooldown_until={"ETH/USD": 3}, peak_equity=4.0,
                              brake_on=True, last_fill_ts=5)
        self.assertEqual(StrategyState.from_dict(state.to_dict()), state)


if __name__ == "__main__":
    unittest.main()
