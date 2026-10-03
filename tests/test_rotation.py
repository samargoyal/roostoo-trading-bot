"""The momentum rotation sleeve."""
import unittest

from bot.config import StrategyConfig
from bot.indicators import Signal
from bot.market_data import HOUR_MS
from bot.strategy import ENTRY, ROTATION, SHORT_ENTRY, Strategy, StrategyState

UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "PAXG/USD"]
DAY = 24 * HOUR_MS


def sig(rot=0.0, trend_up=True, vol=0.01, fast=105.0):
    """A signal with a given 336h return; BTC's slow trend filter on or off."""
    return Signal(0, 110.0, fast, 100.0, 90.0, 1.0, 50.0, 0.0, 0.0, vol,
                  return_rotation=rot, ema_trend_fast=101.0 if trend_up else 99.0, ema_trend_slow=100.0)


def market(trend_up=True):
    return {"BTC/USD": sig(0.05, trend_up), "ETH/USD": sig(0.20, trend_up), "SOL/USD": sig(0.40, trend_up),
            "DOGE/USD": sig(-0.10, trend_up, fast=90.0), "PAXG/USD": sig(0.02, trend_up, vol=0.002)}


def make(signals, **overrides):
    cfg = StrategyConfig(universe=list(UNIVERSE), **overrides)
    strategy = Strategy(cfg)
    strategy.signals = lambda: signals
    return strategy


class RotationTest(unittest.TestCase):
    def test_holds_the_two_strongest_rising_coins_with_40_percent_of_equity(self):
        d = make(market()).decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.rotation, {"SOL/USD": 0.5, "ETH/USD": 0.5})
        # SOL is not held by the other book, so its whole target is the sleeve's 0.4 x 0.5
        self.assertAlmostEqual(d.targets["SOL/USD"], 0.2 + 0.6 * (d.targets["SOL/USD"] - 0.2) / 0.6)
        self.assertGreaterEqual(d.targets["SOL/USD"], 0.2)
        self.assertNotIn("DOGE/USD", d.rotation)          # falling: absolute momentum fails

    def test_the_other_book_is_scaled_to_60_percent(self):
        alone = make(market(), rotation_weight=0.0).decide(DAY, 1.0, {}, StrategyState())
        both = make(market()).decide(DAY, 1.0, {}, StrategyState())
        for pair in ("BTC/USD", "PAXG/USD"):
            expected = 0.6 * alone.targets[pair] + 0.4 * both.rotation.get(pair, 0.0)
            self.assertAlmostEqual(both.targets[pair], expected)

    def test_empty_slots_go_to_paxg_when_it_is_rising(self):
        signals = market()
        signals["ETH/USD"] = sig(-0.05)
        signals["BTC/USD"] = sig(-0.01)                    # only SOL is still rising
        d = make(signals).decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.rotation, {"SOL/USD": 0.5, "PAXG/USD": 0.5})

    def test_trend_filter_off_means_out_at_once(self):
        state = StrategyState()
        make(market()).decide(DAY, 1.0, {}, state)
        d = make(market(trend_up=False)).decide(DAY + HOUR_MS, 1.0, {}, state)  # not a rebalance hour
        self.assertEqual(d.rotation, {})
        self.assertEqual(state.rotation_plan, {"SOL/USD": 0.5, "ETH/USD": 0.5})  # plan kept for later

    def test_plan_holds_between_rebalances_and_changes_at_midnight(self):
        state = StrategyState()
        make(market()).decide(DAY, 1.0, {}, state)
        later = market()
        later["DOGE/USD"] = sig(0.90)
        d = make(later).decide(DAY + 5 * HOUR_MS, 1.0, {}, state)
        self.assertNotIn("DOGE/USD", d.rotation)
        d = make(later).decide(2 * DAY, 1.0, {}, state)
        self.assertIn("DOGE/USD", d.rotation)

    def test_missed_rebalance_is_caught_up(self):
        state = StrategyState()
        make(market()).decide(DAY, 1.0, {}, state)
        later = market()
        later["DOGE/USD"] = sig(0.90)
        d = make(later).decide(2 * DAY + 3 * HOUR_MS, 1.0, {}, state)  # the midnight cycle was missed
        self.assertIn("DOGE/USD", d.rotation)

    def test_rotation_coins_are_not_positions_of_the_other_book(self):
        strategy = make(market())
        state = StrategyState()
        d = strategy.decide(DAY, 1.0, {}, state)
        strategy.reconcile(DAY, {"SOL/USD": 0.2, "ETH/USD": 0.2}, d, state)
        self.assertNotIn("SOL/USD", state.positions)

    def test_never_shorts_a_rotation_coin(self):
        signals = market()
        signals["SOL/USD"] = sig(0.40, vol=0.09)          # the most volatile coin, but held by the sleeve
        d = make(signals, short_exposure=0.15).decide(DAY, 1.0, {}, StrategyState())
        self.assertNotEqual(d.reasons.get("SOL/USD"), SHORT_ENTRY)
        self.assertGreater(d.targets["SOL/USD"], 0)

    def test_switched_off(self):
        d = make(market(), rotation_weight=0.0).decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.rotation, {})
        self.assertNotIn(ROTATION, d.reasons.values())


if __name__ == "__main__":
    unittest.main()
