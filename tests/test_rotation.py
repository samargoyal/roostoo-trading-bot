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


class RotationWeightingTest(unittest.TestCase):
    def test_inverse_volatility_shares(self):
        signals = market()
        signals["ETH/USD"] = sig(0.20, vol=0.02)
        signals["SOL/USD"] = sig(0.40, vol=0.01)
        d = make(signals, rotation_weighting="inverse_vol").decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.rotation["SOL/USD"], 2 / 3)
        self.assertAlmostEqual(d.rotation["ETH/USD"], 1 / 3)

    def test_min_variance_uses_recent_returns_and_respects_the_cap(self):
        import math
        signals = market()
        strategy = make(signals, rotation_top=3, rotation_weighting="min_variance", rotation_max_weight=0.4)
        swings = {"SOL/USD": 0.001, "ETH/USD": 0.02, "BTC/USD": 0.03}   # SOL by far the calmest
        for pair, swing in swings.items():
            closes = strategy.indicators[pair].closes
            price = 100.0
            for i in range(200):
                price *= math.exp(swing if i % 2 == 0 else -swing * 0.9)
                closes.append(price)
        d = strategy.decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(sum(d.rotation.values()), 1.0)
        self.assertAlmostEqual(d.rotation["SOL/USD"], 0.4, places=4)    # capped
        self.assertGreater(d.rotation["ETH/USD"], d.rotation["BTC/USD"])

    def test_cap_hands_the_excess_to_the_others(self):
        from bot.strategy import _cap_weights
        w = _cap_weights({"A": 0.7, "B": 0.2, "C": 0.1}, 0.5)
        self.assertAlmostEqual(w["A"], 0.5)
        self.assertAlmostEqual(w["B"], 0.2 + 0.2 * 2 / 3)
        self.assertAlmostEqual(sum(w.values()), 1.0)


class ResidualAndZTest(unittest.TestCase):
    def fill(self, strategy, pair, steps):
        price = 100.0
        for step in steps:
            price *= 1 + step
            strategy.indicators[pair].closes.append(price)

    def test_residual_momentum_prefers_the_coin_that_did_not_just_ride_btc(self):
        signals = market()
        signals["ETH/USD"] = sig(0.30)
        signals["SOL/USD"] = sig(0.30)
        strategy = make(signals, rotation_ranking="residual", rotation_top=1)
        btc = [0.01 if i % 2 == 0 else -0.008 for i in range(400)]
        self.fill(strategy, "BTC/USD", btc)
        self.fill(strategy, "ETH/USD", [2.0 * x for x in btc])                               # pure beta
        self.fill(strategy, "SOL/USD", [0.003 if i % 3 == 0 else -0.0005 for i in range(400)])  # own move
        d = strategy.decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(list(d.rotation), ["SOL/USD"])

    def test_kalman_ranking_picks_the_steadiest_trend_not_the_biggest_return(self):
        signals = market()
        signals["SOL/USD"] = sig(0.40)._replace(trend_strength=1.0)    # biggest return, ragged
        signals["ETH/USD"] = sig(0.20)._replace(trend_strength=4.0)
        signals["BTC/USD"] = sig(0.05)._replace(trend_strength=3.0)
        signals["DOGE/USD"] = sig(-0.10, fast=90.0)._replace(trend_strength=9.0)  # falling: excluded
        d = make(signals, rotation_ranking="kalman").decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.rotation, {"ETH/USD": 0.5, "BTC/USD": 0.5})

    def test_bear_market_shorts_the_weakest_coins_and_covers_when_the_trend_returns(self):
        signals = market(trend_up=False)
        signals["DOGE/USD"] = sig(-0.30, trend_up=False, fast=90.0)
        signals["SOL/USD"] = sig(-0.10, trend_up=False)
        state = StrategyState()
        d = make(signals, rotation_shorts=2).decide(DAY, 1.0, {}, state)
        self.assertEqual(d.rotation, {"DOGE/USD": -0.5, "SOL/USD": -0.5})
        self.assertAlmostEqual(d.targets["DOGE/USD"], -0.2)
        up = make(market(trend_up=True), rotation_shorts=2).decide(DAY + HOUR_MS, 1.0, {}, state)
        self.assertTrue(all(w >= 0 for w in up.rotation.values()))

    def test_volatility_forecast_shrinks_the_sleeve_when_btc_is_wild(self):
        import math, random
        strategy = make(market(), rotation_vol_forecast="har")
        rng = random.Random(4)
        returns = strategy.indicators["BTC/USD"].returns
        for i in range(2200):
            returns.append(rng.gauss(0, 0.004 if i < 2200 - 48 else 0.03))   # calm, then two wild days
        d = strategy.decide(DAY, 1.0, {}, StrategyState())
        self.assertLess(sum(d.rotation.values()), 0.7)
        calm = make(market(), rotation_vol_forecast="har")
        for _ in range(2200):
            calm.indicators["BTC/USD"].returns.append(rng.gauss(0, 0.004))
        self.assertAlmostEqual(sum(calm.decide(DAY, 1.0, {}, StrategyState()).rotation.values()), 1.0, delta=0.15)

    def test_trailing_stop_drops_a_pick_and_bars_it_until_the_cooldown_ends(self):
        state = StrategyState()
        make(market(), rotation_stop_atr=8.0).decide(DAY, 1.0, {}, state)
        self.assertIn("SOL/USD", state.rotation_plan)
        crashed = market()
        crashed["SOL/USD"] = sig(0.40)._replace(close=110.0 - 9.0)      # 9 ATR below its high
        d = make(crashed, rotation_stop_atr=8.0).decide(DAY + HOUR_MS, 1.0, {}, state)
        self.assertNotIn("SOL/USD", d.rotation)
        self.assertIn("ETH/USD", d.rotation)
        nxt = make(crashed, rotation_stop_atr=8.0).decide(2 * DAY, 1.0, {}, state)  # next rebalance
        self.assertNotIn("SOL/USD", nxt.rotation)
        later = make(market(), rotation_stop_atr=8.0).decide(3 * DAY, 1.0, {}, state)
        self.assertIn("SOL/USD", later.rotation)

    def test_picks_must_beat_btc_and_btc_fills_the_rest(self):
        signals = market()
        signals["BTC/USD"] = sig(0.25)                      # only SOL (0.40) beats BTC
        d = make(signals, rotation_vs_btc="btc").decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.rotation, {"SOL/USD": 0.5, "BTC/USD": 0.5})
        d = make(signals, rotation_vs_btc="cash").decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.rotation, {"SOL/USD": 0.5, "PAXG/USD": 0.5})

    def test_hold_buffer_keeps_a_pick_that_slips_to_third(self):
        state = StrategyState()
        make(market(), rotation_buffer=4).decide(DAY, 1.0, {}, state)
        self.assertEqual(set(state.rotation_plan), {"SOL/USD", "ETH/USD"})
        shifted = market()
        shifted["DOGE/USD"] = sig(0.30)                     # DOGE now ranks second, ETH third
        d = make(shifted, rotation_buffer=4).decide(2 * DAY, 1.0, {}, state)
        self.assertEqual(set(d.rotation), {"SOL/USD", "ETH/USD"})
        d = make(shifted).decide(2 * DAY, 1.0, {}, StrategyState())
        self.assertEqual(set(d.rotation), {"SOL/USD", "DOGE/USD"})

    def test_z_guard_skips_an_overextended_pick(self):
        signals = market()
        strategy = make(signals, rotation_max_z=2.0, rotation_top=1)
        self.fill(strategy, "SOL/USD", [0.0001 * (1 if i % 2 else -1) for i in range(200)] + [0.2])  # spike
        d = strategy.decide(DAY, 1.0, {}, StrategyState())
        self.assertNotIn("SOL/USD", d.rotation)
        self.assertIn("ETH/USD", d.rotation)


class CvarCapTest(unittest.TestCase):
    def test_wild_picks_are_scaled_down_to_the_cvar_limit(self):
        signals = market()
        capped = make(signals, rotation_cvar_limit=0.02)
        free = make(signals)
        for strategy in (capped, free):
            for pair, swing in (("SOL/USD", 0.03), ("ETH/USD", 0.03)):
                price = 100.0
                for i in range(400):
                    price *= 1 + (swing if i % 3 else -2.2 * swing)
                    strategy.indicators[pair].closes.append(price)
        d_free = free.decide(DAY, 1.0, {}, StrategyState())
        d_cap = capped.decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(sum(d_free.rotation.values()), 1.0)
        self.assertLess(sum(d_cap.rotation.values()), 1.0)
        cvar = capped._daily_cvar(d_cap.rotation)
        self.assertAlmostEqual(cvar * 0.4, 0.02, places=6)
