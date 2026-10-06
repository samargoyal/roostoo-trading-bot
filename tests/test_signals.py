"""The research queue's new signals (round 77, research options off by default)."""
import unittest
from collections import deque

from bot.config import StrategyConfig
from bot.indicators import Signal
from bot.market_data import HOUR_MS
from bot.strategy import Strategy, StrategyState

PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "PAXG/USD"]


def sig(close=100.0, atr=1.0, on=True):
    return Signal(0, close, 100.0, 100.0, 90.0, atr, 50.0, 0.0, 0.0, 0.01,
                  ema_trend_fast=101.0 if on else 99.0, ema_trend_slow=100.0)


def strategy(**options):
    return Strategy(StrategyConfig(universe=list(PAIRS), **options))


class SignalTest(unittest.TestCase):
    def test_beta_hedge_shorts_btc_paid_for_by_the_rotation(self):
        s = strategy(beta_hedge=0.5)
        s._beta = lambda pair: 2.0
        out = s._research_risk_breakers(0, 1.0, {"ETH/USD": 0.35, "SOL/USD": 0.35, "BTC/USD": 0.1},
                                        {"ETH/USD": 0.5, "SOL/USD": 0.5}, {}, {p: sig() for p in PAIRS},
                                        StrategyState(), set())
        # hedge = 0.5 x (0.35 x 2 + 0.35 x 2) = 0.7, capped at half the rotation's 0.7 -> 0.35
        self.assertAlmostEqual(out["ETH/USD"], 0.175)
        self.assertAlmostEqual(out["BTC/USD"], 0.1 - 0.35)

    def test_permutation_entropy_of_a_trend_and_of_noise(self):
        s = strategy()
        s.indicators["ETH/USD"].closes.extend(float(i) for i in range(200))
        noisy = [100.0 + (7 * i) % 11 - (3 * i) % 5 for i in range(200)]
        s.indicators["SOL/USD"].closes.extend(noisy)
        self.assertAlmostEqual(s._entropy("ETH/USD", 3), 0.0)
        self.assertGreater(s._entropy("SOL/USD", 3), 0.5)

    def test_volume_acceleration(self):
        s = strategy(volume_accel_hours=2)
        s.indicators["ETH/USD"].dollar.extend([1.0, 1.0, 1.0, 2.0, 4.0])
        s.indicators["SOL/USD"].dollar.extend([1.0, 2.0, 4.0, 4.0, 4.0])   # growth stalling
        self.assertTrue(s._volume_accelerating("ETH/USD"))
        self.assertFalse(s._volume_accelerating("SOL/USD"))

    def test_capitulation_buys_a_shock_bar_closing_off_its_low_and_exits_in_a_day(self):
        s = strategy(capitulation_size=0.02)
        ind = s.indicators["ETH/USD"]
        ind.returns = deque([0.01, -0.01] * 100 + [-0.08])
        ind.dollar.extend([1.0] * 700 + [10.0])
        ind.highs.append(110.0)
        ind.lows.append(90.0)
        state, out = StrategyState(), {}
        signals = {"BTC/USD": sig(), "ETH/USD": sig(close=100.0, atr=2.0)}
        s._capitulation(10 * HOUR_MS, out, signals, state, set())
        self.assertEqual(out, {"ETH/USD": 0.02})
        out = {}
        ind.returns.append(0.0)                                 # a day later, an ordinary hour
        s._capitulation(35 * HOUR_MS, out, {"BTC/USD": sig(), "ETH/USD": sig(close=99.0)}, state, set())
        self.assertNotIn("ETH/USD", state.capitulation)

    def test_returns_by_weekday_and_by_session(self):
        s = strategy()
        ind = s.indicators["ETH/USD"]
        monday = 4 * 24 * HOUR_MS                                # 5 January 1970 was a Monday
        ind.returns = deque([0.01] * (7 * 24))                   # a week of hourly bars, Monday to Sunday
        ind.last_ts = monday + (7 * 24 - 1) * HOUR_MS
        self.assertAlmostEqual(s._hours_return("ETH/USD", 7 * 24), 1.68)
        self.assertAlmostEqual(s._hours_return("ETH/USD", 7 * 24, skip_days=[5, 6]), 1.20)
        self.assertAlmostEqual(s._hours_return("ETH/USD", 7 * 24, hours_of_day=range(13, 21)), 0.56)

    def test_not_while_btc_is_falling_when_limited_to_uptrends(self):
        s = strategy(capitulation_size=0.02)
        state = StrategyState()
        s._capitulation(0, {}, {"BTC/USD": sig(on=False)}, state, set())
        self.assertEqual(state.capitulation, {})


if __name__ == "__main__":
    unittest.main()
