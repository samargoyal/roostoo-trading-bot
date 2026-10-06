"""Risk circuit breakers (round 75, research options off by default)."""
import unittest
from collections import deque

from bot.config import StrategyConfig
from bot.indicators import Signal
from bot.market_data import HOUR_MS
from bot.strategy import Strategy, StrategyState

DAY = 24 * HOUR_MS
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD"]


def sig(close=100.0, atr=1.0):
    return Signal(0, close, 100.0, 100.0, 90.0, atr, 50.0, 0.0, 0.0, 0.01)


def strategy(**options):
    s = Strategy(StrategyConfig(universe=list(PAIRS), rotation_weight=0.0, **options))
    for p in PAIRS:
        s.indicators[p].closes.extend([100.0] * 30)
    return s


class RiskBreakersTest(unittest.TestCase):
    signals = {p: sig() for p in PAIRS}

    def test_all_off_changes_nothing(self):
        targets = {"BTC/USD": 0.5, "ETH/USD": -0.2}
        out = strategy()._research_risk_breakers(DAY, 1.0, targets, {}, {}, self.signals, StrategyState(), set())
        self.assertEqual(out, targets)

    def test_day_loss_stops_new_entries_but_lets_positions_shrink(self):
        s, state = strategy(day_loss_stop=0.03), StrategyState()
        s._research_risk_breakers(DAY, 1.00, {}, {}, {}, self.signals, state, set())     # 00:00 value
        out = s._research_risk_breakers(DAY + 5 * HOUR_MS, 0.96, {"BTC/USD": 0.5, "ETH/USD": 0.1, "SOL/USD": -0.3},
                                        {}, {"BTC/USD": 0.3, "ETH/USD": 0.2}, self.signals, state, set())
        self.assertEqual(out, {"BTC/USD": 0.3, "ETH/USD": 0.1, "SOL/USD": 0.0})
        out = s._research_risk_breakers(2 * DAY, 0.96, {"BTC/USD": 0.5}, {}, {"BTC/USD": 0.3},
                                        self.signals, state, set())                     # a new day
        self.assertEqual(out, {"BTC/USD": 0.5})

    def test_drawdown_ladder_steps_down_and_releases_with_hysteresis(self):
        s, state = strategy(dd_ladder=[0.04, 0.07, 0.10]), StrategyState(peak_equity=1.0)
        run = lambda ts, eq: s._research_risk_breakers(ts, eq, {"BTC/USD": 1.0}, {}, {}, self.signals, state, set())
        self.assertEqual(run(0, 0.95), {"BTC/USD": 0.5})
        self.assertEqual(run(HOUR_MS, 0.92), {"BTC/USD": 0.25})
        self.assertEqual(run(2 * HOUR_MS, 0.99), {"BTC/USD": 0.25})          # recovered, but too soon
        self.assertEqual(run(15 * HOUR_MS, 0.99), {"BTC/USD": 0.5})          # one step every 12 hours
        self.assertEqual(run(27 * HOUR_MS, 0.99), {"BTC/USD": 1.0})

    def test_squeeze_covers_the_short_and_caps_the_rest(self):
        s = strategy(squeeze_rise=0.10)
        s.indicators["SOL/USD"].closes.append(112.0)
        state = StrategyState()
        out = s._research_risk_breakers(DAY, 1.0, {"SOL/USD": -0.1, "ETH/USD": -0.3}, {},
                                        {"SOL/USD": -0.1, "ETH/USD": -0.3},
                                        dict(self.signals, **{"SOL/USD": sig(112.0)}), state, set())
        self.assertEqual(out["SOL/USD"], 0.0)
        self.assertAlmostEqual(out["ETH/USD"], -0.15)
        self.assertEqual(state.short_cooldown_until["SOL/USD"], DAY + 48 * HOUR_MS)

    def test_volatility_ratio(self):
        s = strategy(vol_regime=2.5)
        s.indicators["BTC/USD"].returns = deque([0.01] * 720 + [0.03] * 24)
        self.assertAlmostEqual(s._vol_ratio(), 9.0)


if __name__ == "__main__":
    unittest.main()
