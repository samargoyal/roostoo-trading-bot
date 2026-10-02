import math
import unittest

from bot.indicators import ATR, EMA, RSI, IndicatorSet, RollingStd
from bot.market_data import HOUR_MS, Bar


class EMATest(unittest.TestCase):
    def test_seeded_with_simple_average_then_smoothed(self):
        ema = EMA(3)
        self.assertIsNone(ema.update(1.0))
        self.assertIsNone(ema.update(2.0))
        self.assertAlmostEqual(ema.update(3.0), 2.0)
        self.assertAlmostEqual(ema.update(6.0), 2.0 + 0.5 * (6.0 - 2.0))

    def test_constant_series_stays_constant(self):
        ema = EMA(10)
        for _ in range(50):
            value = ema.update(5.0)
        self.assertAlmostEqual(value, 5.0)


class ATRTest(unittest.TestCase):
    def test_true_range_uses_previous_close_gaps(self):
        atr = ATR(2)
        atr.update(10.0, 9.0, 9.5)              # range 1.0
        value = atr.update(12.0, 11.0, 11.5)    # gap up: true range 12 - 9.5 = 2.5
        self.assertAlmostEqual(value, (1.0 + 2.5) / 2)
        value = atr.update(11.6, 11.4, 11.5)    # range 0.2; Wilder: 1.75 + (0.2 - 1.75) / 2
        self.assertAlmostEqual(value, 1.75 + (0.2 - 1.75) / 2)


class RSITest(unittest.TestCase):
    def test_only_rising_prices_give_100(self):
        rsi = RSI(3)
        for price in [1, 2, 3, 4, 5]:
            value = rsi.update(float(price))
        self.assertEqual(value, 100.0)

    def test_equal_gains_and_losses_give_50(self):
        rsi = RSI(2)
        for price in [10, 11, 10]:
            value = rsi.update(float(price))
        self.assertAlmostEqual(value, 50.0)

    def test_flat_prices_give_50(self):
        rsi = RSI(2)
        for _ in range(5):
            value = rsi.update(10.0)
        self.assertEqual(value, 50.0)


class RollingStdTest(unittest.TestCase):
    def test_matches_direct_sample_stdev_over_the_window(self):
        xs = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02]
        std = RollingStd(4)
        for x in xs:
            value = std.update(x)
        window = xs[-4:]
        mean = sum(window) / 4
        expected = math.sqrt(sum((x - mean) ** 2 for x in window) / 3)
        self.assertAlmostEqual(value, expected)


class IndicatorSetTest(unittest.TestCase):
    def make(self):
        return IndicatorSet(fast_ema=3, slow_ema=5, regime_ema=5, atr_period=3, rsi_period=3,
                            momentum_short=2, momentum_long=4, volatility_window=3)

    def test_signal_only_after_warm_up_and_returns_are_correct(self):
        ind = self.make()
        closes = [100, 101, 102, 103, 104, 105, 106]
        for i, c in enumerate(closes):
            ind.update(Bar(i * HOUR_MS, c, c + 1, c - 1, c, 0.0))
            if i < 4:
                self.assertIsNone(ind.signal())
        s = ind.signal()
        self.assertIsNotNone(s)
        self.assertEqual(s.close, 106)
        self.assertAlmostEqual(s.return_short, 106 / 104 - 1)
        self.assertAlmostEqual(s.return_long, 106 / 102 - 1)
        self.assertGreater(s.ema_fast, s.ema_slow)

    def test_repeated_bar_is_ignored(self):
        ind = self.make()
        bar = Bar(0, 1.0, 1.0, 1.0, 1.0, 0.0)
        ind.update(bar)
        ind.update(bar)
        self.assertEqual(len(ind.closes), 1)


if __name__ == "__main__":
    unittest.main()
