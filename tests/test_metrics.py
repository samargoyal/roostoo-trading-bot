import unittest

from bot.metrics import DAY_MS, composite, max_drawdown, sharpe, sortino, summarize

HOUR = DAY_MS // 24


class MetricsTest(unittest.TestCase):
    def test_max_drawdown(self):
        self.assertAlmostEqual(max_drawdown([100, 120, 90, 130, 117]), 0.25)
        self.assertEqual(max_drawdown([1, 2, 3]), 0.0)

    def test_ratios_of_a_flat_series_are_zero(self):
        self.assertEqual(sharpe([0.0, 0.0, 0.0]), 0.0)
        self.assertEqual(sortino([0.01, 0.02]), 0.0)  # no downside at all

    def test_composite_weights(self):
        self.assertAlmostEqual(composite(1.0, 2.0, 3.0), 0.4 + 0.6 + 0.9)

    def test_summary_of_a_steady_climb(self):
        curve = [(i * HOUR, 100.0 * (1 + 0.0001 * i)) for i in range(1, 24 * 30 + 1)]
        stats = summarize(curve, 100.0, [(HOUR, 50.0)])
        self.assertAlmostEqual(stats["total_return"], 0.072)
        self.assertEqual(stats["max_drawdown"], 0.0)
        self.assertEqual(stats["trades"], 1.0)
        self.assertEqual(stats["window_positive_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
