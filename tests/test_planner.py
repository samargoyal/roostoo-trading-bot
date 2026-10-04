import unittest

from bot.config import ExecutionConfig
from bot.market_data import HOUR_MS
from bot.planner import (ACTIVITY, BUY, COVER, EXIT_UNIVERSE, FUNDING, REBALANCE, SELL, SHORT,
                         activity_due, plan_trades)
from bot.strategy import ENTRY, EXIT_TREND, HOLD, Decision

DAY = 24 * HOUR_MS
CFG = ExecutionConfig()  # 4% threshold, $10 minimum, 8-hour activity blocks


def decision(targets, reasons=None):
    return Decision(ts=0, risk_on=True, exposure_limit=0.75, drawdown=0.0, brake_on=False,
                    targets=targets, reasons=reasons or {}, scores={})


def plan(targets, weights, reasons=None, ts=DAY + HOUR_MS, last_fill=DAY):
    return plan_trades(decision(targets, reasons), weights, 100000.0, ts, last_fill, CFG, 0.005)


class ThresholdTest(unittest.TestCase):
    def test_small_drift_is_ignored_large_drift_rebalanced(self):
        self.assertEqual(plan({"BTC/USD": 0.15}, {"BTC/USD": 0.13}, {"BTC/USD": HOLD}), [])
        trades = plan({"BTC/USD": 0.15}, {"BTC/USD": 0.10}, {"BTC/USD": HOLD})
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].side, BUY)
        self.assertEqual(trades[0].reason, REBALANCE)
        self.assertAlmostEqual(trades[0].usd, 5000.0)

    def test_entries_and_exits_ignore_the_threshold(self):
        trades = plan({"BTC/USD": 0.02, "ETH/USD": 0.0}, {"ETH/USD": 0.03},
                      {"BTC/USD": ENTRY, "ETH/USD": EXIT_TREND})
        by_pair = {t.pair: t for t in trades}
        self.assertEqual(by_pair["BTC/USD"].reason, ENTRY)
        self.assertEqual(by_pair["ETH/USD"].reason, EXIT_TREND)
        self.assertTrue(by_pair["ETH/USD"].close_position)

    def test_sells_come_before_buys(self):
        trades = plan({"BTC/USD": 0.15, "ETH/USD": 0.0}, {"ETH/USD": 0.10},
                      {"BTC/USD": ENTRY, "ETH/USD": EXIT_TREND})
        self.assertEqual([t.side for t in trades], [SELL, BUY])

    def test_holdings_outside_the_universe_are_sold(self):
        trades = plan({"BTC/USD": 0.0}, {"OLD/USD": 0.05})
        self.assertEqual(len(trades), 1)
        self.assertEqual((trades[0].pair, trades[0].side, trades[0].reason),
                         ("OLD/USD", SELL, EXIT_UNIVERSE))
        self.assertTrue(trades[0].close_position)

    def test_entries_beyond_free_cash_are_paid_for_by_trimming_the_most_overweight(self):
        # Fully invested: SOL enters at 10% while BTC and ETH sit 6% and 4% above their targets.
        trades = plan({"BTC/USD": 0.47, "ETH/USD": 0.43, "SOL/USD": 0.10}, {"BTC/USD": 0.53, "ETH/USD": 0.47},
                      {"BTC/USD": HOLD, "ETH/USD": HOLD, "SOL/USD": ENTRY})
        by_pair = {t.pair: t for t in trades}
        self.assertEqual(by_pair["SOL/USD"].side, BUY)
        self.assertEqual(by_pair["BTC/USD"].reason, REBALANCE)           # 6% off: the threshold trims it
        self.assertEqual((by_pair["ETH/USD"].side, by_pair["ETH/USD"].reason), (SELL, FUNDING))
        self.assertAlmostEqual(by_pair["ETH/USD"].usd, 4000.0)            # down to its target, no further
        self.assertEqual([t.side for t in trades][-1], BUY)               # sales first

    def test_no_funding_sales_while_cash_covers_the_entries(self):
        trades = plan({"BTC/USD": 0.47, "SOL/USD": 0.10}, {"BTC/USD": 0.50}, {"BTC/USD": HOLD, "SOL/USD": ENTRY})
        self.assertEqual([t.pair for t in trades], ["SOL/USD"])

    def test_orders_below_the_minimum_are_dropped(self):
        trades = plan({"BTC/USD": 0.00005}, {}, {"BTC/USD": ENTRY})  # $5
        self.assertEqual(trades, [])


class ActivityTest(unittest.TestCase):
    def test_due_only_late_in_a_block_without_fills(self):
        block_end = DAY + 8 * HOUR_MS
        self.assertFalse(activity_due(block_end - 3 * HOUR_MS, DAY - HOUR_MS, CFG))
        self.assertTrue(activity_due(block_end - 2 * HOUR_MS, DAY - HOUR_MS, CFG))
        self.assertFalse(activity_due(block_end - 2 * HOUR_MS, DAY + HOUR_MS, CFG))

    def test_forces_a_trade_on_the_most_drifted_position(self):
        ts = DAY + 6 * HOUR_MS
        trades = plan({"BTC/USD": 0.15, "PAXG/USD": 0.05}, {"BTC/USD": 0.14, "PAXG/USD": 0.051},
                      {"BTC/USD": HOLD}, ts=ts, last_fill=DAY - HOUR_MS)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].reason, ACTIVITY)
        self.assertEqual(trades[0].pair, "BTC/USD")
        self.assertEqual(trades[0].side, BUY)
        self.assertAlmostEqual(trades[0].usd, 1000.0)

    def test_tiny_drift_still_trades_the_minimum(self):
        ts = DAY + 6 * HOUR_MS
        trades = plan({"PAXG/USD": 0.05}, {"PAXG/USD": 0.05}, ts=ts, last_fill=0)
        self.assertEqual(trades[0].usd, CFG.activity_min_usd)

    def test_a_book_of_shorts_only_adjusts_its_most_drifted_short(self):
        ts = DAY + 6 * HOUR_MS
        trades = plan({"SOL/USD": -0.10, "ETH/USD": -0.10}, {"SOL/USD": -0.095, "ETH/USD": -0.099},
                      ts=ts, last_fill=DAY - HOUR_MS)
        self.assertEqual(len(trades), 1)
        self.assertEqual((trades[0].pair, trades[0].side, trades[0].reason), ("SOL/USD", SHORT, ACTIVITY))
        self.assertAlmostEqual(trades[0].usd, 500.0)
        trades = plan({"SOL/USD": -0.08}, {"SOL/USD": -0.10}, ts=ts, last_fill=DAY - HOUR_MS)
        self.assertEqual(trades[0].side, COVER)
        self.assertAlmostEqual(trades[0].usd, 2000.0)

    def test_fully_invested_trims_the_most_overweight_holding_instead_of_buying(self):
        ts = DAY + 6 * HOUR_MS
        trades = plan({"BTC/USD": 0.52, "ETH/USD": 0.48}, {"BTC/USD": 0.505, "ETH/USD": 0.495},
                      ts=ts, last_fill=DAY - HOUR_MS)
        self.assertEqual((trades[0].pair, trades[0].side, trades[0].reason), ("ETH/USD", SELL, ACTIVITY))
        self.assertAlmostEqual(trades[0].usd, CFG.activity_min_usd)

    def test_no_forced_trade_when_regular_trades_exist(self):
        ts = DAY + 6 * HOUR_MS
        trades = plan({"BTC/USD": 0.15}, {}, {"BTC/USD": ENTRY}, ts=ts, last_fill=0)
        self.assertEqual([t.reason for t in trades], [ENTRY])


if __name__ == "__main__":
    unittest.main()
