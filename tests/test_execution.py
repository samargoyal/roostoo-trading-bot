import unittest
from decimal import Decimal

from bot.config import ExecutionConfig
from bot.execution import (DRY_RUN, LIMIT, MARKET, Executor, PairRules, parse_rules, round_down,
                           round_up, to_str)
from bot.planner import BUY, SELL, PlannedTrade
from bot.strategy import ENTRY, EXIT_STOP, EXIT_TREND
from tests.fakes import FakeExchange


def trade(pair, side, usd, close=False, reason=ENTRY):
    return PlannedTrade(pair, side, usd, close, 0.0, 0.1, reason)


class RoundingTest(unittest.TestCase):
    def test_round_down_to_amount_precision(self):
        self.assertEqual(round_down(0.123456789, 4), Decimal("0.1234"))
        self.assertEqual(round_down(12.99, 0), Decimal("12"))
        self.assertEqual(round_down(0.00001, 5), Decimal("0.00001"))

    def test_round_up_for_sell_prices(self):
        self.assertEqual(round_up(1.231, 2), Decimal("1.24"))
        self.assertEqual(round_up(1.23, 2), Decimal("1.23"))

    def test_never_scientific_notation(self):
        self.assertEqual(to_str(round_down(1e-05, 5)), "0.00001")
        self.assertEqual(to_str(round_down(150.0, 0)), "150")

    def test_parse_rules_skips_untradable_pairs(self):
        info = {"TradePairs": {
            "BTC/USD": {"PricePrecision": 2, "AmountPrecision": 5, "MiniOrder": 1, "CanTrade": True},
            "OLD/USD": {"PricePrecision": 2, "AmountPrecision": 5, "MiniOrder": 1, "CanTrade": False}}}
        rules = parse_rules(info)
        self.assertEqual(rules["BTC/USD"], PairRules(2, 5, 1.0))
        self.assertNotIn("OLD/USD", rules)


class ExecutorTest(unittest.TestCase):
    def make(self, exchange, dry_run=False, **cfg):
        self.recorded = []
        self.clock = [0.0]

        def sleep(seconds):
            self.clock[0] += seconds

        rules = parse_rules(exchange.exchange_info())
        return Executor(exchange, rules, ExecutionConfig(**cfg), self.recorded.append,
                        dry_run=dry_run, sleep=sleep, clock=lambda: self.clock[0])

    def test_limit_buy_rests_at_the_bid_and_fills(self):
        ex = FakeExchange({"BTC/USD": 50000.0})
        results = self.make(ex).execute([trade("BTC/USD", BUY, 10000.0)], ex.ticker())
        self.assertEqual(len(results), 1)
        order = ex.placed[0]
        self.assertEqual(order["Type"], LIMIT)
        self.assertEqual(order["Price"], ex.ticker()["BTC/USD"]["MaxBid"])
        self.assertEqual(results[0].status, "FILLED")
        self.assertAlmostEqual(ex.wallet["BTC"]["Free"], 0.1999)  # 10000 / ask, rounded down
        self.assertEqual(len(self.recorded), 1)

    def test_unfilled_limit_is_cancelled_and_sent_at_market(self):
        ex = FakeExchange({"BTC/USD": 50000.0}, fill_limits=False)
        results = self.make(ex, limit_timeout_sec=60, poll_interval_sec=30).execute(
            [trade("BTC/USD", BUY, 10000.0)], ex.ticker())
        self.assertEqual([r.order_type for r in results], [LIMIT, MARKET])
        self.assertEqual(results[0].status, "CANCELED")
        self.assertEqual(results[1].status, "FILLED")
        self.assertEqual(ex.wallet["USD"]["Lock"], 0.0)
        self.assertAlmostEqual(ex.wallet["BTC"]["Free"], 0.1999)

    def test_stop_exit_goes_straight_to_market(self):
        ex = FakeExchange({"BTC/USD": 50000.0})
        ex.wallet["BTC"] = {"Free": 0.5, "Lock": 0.0}
        self.make(ex).execute([trade("BTC/USD", SELL, 0.0, close=True, reason=EXIT_STOP)], ex.ticker())
        self.assertEqual(ex.placed[0]["Type"], MARKET)
        self.assertEqual(ex.placed[0]["Quantity"], 0.5)

    def test_sell_is_capped_at_the_free_balance(self):
        ex = FakeExchange({"ETH/USD": 2000.0})
        ex.wallet["ETH"] = {"Free": 1.0, "Lock": 0.0}
        self.make(ex).execute([trade("ETH/USD", SELL, 5000.0, reason=EXIT_TREND)], ex.ticker())
        self.assertEqual(ex.placed[0]["Quantity"], 1.0)

    def test_buys_are_scaled_to_available_cash(self):
        ex = FakeExchange({"BTC/USD": 50000.0, "ETH/USD": 2000.0}, cash=10000.0)
        self.make(ex, use_limit_orders=False).execute(
            [trade("BTC/USD", BUY, 10000.0), trade("ETH/USD", BUY, 10000.0)], ex.ticker())
        spent = sum(o["Quantity"] * o["FilledAverPrice"] for o in ex.placed)
        self.assertLessEqual(spent, 10000.0 * 0.99)
        self.assertGreater(spent, 10000.0 * 0.98)
        self.assertGreaterEqual(ex.wallet["USD"]["Free"], 0.0)

    def test_orders_below_the_minimum_are_skipped(self):
        ex = FakeExchange({"BTC/USD": 50000.0})
        ex.wallet["BTC"] = {"Free": 0.00001, "Lock": 0.0}  # worth 0.50 USD, below MiniOrder
        results = self.make(ex).execute(
            [trade("BTC/USD", BUY, 5.0), trade("BTC/USD", SELL, 0.0, close=True, reason=EXIT_TREND)],
            ex.ticker())
        self.assertEqual(results, [])
        self.assertEqual(ex.placed, [])

    def test_whole_coin_precision(self):
        ex = FakeExchange({"DOGE/USD": 0.1}, amount_decimals=0)
        self.make(ex, use_limit_orders=False).execute([trade("DOGE/USD", BUY, 99.95)], ex.ticker())
        self.assertEqual(ex.placed[0]["Quantity"], 999.0)  # 999.5 coins, rounded down

    def test_dry_run_sends_nothing(self):
        ex = FakeExchange({"BTC/USD": 50000.0})
        results = self.make(ex, dry_run=True).execute([trade("BTC/USD", BUY, 10000.0)], ex.ticker())
        self.assertEqual(ex.placed, [])
        self.assertEqual(results[0].status, DRY_RUN)
        self.assertEqual(len(self.recorded), 1)


if __name__ == "__main__":
    unittest.main()
