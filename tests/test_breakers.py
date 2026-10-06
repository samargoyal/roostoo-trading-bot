"""Operational circuit breakers (bot/breakers.py), by fault injection."""
import csv
import os
import shutil
import tempfile
import unittest

from bot.breakers import Breakers
from bot.config import BreakerConfig, Config
from bot.execution import Executor, OrderResult, parse_rules
from bot.live import LiveBot
from bot.market_data import HOUR_MS, Bar
from bot.planner import ACTIVITY, BUY, COVER, SELL, SHORT, PlannedTrade
from bot.roostoo import RoostooError
from tests.fakes import NOW, FakeBinance, FakeExchange, zigzag

PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD"]


def quote(price, spread=0.0002):
    return {"MaxBid": price * (1 - spread / 2), "MinAsk": price * (1 + spread / 2), "LastPrice": price}


def bar(price, ts=NOW - 2 * HOUR_MS, move=0.0):
    return Bar(ts, price / (1 + move), price, price, price, 1.0)


def trade(pair, side, usd=1000.0, reason="rotation"):
    return PlannedTrade(pair, side, usd, side in (SELL, COVER), 0.0, 0.0, reason)


class ScreenTest(unittest.TestCase):
    def setUp(self):
        self.b = Breakers(BreakerConfig(enabled=True))
        self.quotes = {p: quote(100.0) for p in PAIRS}
        self.bars = {p: bar(100.0) for p in PAIRS}

    def screen(self, trades, equity=100000.0, failed=frozenset(), quantities=None):
        return self.b.screen(trades, NOW, equity, self.quotes, self.bars, set(failed), quantities or {})

    def test_quiet_market_passes_everything(self):
        trades = [trade("BTC/USD", BUY), trade("ETH/USD", SELL)]
        allowed, limit_only, trips = self.screen(trades)
        self.assertEqual((allowed, limit_only, trips), (trades, set(), []))

    def test_a10_halt_and_reduce_only(self):
        trades = [trade("BTC/USD", BUY), trade("ETH/USD", SELL), trade("SOL/USD", BUY, 50, ACTIVITY)]
        self.b.cfg.trading_halt = True
        self.assertEqual(self.screen(trades)[0], [])
        self.b.cfg.trading_halt, self.b.cfg.reduce_only = False, True
        allowed = self.screen(trades)[0]
        self.assertEqual([(t.pair, t.side) for t in allowed], [("ETH/USD", SELL), ("SOL/USD", BUY)])

    def test_a1_divergent_pair_gets_no_orders_and_many_skip_the_cycle(self):
        self.quotes["BTC/USD"] = quote(105.0)                    # 5% off Binance
        allowed, _, trips = self.screen([trade("BTC/USD", SELL), trade("ETH/USD", BUY)])
        self.assertEqual([t.pair for t in allowed], ["ETH/USD"])
        self.assertEqual(trips[0].name, "A1 price divergence")
        self.b.cfg.max_divergent_pairs = 1
        self.quotes["ETH/USD"] = quote(95.0)
        allowed, _, trips = self.screen([trade("SOL/USD", BUY)])
        self.assertEqual(allowed, [])
        self.assertEqual(trips[-1].action, "cycle skipped")

    def test_a2_frozen_quotes_while_binance_moves(self):
        self.bars["BTC/USD"] = bar(100.0, move=0.01)
        for _ in range(3):
            self.b.sample({"BTC/USD": quote(100.0)})
        allowed, _, trips = self.screen([trade("BTC/USD", SELL)])
        self.assertEqual(allowed, [])
        self.assertEqual(trips[0].name, "A2 stale quote")

    def test_a3_missing_bars_stop_entries_not_exits(self):
        self.bars["ETH/USD"] = bar(100.0, ts=NOW - 4 * HOUR_MS)  # newest bar 3 hours old
        trades = [trade("BTC/USD", BUY), trade("BTC/USD", SELL), trade("ETH/USD", BUY)]
        allowed = self.screen(trades, failed={"BTC/USD"})[0]
        self.assertEqual([(t.pair, t.side) for t in allowed], [("BTC/USD", SELL)])

    def test_a4_wide_spread_is_limit_only_and_blocks_shorts(self):
        self.quotes["SOL/USD"] = quote(100.0, spread=0.01)
        allowed, limit_only, _ = self.screen([trade("SOL/USD", BUY), trade("SOL/USD", SHORT)])
        self.assertEqual(limit_only, {"SOL/USD"})
        self.assertEqual([t.side for t in allowed], [BUY])

    def test_a7_refuses_oversized_entries_and_caps_orders_exits_first(self):
        allowed = self.screen([trade("BTC/USD", BUY, 50000), trade("ETH/USD", SELL, 60000)])[0]
        self.assertEqual([t.side for t in allowed], [SELL])
        self.b.cfg.max_orders = 2
        allowed = self.screen([trade("BTC/USD", BUY), trade("ETH/USD", BUY), trade("SOL/USD", SELL)])[0]
        self.assertEqual([t.side for t in allowed], [SELL, BUY])

    def test_a8_fee_budget_allows_exits_only(self):
        self.b.fees.append((NOW - HOUR_MS, 400.0))               # 0.4% of equity in a day
        allowed = self.screen([trade("BTC/USD", BUY), trade("ETH/USD", SELL)])[0]
        self.assertEqual([t.side for t in allowed], [SELL])
        self.b.fees[0] = (NOW - 25 * HOUR_MS, 400.0)            # older than a day: forgotten
        self.assertEqual(len(self.screen([trade("BTC/USD", BUY)])[0]), 1)

    def test_a9_holdings_that_moved_without_a_trade(self):
        self.b.remember_holdings({"BTC/USD": 100.0})
        allowed, _, trips = self.screen([trade("ETH/USD", BUY)], quantities={"BTC/USD": 70.0})
        self.assertEqual(allowed, [])
        self.assertEqual(trips[0].name, "A9 state mismatch")
        self.assertEqual(len(self.screen([trade("ETH/USD", BUY)], quantities={"BTC/USD": 100.0})[0]), 1)

    def test_a6_repeated_slippage_makes_the_pair_limit_only(self):
        for i in range(3):
            r = OrderResult(trade("BTC/USD", BUY), "MARKET", 1, status="FILLED", filled=1.0, avg_price=101.0)
            trips = self.b.after([r], self.quotes, NOW + i)
        self.assertEqual(trips[0].action, "limit orders only for 24 hours")
        self.assertEqual(self.screen([])[1], {"BTC/USD"})


class RejectingExchange(FakeExchange):
    def place_order(self, *args, **kwargs):
        raise RoostooError("order rejected")


class OrderFailureTest(unittest.TestCase):
    def test_a5_stops_the_cycle_after_three_rejections_in_a_row(self):
        prices = {"BTC/USD": 100.0, "ETH/USD": 50.0, "SOL/USD": 20.0, "XRP/USD": 1.0}
        exchange = RejectingExchange(prices)
        cfg = Config().execution
        cfg.use_limit_orders = False
        trips = []
        ex = Executor(exchange, parse_rules(exchange.exchange_info()), cfg, lambda order: None,
                      sleep=lambda s: None)
        ex.failure_limits = (3, 0.3)
        ex.on_trip = trips.append
        results = ex.execute([trade(p, BUY, 100.0) for p in prices], exchange.ticker())
        self.assertEqual(len(results), 3)                        # the fourth is never sent
        self.assertEqual(trips[0].name, "A5 order failures")

    def test_limit_only_pairs_get_no_market_fallback(self):
        exchange = FakeExchange({"BTC/USD": 100.0}, fill_limits=False)
        cfg = Config().execution
        cfg.limit_timeout_sec = 0
        ex = Executor(exchange, parse_rules(exchange.exchange_info()), cfg, lambda order: None,
                      sleep=lambda s: None)
        ex.execute([trade("BTC/USD", BUY, 1000.0)], exchange.ticker(), limit_only={"BTC/USD"})
        self.assertEqual({o["Type"] for o in exchange.placed}, {"LIMIT"})


class LiveFaultTest(unittest.TestCase):
    """Whole cycles against the fake exchange with the breakers on."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        series = {"BTCUSDT": zigzag(NOW, 50000.0, 0.0005), "ETHUSDT": zigzag(NOW, 2000.0, 0.0004),
                  "PAXGUSDT": zigzag(NOW, 3000.0, -0.0012)}
        self.binance = FakeBinance(series)
        self.exchange = FakeExchange({p: series[p.split("/")[0] + "USDT"][-1].close
                                      for p in ("BTC/USD", "ETH/USD", "PAXG/USD")})

    def tearDown(self):
        shutil.rmtree(self.dir)

    def bot(self):
        cfg = Config()
        cfg.strategy.universe = ["BTC/USD", "ETH/USD", "PAXG/USD"]
        cfg.execution.poll_interval_sec = 1
        cfg.breakers.enabled = True
        bot = LiveBot(cfg, self.dir, client=self.exchange, binance=self.binance, sleep=lambda s: None)
        bot.start()
        return bot

    def trips(self):
        path = os.path.join(self.dir, "journal", "breakers.csv")
        if not os.path.exists(path):
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def test_a_quiet_cycle_trades_and_trips_nothing(self):
        self.bot().cycle()
        self.assertTrue(self.exchange.placed)
        self.assertEqual(self.trips(), [])

    def test_divergent_price_gets_no_orders(self):
        self.exchange.prices["BTC/USD"] *= 1.05
        self.bot().cycle()
        self.assertNotIn("BTC/USD", {o["Pair"] for o in self.exchange.placed})
        self.assertIn("A1 price divergence", {t["name"] for t in self.trips()})

    def test_failed_candle_fetch_buys_nothing(self):
        bot = self.bot()
        self.binance.fail = True
        bot.cycle()
        self.assertFalse([o for o in self.exchange.placed if o["Side"] == "BUY"])
        self.assertIn("A3 missing bars", {t["name"] for t in self.trips()})

    def test_wallet_that_changed_between_cycles(self):
        bot = self.bot()
        bot.cycle()
        self.exchange.wallet["BTC"]["Free"] *= 0.5               # coins gone without a trade
        self.exchange.now += HOUR_MS
        bot.cycle()
        self.assertIn("A9 state mismatch", {t["name"] for t in self.trips()})


if __name__ == "__main__":
    unittest.main()
