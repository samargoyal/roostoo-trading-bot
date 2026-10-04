import csv
import json
import os
import shutil
import tempfile
import unittest

from bot.config import Config
from bot.live import LiveBot, PriceSampler, code_version, config_path, portfolio_value
from bot.market_data import HOUR_MS
from tests.fakes import NOW, FakeBinance, FakeExchange, zigzag

UNIVERSE = ["BTC/USD", "ETH/USD", "PAXG/USD"]


def make_config() -> Config:
    cfg = Config()
    cfg.strategy.universe = list(UNIVERSE)
    cfg.execution.poll_interval_sec = 1
    return cfg


def market():
    """BTC and ETH trending up (risk-on, eligible), PAXG drifting down (core only)."""
    series = {
        "BTCUSDT": zigzag(NOW, 50000.0, 0.0005),
        "ETHUSDT": zigzag(NOW, 2000.0, 0.0004),
        "PAXGUSDT": zigzag(NOW, 3000.0, -0.0012),
    }
    prices = {"BTC/USD": series["BTCUSDT"][-1].close, "ETH/USD": series["ETHUSDT"][-1].close,
              "PAXG/USD": series["PAXGUSDT"][-1].close}
    return series, prices


class LiveBotTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        series, prices = market()
        self.binance = FakeBinance(series)
        self.exchange = FakeExchange(prices)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def bot(self, dry_run=False):
        bot = LiveBot(make_config(), self.dir, dry_run=dry_run, client=self.exchange,
                      binance=self.binance, sleep=lambda s: None)
        bot.start()
        return bot

    def rows(self, name):
        with open(os.path.join(self.dir, "journal", name), newline="") as f:
            return list(csv.DictReader(f))

    def test_dry_run_records_intended_orders_but_trades_nothing(self):
        self.bot(dry_run=True).cycle()
        self.assertEqual(self.exchange.placed, [])
        orders = self.rows("orders.csv")
        self.assertEqual({o["pair"] for o in orders}, set(UNIVERSE))
        self.assertTrue(all(o["status"] == "DRY_RUN" for o in orders))
        self.assertEqual(len(self.rows("equity.csv")), 1)

    def test_cycle_trades_towards_targets_and_saves_state(self):
        bot = self.bot()
        bot.cycle()
        held = {c for c, b in self.exchange.wallet.items() if c != "USD" and b["Free"] > 0}
        self.assertEqual(held, {"BTC", "ETH", "PAXG"})
        self.assertEqual(set(bot.state.positions), {"BTC/USD", "ETH/USD"})  # PAXG is only the core
        self.assertEqual(bot.state.last_fill_ts, NOW)
        with open(os.path.join(self.dir, "state.json")) as f:
            self.assertEqual(set(json.load(f)["positions"]), {"BTC/USD", "ETH/USD"})
        with open(os.path.join(self.dir, "journal", "decisions.jsonl")) as f:
            decision = json.loads(f.readline())
        self.assertTrue(decision["risk_on"])
        self.assertEqual(decision["reasons"]["BTC/USD"], "entry")

    def test_restart_after_a_crash_cancels_leftovers_and_keeps_positions(self):
        self.bot().cycle()
        with open(os.path.join(self.dir, "state.json")) as f:
            entry_ts = json.load(f)["positions"]["BTC/USD"]["entry_ts"]
        # A crash left a resting order behind.
        self.exchange.fill_limits = False
        self.exchange.place_order("ETH/USD", "SELL", "0.1", "LIMIT", "99999")
        placed_before = len(self.exchange.placed)

        self.exchange.now += HOUR_MS
        restarted = self.bot()
        self.assertEqual(set(restarted.state.positions), {"BTC/USD", "ETH/USD"})
        restarted.cycle()
        self.assertEqual(self.exchange.pending_count(), 0)
        self.assertEqual(restarted.state.positions["BTC/USD"].entry_ts, entry_ts)
        self.assertEqual(len(self.exchange.placed), placed_before)  # already on target: no trades

    def test_binance_outage_falls_back_to_cached_and_roostoo_bars(self):
        bot = self.bot()
        bot.cycle()
        self.binance.fail = True
        bot.sampler.add(self.exchange.ticker(), self.exchange.now + 5 * 60 * 1000)
        self.exchange.now += HOUR_MS
        bot.cycle()  # must not raise
        self.assertEqual(len(self.rows("equity.csv")), 2)


class HelpersTest(unittest.TestCase):
    def test_portfolio_value_counts_locked_funds(self):
        wallet = {"USD": {"Free": 100.0, "Lock": 50.0}, "BTC": {"Free": 0.5, "Lock": 0.5},
                  "ETH": {"Free": 0.0, "Lock": 0.0}}
        quotes = {"BTC/USD": {"LastPrice": 100.0}}
        equity, cash, values = portfolio_value(wallet, quotes)
        self.assertEqual((equity, cash, values), (250.0, 150.0, {"BTC/USD": 100.0}))

    def test_price_sampler_builds_hourly_bars(self):
        sampler = PriceSampler()
        hour = 10 * HOUR_MS
        for minute, price in [(0, 10.0), (20, 12.0), (40, 9.0), (59, 11.0)]:
            sampler.add({"BTC/USD": {"LastPrice": price}}, hour + minute * 60000)
        self.assertEqual(sampler.closed_bars("BTC/USD", None, hour + 30 * 60000), [])
        (bar,) = sampler.closed_bars("BTC/USD", None, hour + HOUR_MS)
        self.assertEqual((bar.open, bar.high, bar.low, bar.close), (10.0, 12.0, 9.0, 11.0))


class CodeVersionTest(unittest.TestCase):
    def test_names_the_commit_or_says_unknown(self):
        version = code_version()
        self.assertTrue(version == "unknown" or len(version.split()[0]) >= 7, version)
        self.assertEqual(code_version(tempfile.gettempdir()), "unknown")


class ConfigPathTest(unittest.TestCase):
    def test_account_file_is_used_when_present_unless_one_is_given(self):
        directory = tempfile.mkdtemp()
        try:
            self.assertIsNone(config_path("comp", directory=directory))
            path = os.path.join(directory, "comp.json")
            with open(path, "w") as f:
                f.write("{}")
            self.assertEqual(config_path("comp", directory=directory), path)
            self.assertIsNone(config_path("test", directory=directory))
            self.assertEqual(config_path("comp", "other.json", directory=directory), "other.json")
        finally:
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
