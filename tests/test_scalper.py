import csv
import json
import os
import tempfile
import unittest

from bot.config import ScalperConfig
from bot.execution import parse_rules
from bot.scalper import Scalper, paused
from tests.fakes import NOW, FakeExchange

T0 = NOW - 60_000 + 20 * 60_000          # twenty minutes past an hour, outside the pause


class Feed:
    def __init__(self, mid):
        self.mid = mid

    def __call__(self):
        return {"BTC/USD": (self.mid * 0.99995, self.mid * 1.00005)}


def make(paper=True, **settings):
    cfg = ScalperConfig(pairs=["BTC/USD"], **settings)
    ex = FakeExchange({"BTC/USD": 100000.0})
    feed = Feed(100000.0)
    run_dir = tempfile.mkdtemp()
    s = Scalper(cfg, parse_rules(ex.exchange_info()), feed, run_dir, None if paper else ex, paper=paper)
    return s, ex, feed


def warm_up_and_jump(s, feed, jump=1.002):
    """Twenty-one quiet seconds, then the mid jumps: the scalper buys."""
    for i in range(21):
        s.step(T0 + i * 1000)
    feed.mid *= jump
    s.step(T0 + 21_000)
    return T0 + 21_000


def trips(s):
    with open(s.trips_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class PauseTest(unittest.TestCase):
    def test_pause_window_around_the_hour(self):
        cfg = ScalperConfig()
        hour = NOW - 60_000
        self.assertTrue(paused(hour - 2 * 60_000, cfg))      # :58
        self.assertTrue(paused(hour + 5 * 60_000, cfg))      # :05
        self.assertFalse(paused(hour + 30 * 60_000, cfg))    # :30
        self.assertFalse(paused(hour + 12 * 60_000, cfg))    # :12


class PaperTest(unittest.TestCase):
    def test_no_entry_without_a_move(self):
        s, _, feed = make()
        for i in range(40):
            s.step(T0 + i * 1000)
        self.assertIsNone(s.open)

    def test_jump_buys_and_the_target_sells(self):
        s, _, feed = make()
        t = warm_up_and_jump(s, feed)
        self.assertIsNotNone(s.open)
        feed.mid *= 1.004
        s.step(t + 5000)
        self.assertIsNone(s.open)
        row = trips(s)[0]
        self.assertEqual(row["how"], "target")
        self.assertAlmostEqual(float(row["pnl_bp"]), 25 - 15, delta=2)   # +25 bp less 0.1% in, 0.05% out

    def test_always_out_within_a_minute(self):
        s, _, feed = make()
        t = warm_up_and_jump(s, feed)
        s.step(t + 54_000)
        self.assertIsNotNone(s.open)
        s.step(t + 55_000)
        self.assertIsNone(s.open)
        self.assertEqual(trips(s)[0]["how"], "time")

    def test_stop(self):
        s, _, feed = make()
        t = warm_up_and_jump(s, feed)
        feed.mid *= 0.997
        s.step(t + 2000)
        self.assertEqual(trips(s)[0]["how"], "stop")

    def test_flat_before_the_hourly_bot_runs(self):
        s, _, feed = make()
        hour_end = NOW - 60_000 + 3600_000
        start = hour_end - 3 * 60_000 - 30_000
        for i in range(21):
            s.step(start + i * 1000)
        feed.mid *= 1.002
        s.step(start + 21_000)
        self.assertIsNotNone(s.open)
        s.step(hour_end - 3 * 60_000)
        self.assertEqual(trips(s)[0]["how"], "pause")

    def test_daily_loss_limit_stops_new_scalps(self):
        s, _, feed = make(max_daily_loss_usd=1.0)
        t = warm_up_and_jump(s, feed)
        feed.mid *= 0.997
        s.step(t + 2000)                                    # a stop loses more than 1 USD
        for i in range(120):
            feed.mid *= 1.0002
            s.step(t + 3000 + i * 1000)
        self.assertEqual(len(trips(s)), 1)


class LagGuardTest(unittest.TestCase):
    def test_no_entry_while_roostoo_lags_binance(self):
        s, _, feed = make()
        s.roostoo_quote = lambda pair: (99990.0, 100010.0)    # still at the old price
        warm_up_and_jump(s, feed)
        self.assertIsNone(s.open)

    def test_entry_once_roostoo_shows_the_move(self):
        s, _, feed = make()
        s.roostoo_quote = lambda pair: (feed.mid * 0.99995, feed.mid * 1.00005)
        warm_up_and_jump(s, feed)
        self.assertIsNotNone(s.open)


class LiveTest(unittest.TestCase):
    def test_market_buy_then_resting_target_fills(self):
        s, ex, feed = make(paper=False)
        t = warm_up_and_jump(s, feed)
        types = [(o["Side"], o["Type"]) for o in ex.placed]
        self.assertEqual(types, [("BUY", "MARKET"), ("SELL", "LIMIT")])
        feed.mid *= 1.004
        s.step(t + 3000)
        self.assertIsNone(s.open)
        self.assertAlmostEqual(ex.wallet["BTC"]["Free"] + ex.wallet["BTC"]["Lock"], 0.0, places=8)
        self.assertEqual(trips(s)[0]["how"], "target")

    def test_stop_cancels_the_target_and_sells_only_what_it_bought(self):
        s, ex, feed = make(paper=False)
        ex.fill_limits = False
        ex.wallet["BTC"] = {"Free": 0.5, "Lock": 0.0}       # the hourly bot's holding
        t = warm_up_and_jump(s, feed)
        feed.mid *= 0.995
        s.step(t + 2000)
        self.assertIsNone(s.open)
        self.assertEqual([o["Status"] for o in ex.placed], ["FILLED", "CANCELED", "FILLED"])
        self.assertAlmostEqual(ex.wallet["BTC"]["Free"], 0.5, places=8)
        self.assertEqual(trips(s)[0]["how"], "stop")

    def test_a_scalp_left_open_is_closed_on_restart(self):
        s, ex, feed = make(paper=False)
        ex.fill_limits = False
        warm_up_and_jump(s, feed)
        s2 = Scalper(s.cfg, s.rules, feed, s.run_dir, ex)
        with open(s.state_path, encoding="utf-8") as f:
            self.assertIsNotNone(json.load(f)["open"])
        s2.recover(T0 + 30_000)
        self.assertIsNone(s2.open)
        self.assertAlmostEqual(ex.wallet["BTC"]["Free"] + ex.wallet["BTC"]["Lock"], 0.0, places=8)
        self.assertEqual(trips(s2)[0]["how"], "restart")
        self.assertTrue(os.path.exists(s2.state_path))


if __name__ == "__main__":
    unittest.main()
