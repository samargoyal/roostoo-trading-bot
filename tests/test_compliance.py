import csv
import json
import os
import shutil
import tempfile
import time
import unittest

from bot import compliance
from bot.config import load_config
from bot.live import LiveBot
from tests.fakes import NOW, FakeBinance, FakeExchange
from tests.test_live import make_config, market

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ComplianceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        series, prices = market()
        self.exchange = FakeExchange(prices)
        self.exchange.now = int(time.time() * 1000)                  # orders are stamped with the real clock
        bot = LiveBot(make_config(), self.dir, client=self.exchange, binance=FakeBinance(series), sleep=lambda s: None)
        bot.start()
        bot.cycle()                                                   # a real journal: decisions and orders
        self.write_log("HEAD", load_config(os.path.join(REPO, "config", "comp.json")).to_dict())

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write_log(self, commit, config, path="config/comp.json"):
        os.makedirs(os.path.join(self.dir, "logs"), exist_ok=True)
        with open(os.path.join(self.dir, "logs", "bot.log"), "w", encoding="utf-8") as f:
            f.write("2026-10-04T12:58:01.000Z INFO bot.live: starting: account=comp dry_run=False once=False "
                    "commit=%s config=%s\n" % (commit, path))
            f.write("2026-10-04T12:58:01.000Z INFO bot.live: config: %s\n" % json.dumps(config, sort_keys=True))

    def build(self, history=None):
        return compliance.build(self.dir, REPO, history)

    def test_the_bots_own_records_pass(self):
        report = self.build()
        self.assertTrue(report["passed"], compliance.render(report, "test"))
        self.assertGreater(report["orders"]["matched"], 0)
        self.assertEqual(len(report["activity"]["traded"]), 1)
        self.assertIn("Overall: **PASS**", compliance.render(report, "test"))

    def test_a_run_with_uncommitted_changes_is_flagged(self):
        self.write_log("HEAD (with local changes)", load_config(os.path.join(REPO, "config", "comp.json")).to_dict())
        report = self.build()
        self.assertFalse(report["passed"])
        self.assertIn("uncommitted", report["starts"][0]["problems"][0])

    def test_a_config_that_differs_from_its_commit_is_flagged(self):
        config = load_config(os.path.join(REPO, "config", "comp.json")).to_dict()
        config["strategy"]["rotation_weight"] = 0.9
        self.write_log("HEAD", config)
        report = self.build()
        self.assertFalse(report["passed"])
        self.assertIn("rotation_weight", report["starts"][0]["problems"][0])

    def test_an_order_no_decision_explains_is_flagged(self):
        path = os.path.join(self.dir, "journal", "orders.csv")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        extra = dict(rows[0], pair="ETH/USD", side="BUY", order_id="999", target_weight="0.9", current_weight="0",
                     reason="entry")
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(rows[0])).writerow(extra)
        report = self.build()
        self.assertFalse(report["passed"])
        self.assertEqual([o["order_id"] for o in report["orders"]["unmatched"]], ["999"])

    def test_an_order_on_the_exchange_the_bot_did_not_record_is_flagged(self):
        history = [{"OrderID": o["OrderID"], "Side": o["Side"], "Pair": o["Pair"]} for o in self.exchange.placed]
        self.assertTrue(self.build(history)["passed"])
        history.append({"OrderID": 4242, "Side": "BUY", "Pair": "BTC/USD", "CreateTimestamp": NOW})
        report = self.build(history)
        self.assertFalse(report["passed"])
        self.assertEqual([h["OrderID"] for h in report["exchange"]], [4242])


if __name__ == "__main__":
    unittest.main()
