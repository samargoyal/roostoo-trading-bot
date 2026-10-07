"""The long-short book weighted by each coin's efficiency ratio (rounds 94-97, off by default;
config/comp.json turns it on since 7 October 2026)."""
import json
import unittest

from bot.config import load_config
from bot.strategy import StrategyState
from tests.test_long_short import DAY, make, sig

def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CLEAN_UP, CLEAN_DOWN, CHOPPY = [0.01] * 4, [-0.01] * 4, [0.01, -0.01, 0.01, -0.01]


def book(returns, **overrides):
    signals = {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01), "SOL/USD": sig(False, 0.02),
               "DOGE/USD": sig(False, 0.02), "PAXG/USD": sig(True, 0.002)}
    strategy = make(signals, **overrides)
    for pair, r in returns.items():
        strategy.indicators[pair].returns.extend(r)
    return strategy.decide(DAY, 1.0, {}, StrategyState()).targets


class EfficiencyRatioTest(unittest.TestCase):
    returns = {"BTC/USD": CLEAN_UP, "ETH/USD": CHOPPY, "SOL/USD": CLEAN_DOWN, "DOGE/USD": CHOPPY}

    def test_off_by_default(self):
        t = book(self.returns)
        self.assertAlmostEqual(t["BTC/USD"], 0.5)
        self.assertAlmostEqual(t["ETH/USD"], 0.25)

    def test_clean_trends_take_the_choppy_ones_share(self):
        # efficiency: BTC and SOL 1, ETH and DOGE 0, mean 0.5; x2 and x0, the gross kept:
        # 1/vol 200 and 50 doubled to 400 and 100, re-scaled to the old 400: 320 and 80
        t = book(self.returns, ls_er_hours=4)
        self.assertAlmostEqual(t["BTC/USD"], 0.8)
        self.assertAlmostEqual(t["SOL/USD"], -0.2)
        self.assertEqual((t["ETH/USD"], t["DOGE/USD"]), (0.0, 0.0))
        self.assertAlmostEqual(sum(abs(x) for x in t.values()), 1.0)

    def test_a_coins_tilt_is_capped(self):
        # one clean coin among three choppy ones: efficiency 1 against a mean of 0.25 is x4, capped at 3
        returns = {"BTC/USD": CHOPPY, "ETH/USD": CLEAN_UP, "SOL/USD": CHOPPY, "DOGE/USD": CHOPPY}
        t = book(returns, ls_er_hours=4, ls_sizing_cap=3.0)
        self.assertAlmostEqual(t["ETH/USD"], 1.0)                # alone, it fills the gross whatever the cap
        self.assertEqual(t["BTC/USD"], 0.0)

    def test_the_competition_config_is_the_previous_one_plus_the_ratio(self):
        before, live = (read(p)["strategy"] for p in ("config/comp_k2_3.json", "config/comp.json"))
        self.assertEqual(dict(before, ls_er_hours=720, rotation_weight=0.0, ls_absorb_rotation=1.0,
                              ls_trend=[24, 72]), live)
        self.assertEqual(dict(live, ls_trend=[240, 960], ls_short_trend=[24, 72]),
                         read("config/comp_book_fast_shorts.json")["strategy"])
        self.assertEqual(dict(live, ls_trend=[240, 960], ls_short_trend=[24, 72], rotation_weight=0.55),
                         read("config/comp_b1_fast.json")["strategy"])
        self.assertEqual(dict(before, ls_er_hours=720, rotation_weight=0.55, ls_absorb_rotation=1.0),
                         read("config/comp_b1.json")["strategy"])
        self.assertEqual(dict(before, ls_er_hours=720, rotation_weight=0.0), read("config/comp_book.json")["strategy"])
        self.assertEqual(dict(before, ls_er_hours=720, rotation_weight=0.55), read("config/comp_er_55.json")["strategy"])
        self.assertEqual(dict(before, ls_er_hours=720), read("config/comp_er_75.json")["strategy"])
        self.assertEqual(load_config("config/comp.json").strategy.ls_er_hours, 720)
        self.assertEqual(load_config("config/comp_k2_3.json").strategy.ls_er_hours, 0)


if __name__ == "__main__":
    unittest.main()
