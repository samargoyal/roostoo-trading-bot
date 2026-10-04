"""Perpetual funding rates: the long-short book's crowding filter (research round 54)."""
import shutil
import tempfile
import unittest

from bot.config import Config, StrategyConfig
from bot.live import LiveBot
from bot.market_data import HOUR_MS, BinanceClient, BinanceError
from bot.strategy import Strategy, StrategyState
from tests.fakes import NOW, FakeBinance, FakeExchange, zigzag
from tests.test_long_short import DAY, UNIVERSE, sig


class FakeFunding:
    def __init__(self, prints):
        self.prints = prints            # {symbol: [(time, rate)]}
        self.calls = []

    def funding_rates(self, symbol, start_ms, end_ms):
        self.calls.append((symbol, start_ms, end_ms))
        if symbol not in self.prints:
            raise BinanceError("no perpetual")
        return [(t, r) for t, r in self.prints[symbol] if start_ms <= t <= end_ms]


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class FakeSession:
    def __init__(self, body):
        self.body = body
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url, params))
        return FakeResponse(self.body)


class FundingClientTest(unittest.TestCase):
    def test_parses_binance_funding_prints(self):
        session = FakeSession([{"symbol": "BTCUSDT", "fundingTime": 1700000000000, "fundingRate": "-0.00012"}])
        client = BinanceClient("https://fapi.binance.com", session=session)
        self.assertEqual(client.funding_rates("BTCUSDT", 1, 2), [(1700000000000, -0.00012)])
        url, params = session.requests[0]
        self.assertTrue(url.endswith("/fapi/v1/fundingRate"))
        self.assertEqual((params["startTime"], params["endTime"]), (1, 2))


class FundingFilterTest(unittest.TestCase):
    def test_no_short_where_funding_is_negative(self):
        signals = {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01), "SOL/USD": sig(False, 0.02),
                   "DOGE/USD": sig(False, 0.02), "PAXG/USD": sig(True, 0.002)}
        day = DAY // (24 * HOUR_MS) * (24 * HOUR_MS)
        cfg = StrategyConfig(universe=list(UNIVERSE), rotation_weight=0.0, book_mode="long_short",
                             short_exclude_external=True)
        strategy = Strategy(cfg, external_scores={day: {"SOL/USD": -0.0001, "DOGE/USD": 0.0001}})
        strategy.signals = lambda: signals
        d = strategy.decide(DAY + HOUR_MS, 1.0, {}, StrategyState())
        self.assertEqual(d.targets["SOL/USD"], 0.0)                 # crowded shorts
        self.assertLess(d.targets["DOGE/USD"], 0.0)


class CrowdedLongTest(unittest.TestCase):
    def test_no_long_where_funding_is_far_above_normal(self):
        signals = {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01), "SOL/USD": sig(False, 0.02),
                   "DOGE/USD": sig(False, 0.02), "PAXG/USD": sig(True, 0.002)}
        day = DAY // (24 * HOUR_MS) * (24 * HOUR_MS)
        cfg = StrategyConfig(universe=list(UNIVERSE), rotation_weight=0.0, book_mode="long_short",
                             long_max_external=0.0003)
        strategy = Strategy(cfg, external_scores={day: {"ETH/USD": 0.0005, "BTC/USD": 0.0001}})
        strategy.signals = lambda: signals
        d = strategy.decide(DAY + HOUR_MS, 1.0, {}, StrategyState())
        self.assertEqual(d.targets["ETH/USD"], 0.0)
        self.assertGreater(d.targets["BTC/USD"], 0.0)


class FundingRankTest(unittest.TestCase):
    def test_shorts_only_where_longs_are_crowded_and_rotation_skips_crowded_picks(self):
        signals = {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01), "SOL/USD": sig(False, 0.02),
                   "DOGE/USD": sig(False, 0.02), "PAXG/USD": sig(True, 0.002)}
        day = DAY // (24 * HOUR_MS) * (24 * HOUR_MS)
        funding = {"BTC/USD": 0.0001, "ETH/USD": 0.0004, "SOL/USD": 0.00005, "DOGE/USD": 0.0002}
        cfg = StrategyConfig(universe=list(UNIVERSE), rotation_weight=0.0, book_mode="long_short",
                             short_min_funding_rank=0.5)
        strategy = Strategy(cfg, external_scores={day: funding})
        strategy.signals = lambda: signals
        d = strategy.decide(DAY + HOUR_MS, 1.0, {}, StrategyState())
        self.assertEqual(d.targets["SOL/USD"], 0.0)                 # lowest funding: rank 0
        self.assertLess(d.targets["DOGE/USD"], 0.0)                 # rank 2 of 3
        cfg = StrategyConfig(universe=list(UNIVERSE), rotation_weight=0.7, rotation_max_external=0.0003)
        strategy = Strategy(cfg, external_scores={day: funding})
        strategy.signals = lambda: signals
        d = strategy.decide(DAY + HOUR_MS, 1.0, {}, StrategyState())
        self.assertNotIn("ETH/USD", d.rotation)


class ShortEntryTest(unittest.TestCase):
    def test_new_shorts_need_clear_funding_and_no_oversold_reading_but_open_ones_stay(self):
        from bot.strategy import ShortInfo
        signals = {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01), "SOL/USD": sig(False, 0.02),
                   "DOGE/USD": sig(False, 0.02)._replace(rsi=20.0), "PAXG/USD": sig(True, 0.002)}
        day = DAY // (24 * HOUR_MS) * (24 * HOUR_MS)
        cfg = StrategyConfig(universe=list(UNIVERSE), rotation_weight=0.0, book_mode="long_short",
                             short_exclude_external=True, short_entry_min_funding=0.00005, short_entry_rsi_min=30.0)
        strategy = Strategy(cfg, external_scores={day: {"SOL/USD": 0.00002, "DOGE/USD": 0.0001}})
        strategy.signals = lambda: signals
        d = strategy.decide(DAY + HOUR_MS, 1.0, {}, StrategyState())
        self.assertEqual(d.targets["SOL/USD"], 0.0)                  # funding too close to zero to open
        self.assertEqual(d.targets["DOGE/USD"], 0.0)                 # oversold: wait for the bounce
        held = StrategyState(shorts={"SOL/USD": ShortInfo(0, 110.0), "DOGE/USD": ShortInfo(0, 110.0)})
        d = strategy.decide(DAY + HOUR_MS, 1.0, {"SOL/USD": -0.1, "DOGE/USD": -0.1}, held)
        self.assertLess(d.targets["SOL/USD"], 0.0)                   # open shorts stay
        self.assertLess(d.targets["DOGE/USD"], 0.0)


class LiveFundingTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_fetched_once_a_day_as_the_mean_up_to_midnight(self):
        cfg = Config()
        cfg.strategy.universe = ["BTC/USD", "ETH/USD", "PAXG/USD"]
        cfg.strategy.short_exclude_external = True
        day = (NOW - HOUR_MS) // (24 * HOUR_MS) * (24 * HOUR_MS)
        eight = 8 * HOUR_MS
        funding = FakeFunding({"BTCUSDT": [(day - 9 * eight, 0.5), (day - eight, -0.001), (day, -0.003)],
                               "ETHUSDT": [(day, 0.002), (day + eight, -1.0)]})
        series = {s: zigzag(NOW, 100.0, 0.0) for s in ("BTCUSDT", "ETHUSDT", "PAXGUSDT")}
        bot = LiveBot(cfg, self.dir, client=FakeExchange({p: 100.0 for p in cfg.strategy.universe}),
                      binance=FakeBinance(series), sleep=lambda s: None, funding=funding)
        table = bot._funding_scores(NOW)
        # the 72 hours up to that day's midnight: BTC's print 9 x 8h earlier is too old,
        # ETH's after midnight not yet known; PAXG has no perpetual and is left out
        self.assertEqual(set(table), {day})
        self.assertAlmostEqual(table[day]["BTC/USD"], -0.002)
        self.assertAlmostEqual(table[day]["ETH/USD"], 0.002)
        self.assertNotIn("PAXG/USD", table[day])
        calls = len(funding.calls)
        bot._funding_scores(NOW + HOUR_MS)
        self.assertEqual(len(funding.calls), calls)                 # cached for the day

    def test_gives_up_for_the_day_when_binance_futures_is_unreachable(self):
        cfg = Config()
        cfg.strategy.short_exclude_external = True
        funding = FakeFunding({})                                    # every request fails
        bot = LiveBot(cfg, self.dir, client=FakeExchange({}), binance=FakeBinance({}), sleep=lambda s: None,
                      funding=funding)
        day = (NOW - HOUR_MS) // (24 * HOUR_MS) * (24 * HOUR_MS)
        self.assertEqual(bot._funding_scores(NOW), {day: {}})
        self.assertEqual(len(funding.calls), 3)

    def test_off_by_default(self):
        cfg = Config()
        bot = LiveBot(cfg, self.dir, client=FakeExchange({}), binance=FakeBinance({}), sleep=lambda s: None)
        self.assertEqual(bot._funding_scores(NOW), {})


if __name__ == "__main__":
    unittest.main()
