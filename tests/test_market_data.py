import unittest

from bot.market_data import HOUR_MS, MAX_KLINES, Bar, BinanceClient, binance_symbol


class FakePagedBinance(BinanceClient):
    """Serves `hours` candles ending with one still open, a page at a time like Binance."""

    def __init__(self, hours):
        super().__init__()
        self.all = [Bar(i * HOUR_MS, 1, 1, 1, 1, 1) for i in range(hours)]
        self.requests = 0

    def klines(self, symbol, start_ms=None, end_ms=None, limit=MAX_KLINES):
        self.requests += 1
        bars = [b for b in self.all if end_ms is None or b.ts <= end_ms]
        return bars[-limit:]


class RecentClosedTest(unittest.TestCase):
    def test_fetches_an_extra_page_when_the_open_candle_uses_a_slot(self):
        client = FakePagedBinance(3000)
        now = 2999 * HOUR_MS + 1  # candle 2999 is still open
        bars = client.recent_closed("BTCUSDT", 1000, now)
        self.assertEqual(len(bars), 1000)
        self.assertEqual(bars[-1].ts, 2998 * HOUR_MS)
        self.assertEqual(bars[0].ts, 1999 * HOUR_MS)
        self.assertEqual(client.requests, 2)

    def test_recent_listing_returns_what_exists(self):
        client = FakePagedBinance(500)
        bars = client.recent_closed("NEWUSDT", 1000, 499 * HOUR_MS + 1)
        self.assertEqual(len(bars), 499)

    def test_symbol_mapping(self):
        self.assertEqual(binance_symbol("BTC/USD"), "BTCUSDT")
        self.assertEqual(binance_symbol("1000CHEEMS/USD"), "1000CHEEMSUSDT")


if __name__ == "__main__":
    unittest.main()
