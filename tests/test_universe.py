import unittest

from bot.config import UniverseConfig
from bot.market_data import HOUR_MS, Bar
from bot.universe import candidate_pairs, rank_by_volume, select_universe

DAY = 24 * HOUR_MS
AT = 2000 * HOUR_MS


def history(first_ts, volume, price=1.0, end=AT):
    return [Bar(ts, price, price, price, price, volume) for ts in range(first_ts, end, HOUR_MS)]


class CandidateTest(unittest.TestCase):
    def test_filters_asset_type_spread_and_tradability(self):
        info = {"TradePairs": {
            "BTC/USD": {"AssetType": "crypto", "CanTrade": True},
            "PEPE/USD": {"AssetType": "crypto", "CanTrade": True},
            "NVDAB/USD": {"AssetType": "stock", "CanTrade": True},
            "OLD/USD": {"AssetType": "crypto", "CanTrade": False},
        }}
        quotes = {
            "BTC/USD": {"MaxBid": 100.0, "MinAsk": 100.01},
            "PEPE/USD": {"MaxBid": 0.0000099, "MinAsk": 0.0000100},  # 1% spread
            "NVDAB/USD": {"MaxBid": 100.0, "MinAsk": 100.01},
            "OLD/USD": {"MaxBid": 1.0, "MinAsk": 1.0},
        }
        self.assertEqual(list(candidate_pairs(info, quotes, UniverseConfig())), ["BTC/USD"])


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.cfg = UniverseConfig(size=2, volume_days=30, min_history_bars=1000)

    def test_ranks_by_recent_usd_volume(self):
        bars = {"A/USD": history(0, 5.0), "B/USD": history(0, 1.0, price=10.0), "C/USD": history(0, 2.0)}
        self.assertEqual([p for p, _ in rank_by_volume(bars, AT, self.cfg)], ["B/USD", "A/USD", "C/USD"])

    def test_only_volume_before_the_selection_time_counts(self):
        bars = {"A/USD": history(0, 1.0, end=AT + 100 * DAY), "B/USD": history(0, 2.0)}
        late = history(AT, 1000.0, end=AT + 10 * HOUR_MS)
        bars["A/USD"] = history(0, 1.0) + late
        self.assertEqual(rank_by_volume(bars, AT, self.cfg)[0][0], "B/USD")

    def test_recent_listings_are_excluded(self):
        listed_late = AT - 999 * HOUR_MS
        bars = {"OLD/USD": history(0, 1.0), "NEW/USD": history(listed_late, 100.0)}
        self.assertEqual([p for p, _ in rank_by_volume(bars, AT, self.cfg)], ["OLD/USD"])

    def test_defensive_pair_is_added_without_taking_a_slot(self):
        bars = {"A/USD": history(0, 3.0), "B/USD": history(0, 2.0), "C/USD": history(0, 1.0),
                "PAXG/USD": history(0, 9.0)}
        self.assertEqual(select_universe(bars, AT, self.cfg, "PAXG/USD"), ["A/USD", "B/USD", "PAXG/USD"])


if __name__ == "__main__":
    unittest.main()
