"""The long-short trend book and the rotation's bear-market basket (research H50, both off by
default)."""
import unittest

from bot.config import StrategyConfig
from bot.indicators import IndicatorSet, Signal
from bot.market_data import HOUR_MS
from bot.strategy import (EXIT_SHORT, EXIT_SHORT_STOP, EXIT_STOP, HOLD_HALTED, LS_LONG, LS_SHORT, PositionInfo,
                          ShortInfo, Strategy, StrategyState)

UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "PAXG/USD"]
DAY = 24 * HOUR_MS


def sig(up=True, vol=0.01, btc_filter_on=True, rot=0.1):
    """A coin whose own long-short trend (and 168h/672h trend) is up or down."""
    return Signal(0, 110.0, 105.0, 100.0, 90.0, 1.0, 50.0, 0.0, 0.0, vol, return_rotation=rot,
                  ema_trend_fast=101.0 if btc_filter_on else 99.0, ema_trend_slow=100.0,
                  ls_fast=101.0 if up else 99.0, ls_slow=100.0)


def make(signals, **overrides):
    cfg = StrategyConfig(universe=list(UNIVERSE), **dict({"rotation_weight": 0.0, "book_mode": "long_short"},
                                                         **overrides))
    strategy = Strategy(cfg)
    strategy.signals = lambda: signals
    return strategy


class LongShortBookTest(unittest.TestCase):
    def market(self, btc_filter_on=True):
        f = btc_filter_on
        return {"BTC/USD": sig(True, 0.005, f), "ETH/USD": sig(True, 0.01, f), "SOL/USD": sig(False, 0.02, f),
                "DOGE/USD": sig(False, 0.02, f), "PAXG/USD": sig(True, 0.002, f)}

    def test_long_uptrends_short_downtrends_by_inverse_volatility(self):
        d = make(self.market()).decide(DAY, 1.0, {}, StrategyState())
        # 1/vol: BTC 200, ETH 100, SOL 50, DOGE 50; gross 400
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.5)
        self.assertAlmostEqual(d.targets["ETH/USD"], 0.25)
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.125)
        self.assertAlmostEqual(d.targets["DOGE/USD"], -0.125)
        self.assertEqual(d.targets["PAXG/USD"], 0.0)              # the defensive pair stays out
        self.assertEqual((d.reasons["BTC/USD"], d.reasons["SOL/USD"]), (LS_LONG, LS_SHORT))
        self.assertAlmostEqual(sum(abs(t) for t in d.targets.values()), 1.0)

    def test_scaled_to_the_books_share_beside_the_rotation(self):
        d = make(self.market(), rotation_weight=0.7).decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["DOGE/USD"], 0.3 * -0.125)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.3 * 0.5 + 0.7 * d.rotation.get("BTC/USD", 0.0))

    def test_the_rotations_idle_share_joins_the_book_while_its_filter_is_off(self):
        bear = self.market(btc_filter_on=False)
        d = make(bear, rotation_weight=0.7, ls_absorb_rotation=1.0).decide(DAY, 1.0, {}, StrategyState())
        plain = make(bear, rotation_weight=0.7).decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.125)          # the whole account
        self.assertAlmostEqual(plain.targets["SOL/USD"], 0.3 * -0.125)
        self.assertEqual(d.rotation, {})

    def test_absorbs_only_below_the_long_average_when_asked(self):
        bear = {p: x._replace(sma_long=100.0) for p, x in self.market(btc_filter_on=False).items()}  # close 110
        d = make(bear, rotation_weight=0.7, ls_absorb_rotation=1.0, ls_absorb_sma_hours=4800).decide(
            DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["SOL/USD"], 0.3 * -0.125)    # above its average: no absorbing
        bear = {p: x._replace(sma_long=120.0) for p, x in bear.items()}
        d = make(bear, rotation_weight=0.7, ls_absorb_rotation=1.0, ls_absorb_sma_hours=4800).decide(
            DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.125)

    def test_regime_aligned_drops_shorts_in_bull_and_longs_in_bear_markets(self):
        bull = make(self.market(True), ls_regime_aligned=True).decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual((bull.targets["SOL/USD"], bull.targets["DOGE/USD"]), (0.0, 0.0))
        self.assertAlmostEqual(bull.targets["BTC/USD"], 2 / 3)
        bear = make(self.market(False), ls_regime_aligned=True).decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual((bear.targets["BTC/USD"], bear.targets["ETH/USD"]), (0.0, 0.0))
        self.assertAlmostEqual(bear.targets["SOL/USD"], -0.5)

    def test_neutral_zone_and_strength_sizing(self):
        signals = self.market()
        signals["ETH/USD"] = signals["ETH/USD"]._replace(ls_fast=100.5)   # a 0.5% gap: inside a 1% zone
        d = make(signals, ls_band=0.01).decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.targets["ETH/USD"], 0.0)
        d = make(self.market(), ls_full_gap=0.02).decide(DAY, 1.0, {}, StrategyState())
        # every gap is 1%, half of the full 2%: half the book is invested
        self.assertAlmostEqual(sum(abs(t) for t in d.targets.values()), 0.5)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.25)

    def test_shorts_at_a_fraction_of_their_weight(self):
        d = make(self.market(), ls_short_scale=0.5).decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.0625)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.5)

    def test_btc_alone(self):
        d = make(self.market(), ls_pairs="btc").decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["BTC/USD"], 1.0)
        self.assertEqual(sum(abs(t) for p, t in d.targets.items() if p != "BTC/USD"), 0.0)

    def test_a_halted_coin_keeps_its_weight_and_uses_up_budget(self):
        d = make(self.market()).decide(DAY, 1.0, {"SOL/USD": -0.2}, StrategyState(), frozenset({"SOL/USD"}))
        self.assertEqual(d.targets["SOL/USD"], -0.2)
        self.assertEqual(d.reasons["SOL/USD"], HOLD_HALTED)
        self.assertAlmostEqual(sum(abs(t) for t in d.targets.values()), 1.0)


class SqueezeDefenceTest(unittest.TestCase):
    """Round 53: the long-short book's shorts, limited once a squeeze starts."""

    def market(self):
        return {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01), "SOL/USD": sig(False, 0.02),
                "DOGE/USD": sig(False, 0.02), "PAXG/USD": sig(True, 0.002)}

    def test_trailing_stop_covers_and_cools_down(self):
        state = StrategyState(shorts={"SOL/USD": ShortInfo(entry_ts=0, lowest_close=100.0)})
        strategy = make(self.market(), short_stop_atr=5.0)          # close 110 > 100 + 5 x ATR 1
        d = strategy.decide(DAY, 1.0, {"SOL/USD": -0.1}, state)
        self.assertEqual((d.targets["SOL/USD"], d.reasons["SOL/USD"]), (0.0, EXIT_SHORT_STOP))
        self.assertLess(d.targets["DOGE/USD"], 0.0)                 # not held, so no stop to hit
        # its slot stays in cash: the others keep their own share of all four slots
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.5)
        state.short_cooldown_until["SOL/USD"] = DAY + 10 * HOUR_MS
        d = make(self.market(), short_stop_atr=50.0).decide(DAY + HOUR_MS, 1.0, {}, state)
        self.assertEqual(d.targets["SOL/USD"], 0.0)                 # cooling down

    def test_trailing_stop_on_longs(self):
        state = StrategyState(positions={"BTC/USD": PositionInfo(0, 130.0)})
        d = make(self.market(), ls_long_stop_atr=8.0, short_stop_atr=50.0).decide(DAY, 1.0, {"BTC/USD": 0.5}, state)
        self.assertEqual((d.targets["BTC/USD"], d.reasons["BTC/USD"]), (0.0, EXIT_STOP))   # 110 < 130 - 8
        self.assertGreater(d.targets["ETH/USD"], 0.0)

    def test_shorts_on_a_faster_trend_flat_while_the_trends_disagree(self):
        signals = self.market()
        signals["SOL/USD"] = signals["SOL/USD"]._replace(ls_short_fast=99.0, ls_short_slow=100.0)
        signals["DOGE/USD"] = signals["DOGE/USD"]._replace(ls_short_fast=101.0, ls_short_slow=100.0)
        signals["BTC/USD"] = signals["BTC/USD"]._replace(ls_short_fast=99.0, ls_short_slow=100.0)
        signals["ETH/USD"] = signals["ETH/USD"]._replace(ls_short_fast=101.0, ls_short_slow=100.0)
        d = make(signals, ls_short_trend=[168, 672]).decide(DAY, 1.0, {}, StrategyState())
        self.assertLess(d.targets["SOL/USD"], 0.0)                   # both trends down
        self.assertEqual(d.targets["DOGE/USD"], 0.0)                 # slow down, fast up: flat
        self.assertEqual(d.targets["BTC/USD"], 0.0)                  # slow up, fast down: flat
        self.assertGreater(d.targets["ETH/USD"], 0.0)                # both up

    def test_turtle_entries_and_exits(self):
        strategy = make(self.market(), short_entry_channel=480, short_exit_channel=240)
        lows = {"SOL/USD": 120.0, "DOGE/USD": 100.0}                # SOL closes below its 20-day low
        highs = {"SOL/USD": 130.0, "DOGE/USD": 105.0}               # a held DOGE closes above its 10-day high
        strategy._channel = lambda p, h, high: highs[p] if high else lows[p]
        state = StrategyState(shorts={"DOGE/USD": ShortInfo(entry_ts=0, lowest_close=100.0)})
        d = strategy.decide(DAY, 1.0, {"DOGE/USD": -0.1}, state)
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.125)
        self.assertEqual((d.targets["DOGE/USD"], d.reasons["DOGE/USD"]), (0.0, EXIT_SHORT))

    def test_shorts_only_in_the_most_traded(self):
        strategy = make(self.market(), short_top_volume=3)
        volume = {"BTC/USD": 9, "ETH/USD": 8, "SOL/USD": 7, "DOGE/USD": 1, "PAXG/USD": 0}
        for p, v in volume.items():
            strategy.indicators[p].dollar_sum = v
        d = strategy.decide(DAY, 1.0, {}, StrategyState())
        self.assertLess(d.targets["SOL/USD"], 0.0)
        self.assertEqual(d.targets["DOGE/USD"], 0.0)
        self.assertAlmostEqual(sum(abs(t) for t in d.targets.values()), 1.0)   # the others take its share

    def test_shorts_shrink_when_the_last_month_was_wilder(self):
        strategy = make(self.market(), short_vol_ratio=[2, 4])
        strategy.indicators["BTC/USD"].returns.extend([0.01, -0.01, 0.04, -0.04])
        d = strategy.decide(DAY, 1.0, {}, StrategyState())
        plain = make(self.market()).decide(DAY, 1.0, {}, StrategyState())
        base = (sum(x * x for x in [0.01, -0.01, 0.04, -0.04]) / 3) ** 0.5
        recent = (0.08 ** 2 / 2 / 1) ** 0.5
        self.assertAlmostEqual(d.targets["SOL/USD"], plain.targets["SOL/USD"] * base / recent)
        self.assertAlmostEqual(d.targets["BTC/USD"], plain.targets["BTC/USD"])


class EnsembleTest(unittest.TestCase):
    def test_positions_follow_the_vote_of_several_trend_speeds(self):
        signals = {"BTC/USD": sig(True, 0.005)._replace(ls_vote=1.0), "ETH/USD": sig(True, 0.01)._replace(ls_vote=1 / 3),
                   "SOL/USD": sig(False, 0.02)._replace(ls_vote=-1.0), "DOGE/USD": sig(False, 0.02)._replace(ls_vote=0.0),
                   "PAXG/USD": sig(True, 0.002)}
        d = make(signals, ls_ensemble=[[168, 672], [336, 1344]], short_stop_atr=50.0).decide(DAY, 1.0, {}, StrategyState())
        # slots: 1/vol over BTC, ETH, SOL, DOGE = 400
        self.assertAlmostEqual(d.targets["BTC/USD"], 200 / 400)
        self.assertAlmostEqual(d.targets["ETH/USD"], 100 / 3 / 400)
        self.assertAlmostEqual(d.targets["SOL/USD"], -50 / 400)
        self.assertEqual(d.targets["DOGE/USD"], 0.0)                 # a split vote: no position

    def test_the_vote_indicator(self):
        from bot.market_data import Bar
        ind = IndicatorSet(3, 5, 5, 3, 3, 2, 3, 3, rotation_lookback=3, trend_fast=3, trend_slow=5,
                           ls_pair=(2, 4), ls_extra=[(3, 6), (6, 3)])
        for i, close in enumerate(range(1, 30)):
            ind.update(Bar(i * HOUR_MS, close, close, close, close, 1.0))
        self.assertAlmostEqual(ind.signal().ls_vote, 1 / 3)          # rising: two pairs up, the inverted one down


class OverlayTest(unittest.TestCase):
    def test_idle_book_cash_shorts_downtrends_by_their_share_of_all_coins(self):
        down = dict(ema_fast=95.0)                                    # the defensive book will not hold them
        signals = {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01),
                   "SOL/USD": sig(False, 0.02)._replace(**down), "DOGE/USD": sig(False, 0.02)._replace(**down),
                   "PAXG/USD": sig(True, 0.002)}
        d = make(signals, book_mode="overlay").decide(DAY, 1.0, {}, StrategyState())
        idle = 1.0 - sum(t for t in d.targets.values() if t > 0)
        self.assertAlmostEqual(idle, 0.55)                            # BTC, ETH, PAXG at the 15% cap
        # SOL's slot is 50 of 400 (inverse volatility over BTC, ETH, SOL, DOGE)
        self.assertAlmostEqual(d.targets["SOL/USD"], -idle * 50 / 400)
        self.assertAlmostEqual(d.targets["DOGE/USD"], -idle * 50 / 400)
        self.assertEqual(d.reasons["SOL/USD"], LS_SHORT)


class ConfirmedBearTest(unittest.TestCase):
    """short_regime_hours: shorts only while BTC's filter is off and BTC is below its long average."""

    def market(self, btc_filter_on, below_sma):
        f = btc_filter_on
        sma = 120.0 if below_sma else 100.0                          # every close is 110
        return {p: s._replace(sma_long=sma) for p, s in {
            "BTC/USD": sig(True, 0.005, f), "ETH/USD": sig(True, 0.01, f), "SOL/USD": sig(False, 0.02, f),
            "DOGE/USD": sig(False, 0.02, f), "PAXG/USD": sig(True, 0.002, f)}.items()}

    def test_long_short_book_shorts_only_in_a_confirmed_bear_market(self):
        for filter_on, below, shorts in ((True, True, False), (False, False, False), (False, True, True)):
            d = make(self.market(filter_on, below), short_regime_hours=4800).decide(DAY, 1.0, {}, StrategyState())
            self.assertEqual(d.targets["SOL/USD"] < 0, shorts, (filter_on, below))
            self.assertGreater(d.targets["BTC/USD"], 0.0)             # longs are unaffected

    def test_no_shorts_before_the_average_is_ready(self):
        signals = {p: s._replace(sma_long=0.0) for p, s in self.market(False, True).items()}
        d = make(signals, short_regime_hours=4800).decide(DAY, 1.0, {}, StrategyState())
        self.assertEqual(d.targets["SOL/USD"], 0.0)

    def test_hybrid_book_is_defensive_until_a_confirmed_bear_market(self):
        bull = make(self.market(True, False), book_mode="hybrid", short_regime_hours=4800)
        d = bull.decide(DAY, 1.0, {}, StrategyState())
        self.assertTrue(all(t >= 0 for t in d.targets.values()))
        self.assertGreater(d.targets["PAXG/USD"], 0.0)                # the defensive book's core
        bear = make(self.market(False, True), book_mode="hybrid", short_regime_hours=4800)
        d = bear.decide(DAY, 1.0, {}, StrategyState())
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.5)            # only the downtrends, short
        self.assertAlmostEqual(d.targets["DOGE/USD"], -0.5)
        self.assertEqual((d.targets["BTC/USD"], d.targets["PAXG/USD"]), (0.0, 0.0))

    def test_the_long_average_indicator(self):
        ind = IndicatorSet(3, 5, 5, 3, 3, 2, 3, 3, rotation_lookback=3, trend_fast=3, trend_slow=5, long_sma=4)
        from bot.market_data import Bar
        for i, close in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
            ind.update(Bar(i * HOUR_MS, close, close, close, close, 1.0))
        self.assertAlmostEqual(ind.signal().sma_long, (5 + 6 + 7 + 8) / 4)


class TrendBasketTest(unittest.TestCase):
    def test_bear_market_sleeve_shorts_every_downtrending_coin_by_inverse_volatility(self):
        signals = {"BTC/USD": sig(vol=0.005, btc_filter_on=False), "ETH/USD": sig(vol=0.01, btc_filter_on=False),
                   "SOL/USD": sig(vol=0.02, btc_filter_on=False), "DOGE/USD": sig(vol=0.02, btc_filter_on=True),
                   "PAXG/USD": sig(vol=0.002, btc_filter_on=False, rot=-0.01)}
        cfg = StrategyConfig(universe=list(UNIVERSE), rotation_weight=0.7, rotation_shorts=1,
                             rotation_short_ranking="trend_basket", rotation_short_cap=0.5)
        strategy = Strategy(cfg)
        strategy.signals = lambda: signals
        d = strategy.decide(DAY, 1.0, {}, StrategyState())
        # DOGE's own trend is up, PAXG is the defensive pair; 1/vol: BTC 200, ETH 100, SOL 50
        self.assertAlmostEqual(d.rotation["BTC/USD"], -0.5)     # 0.571, capped at 0.5
        self.assertAlmostEqual(d.rotation["ETH/USD"], -100 / 350)
        self.assertAlmostEqual(d.rotation["SOL/USD"], -50 / 350)
        self.assertNotIn("DOGE/USD", d.rotation)


if __name__ == "__main__":
    unittest.main()
