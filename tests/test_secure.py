"""Securing profits in the competition window: a research option (research/h70_secure_profits.py),
not used live."""
import math
import unittest
from collections import deque

from bot.config import StrategyConfig
from bot.indicators import Signal
from bot.market_data import HOUR_MS
from bot.strategy import Strategy, StrategyState

UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "PAXG/USD"]
DAY = 24 * HOUR_MS
START = 10 * DAY


def sig(up=True, vol=0.01):
    return Signal(0, 110.0, 105.0, 100.0, 90.0, 1.0, 50.0, 0.0, 0.0, vol, return_rotation=0.1,
                  ema_trend_fast=101.0, ema_trend_slow=100.0, ls_fast=101.0 if up else 99.0, ls_slow=100.0)


MARKET = {"BTC/USD": sig(True, 0.005), "ETH/USD": sig(True, 0.01), "SOL/USD": sig(False, 0.02),
          "DOGE/USD": sig(False, 0.02), "PAXG/USD": sig(True, 0.002)}


def make(sigma=0.02, **overrides):
    """The long-short book alone, with the portfolio's daily volatility fixed at `sigma`."""
    options = {"rotation_weight": 0.0, "book_mode": "long_short", "secure_k": 1.0, "window_start_ms": START,
               "window_start_equity": 1.0}
    options.update(overrides)
    strategy = Strategy(StrategyConfig(universe=list(UNIVERSE), **options))
    strategy.signals = lambda: MARKET
    strategy._portfolio_volatility = lambda ts, targets: sigma
    return strategy


def locked(strategy, ts, equity, state=None):
    """Whether the profits lock is on after the decision at `ts`."""
    state = state or StrategyState()
    strategy.decide(ts, equity, {}, state)
    return state.profit_locked


class SecureProfitsTest(unittest.TestCase):
    def test_off_by_default(self):
        strategy, state = make(secure_k=0.0), StrategyState()
        d = strategy.decide(START + 4 * DAY, 2.0, {}, state)
        self.assertFalse(state.profit_locked)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.5)

    def test_secures_once_the_gain_beats_a_normal_loss_over_the_days_left(self):
        # 10 days left at 2% daily volatility: the bar is 2% x sqrt(10) = 6.3%.
        strategy, state = make(), StrategyState()
        d = strategy.decide(START + 4 * DAY, 1.06, {}, state)
        self.assertFalse(state.profit_locked)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.5)
        d = strategy.decide(START + 4 * DAY + HOUR_MS, 1.07, {}, state)
        self.assertTrue(state.profit_locked)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.25)
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.0625)
        self.assertAlmostEqual(d.exposure_limit, 0.5)

    def test_the_bar_falls_as_the_window_ends(self):
        self.assertFalse(locked(make(), START + 4 * DAY, 1.04))       # needs 6.3%
        self.assertTrue(locked(make(), START + 12 * DAY, 1.04))       # needs 2.8%

    def test_stays_secured_for_the_window_then_starts_afresh(self):
        strategy, state = make(), StrategyState()
        self.assertTrue(locked(strategy, START + 4 * DAY, 1.10, state))
        self.assertTrue(locked(strategy, START + 6 * DAY, 1.00, state))       # gain gone: still secured
        self.assertFalse(locked(strategy, START + 14 * DAY, 1.00, state))     # the next window
        self.assertEqual(state.window_index, 1)
        self.assertAlmostEqual(state.window_equity, 1.00)                               # counts from its start

    def test_never_before_the_window_or_without_a_volatility(self):
        self.assertFalse(locked(make(), START - HOUR_MS, 2.0))
        self.assertFalse(locked(make(sigma=0.0), START + 4 * DAY, 2.0))

    def test_a_halted_coin_keeps_its_weight_and_the_other_longs_make_room(self):
        strategy, state = make(), StrategyState()
        d = strategy.decide(START + 4 * DAY, 1.10, {"BTC/USD": 0.5}, state, frozen={"BTC/USD"})
        self.assertTrue(state.profit_locked)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.5)       # cannot be sold
        self.assertAlmostEqual(d.targets["ETH/USD"], 0.0)       # longs planned at 0.375 in all: BTC fills them
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.0625)

    def test_state_survives_a_restart(self):
        strategy, state = make(), StrategyState()
        strategy.decide(START + 4 * DAY, 1.10, {}, state)
        restored = StrategyState.from_dict(state.to_dict())
        self.assertEqual((restored.window_index, restored.window_equity, restored.profit_locked), (0, 1.0, True))


class KeepInvestedTest(unittest.TestCase):
    """secure_mode other than "half": the rotation's coins move and the capital stays invested."""
    BOOK = {"BTC/USD": 0.5, "ETH/USD": 0.25, "SOL/USD": -0.25}           # fractions of the book's 30%
    ROTATION = {"DOGE/USD": 0.5, "SOL/USD": 0.5}                          # fractions of the sleeve's 70%
    MERGED = {"BTC/USD": 0.15, "ETH/USD": 0.075, "SOL/USD": 0.275, "DOGE/USD": 0.35}
    RANKED = {"BTC/USD": sig(), "ETH/USD": sig()._replace(return_rotation=0.2),
              "SOL/USD": sig()._replace(return_rotation=0.3), "DOGE/USD": sig()._replace(return_rotation=0.1),
              "PAXG/USD": sig()._replace(return_rotation=0.4)}

    def moved(self, mode, rotation=None, frozen=frozenset()):
        strategy = Strategy(StrategyConfig(universe=list(UNIVERSE), secure_mode=mode, secure_top=2))
        out = strategy._locked_targets(dict(self.MERGED), self.BOOK, rotation or self.ROTATION, self.RANKED,
                                       frozen, {})
        return {p: round(w, 6) for p, w in out.items() if abs(w) > 1e-12}

    def test_into_the_book(self):
        self.assertEqual(self.moved("book"), {"BTC/USD": 0.5, "ETH/USD": 0.25, "SOL/USD": -0.25})

    def test_into_btc(self):
        self.assertEqual(self.moved("btc"), {"BTC/USD": 0.85, "ETH/USD": 0.075, "SOL/USD": -0.075})

    def test_into_gold(self):
        self.assertEqual(self.moved("gold"),
                         {"BTC/USD": 0.15, "ETH/USD": 0.075, "SOL/USD": -0.075, "PAXG/USD": 0.7})

    def test_spread_over_the_top_coins_but_never_the_defensive_pair(self):
        self.assertEqual(self.moved("spread"), {"BTC/USD": 0.15, "ETH/USD": 0.425, "SOL/USD": 0.275})

    def test_nothing_moves_while_the_rotation_holds_no_coins(self):
        self.assertEqual(self.moved("btc", rotation={"PAXG/USD": 1.0}), self.MERGED)

    def test_a_halted_pick_stays(self):
        self.assertEqual(self.moved("btc", frozen={"DOGE/USD"}),
                         {"BTC/USD": 0.5, "ETH/USD": 0.075, "SOL/USD": -0.075, "DOGE/USD": 0.35})

    def test_half_into_gold_keeps_every_dollar_invested(self):
        strategy, state = make(secure_mode="half_gold"), StrategyState()
        d = strategy.decide(START + 4 * DAY, 1.10, {}, state)
        self.assertTrue(state.profit_locked)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.25)
        self.assertAlmostEqual(d.targets["SOL/USD"], -0.0625)
        self.assertAlmostEqual(d.targets["PAXG/USD"], 0.5)
        self.assertAlmostEqual(sum(abs(w) for w in d.targets.values()), 1.0)

    def test_refresh_sells_the_picks_into_gold_then_buys_other_coins(self):
        strategy, state = make(secure_mode="refresh", rotation_weight=0.7), StrategyState()
        d = strategy.decide(START + 4 * DAY, 1.00, {}, state)                 # the rotation picks BTC and ETH
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.5)
        d = strategy.decide(START + 4 * DAY + HOUR_MS, 1.10, {}, state)       # secured: they go into PAXG
        self.assertTrue(state.profit_locked)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.15)                   # the book's share stays
        self.assertAlmostEqual(d.targets["PAXG/USD"], 0.7)
        self.assertEqual(state.rotation_cooldown["BTC/USD"], START + 14 * DAY)
        d = strategy.decide(START + 5 * DAY, 1.10, {}, state)                 # the next pick: other coins
        self.assertAlmostEqual(d.targets["PAXG/USD"], 0.0)
        self.assertAlmostEqual(d.targets["SOL/USD"], 0.3125)
        self.assertAlmostEqual(d.targets["DOGE/USD"], 0.3125)
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.15)

    def test_the_exposure_limit_stays_when_the_capital_stays_invested(self):
        strategy, state = make(secure_mode="gold", rotation_weight=0.7), StrategyState()
        d = strategy.decide(START + 4 * DAY, 1.10, {}, state)
        self.assertTrue(state.profit_locked)
        self.assertAlmostEqual(d.exposure_limit, 1.0)
        self.assertAlmostEqual(d.targets["PAXG/USD"], 0.7)
        self.assertAlmostEqual(sum(abs(w) for w in d.targets.values()), 1.0)


class PortfolioVolatilityTest(unittest.TestCase):
    def strategy(self):
        strategy = Strategy(StrategyConfig(universe=list(UNIVERSE), secure_vol_hours=100))
        for pair in ("ETH/USD", "SOL/USD"):
            strategy.indicators[pair].returns = deque([0.01, -0.01] * 50)
        return strategy

    def test_daily_volatility_of_the_weighted_positions(self):
        sigma = self.strategy()._portfolio_volatility(START, {"ETH/USD": 0.5, "BTC/USD": 0.0})
        hourly = math.sqrt(sum(0.005 ** 2 for _ in range(100)) / 99)
        self.assertAlmostEqual(sigma, hourly * math.sqrt(24))

    def test_a_short_offsets_a_long_in_the_same_moves(self):
        self.assertAlmostEqual(self.strategy()._portfolio_volatility(START, {"ETH/USD": 0.5, "SOL/USD": -0.5}), 0.0)

    def test_measured_once_a_day(self):
        strategy = self.strategy()
        first = strategy._portfolio_volatility(START, {"ETH/USD": 0.5})
        strategy.indicators["ETH/USD"].returns = deque([0.05, -0.05] * 50)
        self.assertEqual(strategy._portfolio_volatility(START + HOUR_MS, {"ETH/USD": 0.5}), first)
        self.assertGreater(strategy._portfolio_volatility(START + DAY, {"ETH/USD": 0.5}), first)


if __name__ == "__main__":
    unittest.main()
