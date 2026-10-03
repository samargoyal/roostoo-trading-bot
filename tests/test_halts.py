"""Pairs Roostoo stops trading. The planner never trades them; with plan_around_halts the
strategy also holds them as they are, never enters them and re-plans the rest around them."""
import json
import math
import os
import random
import shutil
import tempfile
import unittest

from bot.backtest import run_backtest
from bot.config import Config, StrategyConfig
from bot.execution import halted_pairs
from bot.indicators import Signal
from bot.live import LiveBot
from bot.market_data import HOUR_MS
from bot.planner import ACTIVITY, plan_trades
from bot.strategy import (ENTRY, EXIT_TREND, HOLD_HALTED, PositionInfo, ShortInfo, Strategy,
                          StrategyState)
from tests.fakes import NOW, FakeBinance, FakeExchange, zigzag

UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "PAXG/USD"]
DAY = 24 * HOUR_MS


def sig(close=110.0, fast=105.0, slow=100.0, regime=90.0, vol=0.01, rot=0.0, trend_up=True):
    """Eligible and risk-on by default; `rot` is the 336h return the rotation ranks by."""
    return Signal(0, close, fast, slow, regime, 1.0, 50.0, 0.0, 0.0, vol, return_rotation=rot,
                  ema_trend_fast=101.0 if trend_up else 99.0, ema_trend_slow=100.0)


def make_strategy(signals, **overrides):
    base = {"rotation_weight": 0.0, "max_positions_risk_on": 4, "sizing": "inverse_atr",
            "core_weight": 0.0, "max_weight": 1.0}
    strategy = Strategy(StrategyConfig(universe=list(UNIVERSE), **dict(base, **overrides)))
    strategy.signals = lambda: signals
    return strategy


class DefensiveBookTest(unittest.TestCase):
    def test_a_halted_holding_is_kept_whatever_its_signal_says(self):
        signals = {"BTC/USD": sig(), "DOGE/USD": sig(fast=95.0)}   # DOGE's trend has broken
        state = StrategyState(positions={"DOGE/USD": PositionInfo(0, 110.0)})
        weights = {"DOGE/USD": 0.1}
        d = make_strategy(signals).decide(0, 1.0, weights, state, frozenset({"DOGE/USD"}))
        self.assertEqual(d.reasons["DOGE/USD"], HOLD_HALTED)
        self.assertEqual(d.targets["DOGE/USD"], 0.1)
        d = make_strategy(signals).decide(0, 1.0, weights, StrategyState(
            positions={"DOGE/USD": PositionInfo(0, 110.0)}))
        self.assertEqual(d.reasons["DOGE/USD"], EXIT_TREND)

    def test_a_halted_coin_is_never_entered_and_the_next_best_takes_its_place(self):
        signals = {"BTC/USD": sig(vol=0.03), "ETH/USD": sig(vol=0.02), "SOL/USD": sig(vol=0.01)}
        strategy = make_strategy(signals, max_positions_risk_on=1)
        d = strategy.decide(0, 1.0, {}, StrategyState(), frozenset({"SOL/USD"}))
        self.assertEqual([p for p, r in d.reasons.items() if r == ENTRY], ["ETH/USD"])
        self.assertEqual(d.targets["SOL/USD"], 0.0)

    def test_halted_holdings_fill_their_slots_and_use_up_budget(self):
        signals = {p: sig(vol=v) for p, v in [("BTC/USD", 0.01), ("ETH/USD", 0.02), ("SOL/USD", 0.03)]}
        state = StrategyState(positions={"DOGE/USD": PositionInfo(0, 110.0)})
        d = make_strategy(signals, max_positions_risk_on=2).decide(
            0, 1.0, {"DOGE/USD": 0.15}, state, frozenset({"DOGE/USD"}))
        self.assertEqual([p for p, r in d.reasons.items() if r == ENTRY], ["BTC/USD"])
        self.assertAlmostEqual(d.targets["BTC/USD"], 0.75 - 0.15)

    def test_erc_book_is_re_solved_around_the_halted_coin(self):
        signals = {"BTC/USD": sig(fast=95.0), "ETH/USD": sig(), "SOL/USD": sig(), "DOGE/USD": sig()}
        strategy = make_strategy(signals, sizing="erc")
        rng = random.Random(1)
        prices = {p: 100.0 for p in ("ETH/USD", "SOL/USD", "DOGE/USD")}
        for _ in range(300):
            common, e1, e2, e3 = (rng.gauss(0, 0.01) for _ in range(4))
            moves = {"DOGE/USD": common + 0.2 * e1, "ETH/USD": common + 0.2 * e2,   # ETH moves with DOGE
                     "SOL/USD": math.sqrt(1.04) * e3}                               # SOL on its own
            for pair, r in moves.items():
                prices[pair] *= math.exp(r)
                strategy.indicators[pair].closes.append(prices[pair])
        state = StrategyState(positions={"DOGE/USD": PositionInfo(0, 110.0)})
        d = strategy.decide(0, 1.0, {"DOGE/USD": 0.15}, state, frozenset({"DOGE/USD"}))
        self.assertEqual(d.targets["DOGE/USD"], 0.15)
        self.assertLess(d.targets["ETH/USD"], d.targets["SOL/USD"])
        self.assertAlmostEqual(d.targets["ETH/USD"] + d.targets["SOL/USD"], 0.75 - 0.15)
        # Without DOGE at all the two would share the budget about equally.
        del signals["DOGE/USD"]
        plain = strategy.decide(0, 1.0, {}, StrategyState())
        self.assertAlmostEqual(plain.targets["ETH/USD"] / plain.targets["SOL/USD"], 1.0, delta=0.25)

    def test_a_halted_short_is_held_and_halted_coins_are_not_shorted(self):
        # Only BTC is a long; the others are short candidates (none trends up).
        signals = {"BTC/USD": sig(), "ETH/USD": sig(vol=0.05, fast=95.0), "SOL/USD": sig(vol=0.04, fast=95.0),
                   "DOGE/USD": sig(vol=0.03, fast=95.0)}
        state = StrategyState(shorts={"SOL/USD": ShortInfo(0, 100.0)})
        strategy = make_strategy(signals, short_exposure=0.15, max_shorts=2, sizing="inverse_atr")
        d = strategy.decide(0, 1.0, {"SOL/USD": -0.05}, state, frozenset({"SOL/USD", "ETH/USD"}))
        self.assertEqual(d.targets["SOL/USD"], -0.05)
        self.assertEqual(d.reasons["SOL/USD"], HOLD_HALTED)
        self.assertEqual(d.targets["ETH/USD"], 0.0)
        self.assertLess(d.targets["DOGE/USD"], 0.0)


class RotationTest(unittest.TestCase):
    def market(self, trend_up=True):
        """Rotation candidates only: no coin trends up enough for the other book to buy it."""
        return {p: sig(rot=r, trend_up=trend_up, fast=95.0) for p, r in
                [("BTC/USD", 0.05), ("ETH/USD", 0.20), ("SOL/USD", 0.40), ("DOGE/USD", 0.10),
                 ("PAXG/USD", -0.01)]}

    def test_a_halted_coin_is_not_picked(self):
        strategy = make_strategy(self.market(), rotation_weight=0.4)
        d = strategy.decide(DAY, 1.0, {}, StrategyState(), frozenset({"SOL/USD"}))
        self.assertEqual(d.rotation, {"ETH/USD": 0.5, "DOGE/USD": 0.5})

    def test_a_halted_holding_keeps_its_slot_and_survives_the_trend_exit(self):
        state = StrategyState()
        make_strategy(self.market(), rotation_weight=0.4).decide(DAY, 1.0, {}, state)
        self.assertEqual(state.rotation_plan, {"SOL/USD": 0.5, "ETH/USD": 0.5})
        signals = self.market()
        signals["SOL/USD"] = sig(rot=-0.30, fast=95.0)        # no longer a pick, but halted
        d = make_strategy(signals, rotation_weight=0.4).decide(
            2 * DAY, 1.0, {"SOL/USD": 0.2, "ETH/USD": 0.2}, state, frozenset({"SOL/USD"}))
        self.assertEqual(d.rotation, {"SOL/USD": 0.5, "ETH/USD": 0.5})
        d = make_strategy(self.market(trend_up=False), rotation_weight=0.4).decide(
            2 * DAY + HOUR_MS, 1.0, {"SOL/USD": 0.2, "ETH/USD": 0.2}, state, frozenset({"SOL/USD"}))
        self.assertEqual(d.rotation, {"SOL/USD": 0.5})
        self.assertEqual(d.targets["SOL/USD"], 0.2)
        self.assertEqual(d.targets["ETH/USD"], 0.0)


class FinalTargetsTest(unittest.TestCase):
    def test_a_halted_pair_outside_the_plan_is_held_and_the_others_give_way(self):
        signals = {"BTC/USD": sig(vol=0.01), "ETH/USD": sig(vol=0.02)}
        strategy = make_strategy(signals, max_positions_risk_on=2)
        free = strategy.decide(0, 1.0, {}, StrategyState())
        planned = sum(t for t in free.targets.values() if t > 0)
        # 10% of the account is stuck in XRP, which the strategy does not trade at all.
        d = strategy.decide(0, 1.0, {"XRP/USD": 0.1}, StrategyState(), frozenset({"XRP/USD"}))
        self.assertEqual(d.targets["XRP/USD"], 0.1)
        self.assertEqual(d.reasons["XRP/USD"], HOLD_HALTED)
        self.assertAlmostEqual(sum(t for t in d.targets.values() if t > 0), planned)
        self.assertAlmostEqual(d.targets["BTC/USD"] / d.targets["ETH/USD"],
                               free.targets["BTC/USD"] / free.targets["ETH/USD"])

    def test_nothing_changes_without_halts(self):
        signals = {p: sig(vol=0.01 * (i + 1), rot=0.1 * i) for i, p in enumerate(UNIVERSE)}
        a = make_strategy(signals, rotation_weight=0.4).decide(DAY, 1.0, {}, StrategyState())
        b = make_strategy(signals, rotation_weight=0.4).decide(DAY, 1.0, {}, StrategyState(), frozenset())
        self.assertEqual(a.targets, b.targets)
        self.assertEqual(a.reasons, b.reasons)

    def test_the_planner_drops_halted_pairs_so_the_activity_trade_can_fire(self):
        # The strategy is not told: it wants to sell DOGE, whose trend has broken.
        signals = {"BTC/USD": sig(fast=95.0), "DOGE/USD": sig(fast=95.0)}
        state = StrategyState(positions={"DOGE/USD": PositionInfo(0, 110.0)})
        weights = {"DOGE/USD": 0.1, "PAXG/USD": 0.05}
        d = make_strategy(signals, core_weight=0.05).decide(0, 1.0, weights, state)
        cfg = Config().execution
        ts = 1000 * DAY + 7 * HOUR_MS + 30 * 60_000            # late in an 8-hour block, no fills yet
        unaware = plan_trades(d, weights, 100000.0, ts, 0, cfg, 0.005)
        self.assertEqual([t.pair for t in unaware], ["DOGE/USD"])   # would be refused: no fill
        trades = plan_trades(d, weights, 100000.0, ts, 0, cfg, 0.005, frozenset({"DOGE/USD"}))
        self.assertEqual([(t.pair, t.reason) for t in trades], [("PAXG/USD", ACTIVITY)])

    def test_the_activity_trade_never_picks_a_halted_pair(self):
        signals = {"BTC/USD": sig(fast=95.0)}
        state = StrategyState(positions={"DOGE/USD": PositionInfo(0, 110.0)})
        d = make_strategy(signals).decide(0, 1.0, {"DOGE/USD": 0.1}, state, frozenset({"DOGE/USD"}))
        cfg = Config().execution
        ts = 1000 * DAY + 7 * HOUR_MS + 30 * 60_000            # late in an 8-hour block, no fills yet
        trades = plan_trades(d, {"DOGE/USD": 0.1}, 100000.0, ts, 0, cfg, 0.005)
        self.assertTrue(all(t.pair != "DOGE/USD" for t in trades))
        self.assertTrue(all(t.reason != ACTIVITY or t.pair != "DOGE/USD" for t in trades))


class ExchangeInfoTest(unittest.TestCase):
    def test_halted_pairs_are_those_not_tradable_or_not_listed(self):
        info = {"TradePairs": {"BTC/USD": {"CanTrade": True}, "ETH/USD": {"CanTrade": False},
                               "SOL/USD": {}}}
        self.assertEqual(halted_pairs(info, ["BTC/USD", "ETH/USD", "SOL/USD", "XYZ/USD"]),
                         {"ETH/USD", "XYZ/USD"})


class LiveHaltTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        series = {"BTCUSDT": zigzag(NOW, 50000.0, 0.0005), "ETHUSDT": zigzag(NOW, 2000.0, 0.0004),
                  "PAXGUSDT": zigzag(NOW, 3000.0, -0.0012)}
        self.binance = FakeBinance(series)
        self.exchange = FakeExchange({"BTC/USD": series["BTCUSDT"][-1].close,
                                      "ETH/USD": series["ETHUSDT"][-1].close,
                                      "PAXG/USD": series["PAXGUSDT"][-1].close})

    def tearDown(self):
        shutil.rmtree(self.dir)

    def bot(self, plan_around_halts=True):
        cfg = Config()
        cfg.strategy.universe = ["BTC/USD", "ETH/USD", "PAXG/USD"]
        cfg.strategy.plan_around_halts = plan_around_halts
        cfg.execution.poll_interval_sec = 1
        bot = LiveBot(cfg, self.dir, client=self.exchange, binance=self.binance, sleep=lambda s: None)
        bot.start()
        return bot

    def test_starts_with_a_halted_pair_and_never_orders_it(self):
        for told in (True, False):
            with self.subTest(plan_around_halts=told):
                self.setUp()
                self.exchange.halted = {"ETH/USD"}
                bot = self.bot(plan_around_halts=told)     # used to refuse to start
                bot.cycle()
                self.assertTrue(self.exchange.placed)
                self.assertNotIn("ETH/USD", {o["Pair"] for o in self.exchange.placed})
                with open(os.path.join(self.dir, "journal", "decisions.jsonl")) as f:
                    record = json.loads(f.readlines()[-1])
                self.assertEqual(record["halted"], ["ETH/USD"])
                self.tearDown()
        self.setUp()

    def test_rules_are_re_read_every_cycle_and_a_resumed_pair_trades_again(self):
        self.exchange.halted = {"ETH/USD"}
        bot = self.bot()
        bot.cycle()
        calls = self.exchange.info_calls
        self.exchange.halted = set()
        self.exchange.now += HOUR_MS
        bot.cycle()
        self.assertEqual(self.exchange.info_calls, calls + 1)
        self.assertIn("ETH/USD", {o["Pair"] for o in self.exchange.placed})

    def test_the_strategy_holds_a_halted_holding(self):
        bot = self.bot()
        bot.cycle()
        self.assertGreater(self.exchange.wallet.get("ETH", {}).get("Free", 0.0), 0.0)
        self.exchange.halted = {"ETH/USD"}
        self.exchange.now += HOUR_MS
        bot.cycle()
        with open(os.path.join(self.dir, "journal", "decisions.jsonl")) as f:
            record = json.loads(f.readlines()[-1])
        self.assertEqual(record["reasons"]["ETH/USD"], HOLD_HALTED)

    def test_an_empty_pair_list_keeps_the_last_rules(self):
        self.exchange.halted = {"ETH/USD"}
        bot = self.bot()
        bot.cycle()
        self.exchange.exchange_info = lambda: {"TradePairs": {}}
        self.assertEqual(bot._refresh_rules(["BTC/USD", "ETH/USD", "PAXG/USD"]), {"ETH/USD"})
        self.assertIn("BTC/USD", bot.executor.rules)


class BacktestHaltTest(unittest.TestCase):
    def test_no_trades_in_a_halted_pair_whether_or_not_the_strategy_is_told(self):
        end = NOW
        bars = {"BTC/USD": zigzag(end, 50000.0, 0.0005, hours=1600),
                "ETH/USD": zigzag(end, 2000.0, 0.0004, hours=1600),
                "SOL/USD": zigzag(end, 100.0, 0.0006, hours=1600, swing=1.5),
                "PAXG/USD": zigzag(end, 3000.0, 0.0, hours=1600)}
        cfg = Config()
        cfg.strategy.universe = list(bars)
        start = end // HOUR_MS * HOUR_MS - 300 * HOUR_MS
        halt = (start, start + 100 * HOUR_MS)
        halts = {"ETH/USD": [halt]}
        aware = run_backtest(cfg, bars, start, end, 0.001, 0.0, halts=halts)
        cfg.strategy.plan_around_halts = False
        unaware = run_backtest(cfg, bars, start, end, 0.001, 0.0, halts=halts)
        for result in (aware, unaware):
            self.assertFalse([t for t in result.trades if t.pair == "ETH/USD" and halt[0] <= t.ts < halt[1]])
            self.assertTrue([t for t in result.trades if t.pair == "ETH/USD" and t.ts >= halt[1]])


if __name__ == "__main__":
    unittest.main()
