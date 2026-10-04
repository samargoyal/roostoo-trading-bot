"""The optional short sleeve: strategy, planner, executor, valuation, backtest and client."""
import json
import shutil
import tempfile
import unittest

from bot.backtest import run_backtest
from bot.config import Config, ExecutionConfig, StrategyConfig
from bot.execution import SHORT_CLOSE, SHORT_OPEN, Executor, parse_rules
from bot.live import LiveBot, portfolio_value
from bot.market_data import HOUR_MS
from bot.planner import BUY, COVER, REBALANCE, SELL, SHORT, plan_trades
from bot.roostoo import sign
from bot.reasons import ENTRY, EXIT_SHORT, EXIT_SHORT_STOP, SHORT_ENTRY, SHORT_HOLD
from bot.strategy import Decision, ShortInfo, Strategy, StrategyState
from tests.fakes import NOW, FakeBinance, FakeExchange, zigzag
from tests.test_roostoo import DOC_SECRET, FakeResponse, make_client
from tests.test_strategy import sig

UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "PEPE/USD", "PAXG/USD"]


def make_strategy(signals, **overrides):
    base = {"short_exposure": 0.15, "rotation_weight": 0.0, "max_positions_risk_on": 4, "sizing": "inverse_atr"}
    cfg = StrategyConfig(universe=list(UNIVERSE), **dict(base, **overrides))
    strategy = Strategy(cfg)
    strategy.signals = lambda: signals
    return strategy


def market():
    """Calm BTC and ETH in uptrends (the longs); SOL, DOGE and PEPE in downtrends, so they can
    only be shorted, PEPE the wildest."""
    return {
        "BTC/USD": sig(vol=0.005), "ETH/USD": sig(vol=0.006), "SOL/USD": sig(vol=0.010, fast=90.0),
        "DOGE/USD": sig(vol=0.030, fast=90.0), "PEPE/USD": sig(vol=0.050, fast=90.0),
        "PAXG/USD": sig(vol=0.002),
    }


class StrategyShortTest(unittest.TestCase):
    def test_shorts_the_most_volatile_coins_but_never_paxg_or_a_long(self):
        d = make_strategy(market(), max_positions_risk_on=2, max_shorts=2).decide(0, 1.0, {}, StrategyState())
        longs = {p for p, r in d.reasons.items() if r == ENTRY}
        shorts = {p for p, r in d.reasons.items() if r == SHORT_ENTRY}
        self.assertEqual(shorts, {"PEPE/USD", "DOGE/USD"})
        self.assertFalse(longs & shorts)
        self.assertNotIn("PAXG/USD", shorts)
        self.assertTrue(all(d.targets[p] < 0 for p in shorts))
        self.assertLessEqual(-sum(d.targets[p] for p in shorts), 0.15 + 1e-9)

    def test_no_new_shorts_into_oversold_coins(self):
        signals = market()
        signals["PEPE/USD"] = sig(vol=0.050, fast=90.0, rsi=20.0)
        d = make_strategy(signals, max_shorts=1).decide(0, 1.0, {}, StrategyState())
        self.assertNotEqual(d.reasons.get("PEPE/USD"), SHORT_ENTRY)

    def test_trailing_stop_covers_and_reconcile_starts_a_cooldown(self):
        signals = market()
        signals["PEPE/USD"] = sig(vol=0.050, fast=90.0, close=125.0, atr=1.0)
        strategy = make_strategy(signals)
        state = StrategyState(shorts={"PEPE/USD": ShortInfo(0, 110.0)})
        d = strategy.decide(5 * HOUR_MS, 1.0, {"PEPE/USD": -0.05}, state)
        self.assertEqual(d.reasons["PEPE/USD"], EXIT_SHORT_STOP)    # 125 > 110 + 10 x 1
        self.assertEqual(d.targets["PEPE/USD"], 0.0)
        strategy.reconcile(5 * HOUR_MS, {}, d, state)
        self.assertNotIn("PEPE/USD", state.shorts)
        self.assertEqual(state.short_cooldown_until["PEPE/USD"], 29 * HOUR_MS)

    def test_held_short_is_kept_and_its_low_tracked(self):
        signals = market()
        signals["PEPE/USD"] = sig(vol=0.050, fast=90.0, close=95.0)
        state = StrategyState(shorts={"PEPE/USD": ShortInfo(0, 100.0)})
        d = make_strategy(signals).decide(0, 1.0, {"PEPE/USD": -0.05}, state)
        self.assertEqual(d.reasons["PEPE/USD"], SHORT_HOLD)
        self.assertEqual(state.shorts["PEPE/USD"].lowest_close, 95.0)

    def test_brake_halves_shorts(self):
        strategy = make_strategy(market(), max_shorts=1)
        normal = strategy.decide(0, 100.0, {}, StrategyState()).targets["PEPE/USD"]
        braked = strategy.decide(0, 90.0, {}, StrategyState(peak_equity=100.0)).targets["PEPE/USD"]
        self.assertAlmostEqual(braked, normal / 2)

    def test_switching_the_sleeve_off_covers_remembered_shorts(self):
        cfg = StrategyConfig(universe=list(UNIVERSE), short_exposure=0.0, rotation_weight=0.0)
        strategy = Strategy(cfg)
        strategy.signals = lambda: market()
        state = StrategyState(shorts={"PEPE/USD": ShortInfo(0, 100.0)})
        d = strategy.decide(0, 1.0, {"PEPE/USD": -0.05}, state)
        self.assertEqual((d.targets["PEPE/USD"], d.reasons["PEPE/USD"]), (0.0, EXIT_SHORT))

    def test_reconcile_registers_shorts_from_the_wallet(self):
        strategy = make_strategy(market())
        state = StrategyState()
        strategy.reconcile(7, {"PEPE/USD": -0.05, "BTC/USD": 0.15}, None, state)
        self.assertEqual(state.shorts["PEPE/USD"].entry_ts, 7)
        self.assertIn("BTC/USD", state.positions)
        self.assertEqual(StrategyState.from_dict(json.loads(json.dumps(state.to_dict()))), state)


def decision(targets, reasons=None):
    return Decision(0, True, 0.75, 0.0, False, targets, reasons or {}, {})


class PlannerShortTest(unittest.TestCase):
    def plan(self, targets, weights, reasons=None):
        return plan_trades(decision(targets, reasons), weights, 100000.0, 0, 0, ExecutionConfig(), 0.005)

    def test_open_resize_and_cover(self):
        (t,) = self.plan({"PEPE/USD": -0.05}, {})
        self.assertEqual((t.side, t.reason, round(t.usd)), (SHORT, SHORT_ENTRY, 5000))
        (t,) = self.plan({"PEPE/USD": -0.02}, {"PEPE/USD": -0.07}, {"PEPE/USD": SHORT_HOLD})
        self.assertEqual((t.side, t.reason, t.close_position), (COVER, REBALANCE, False))
        (t,) = self.plan({"PEPE/USD": 0.0}, {"PEPE/USD": -0.05}, {"PEPE/USD": EXIT_SHORT_STOP})
        self.assertEqual((t.side, t.reason, t.close_position), (COVER, EXIT_SHORT_STOP, True))
        self.assertEqual(self.plan({"PEPE/USD": -0.06}, {"PEPE/USD": -0.05}, {"PEPE/USD": SHORT_HOLD}), [])

    def test_switching_sides_closes_first_then_opens(self):
        trades = self.plan({"DOGE/USD": -0.05, "BTC/USD": 0.10}, {"DOGE/USD": 0.08}, {"DOGE/USD": SHORT_ENTRY})
        self.assertEqual([(t.pair, t.side) for t in trades],
                         [("DOGE/USD", SELL), ("DOGE/USD", SHORT), ("BTC/USD", BUY)])
        self.assertTrue(trades[0].close_position)


def short_trade(side, usd, close=False):
    from bot.planner import PlannedTrade
    return PlannedTrade("PEPE/USD", side, usd, close, 0.0, -0.05, SHORT_ENTRY)


class ExecutorShortTest(unittest.TestCase):
    def make(self, exchange, dry_run=False):
        self.rows = []
        return Executor(exchange, parse_rules(exchange.exchange_info()), ExecutionConfig(),
                        lambda o: self.rows.append(o.as_row()), dry_run=dry_run, sleep=lambda s: None)

    def test_short_open_locks_collateral_and_is_journalled(self):
        ex = FakeExchange({"PEPE/USD": 10.0})
        (order,) = self.make(ex).execute([short_trade(SHORT, 5000.0)], ex.ticker())
        self.assertEqual(ex.short_calls, [("open", "PEPE/USD", "5000.00")])
        self.assertEqual((order.order_type, order.status), (SHORT_OPEN, "OPEN"))
        self.assertAlmostEqual(ex.wallet["USD"]["ShortCollateral"], 5000.0)
        self.assertEqual(self.rows[0]["collateral"], "5000.00")

    def test_partial_and_full_cover(self):
        ex = FakeExchange({"PEPE/USD": 10.0})
        ex.short_open("PEPE/USD", "5000")
        ex.short_calls.clear()
        executor = self.make(ex)
        executor.execute([short_trade(COVER, 2000.0)], ex.ticker())
        self.assertEqual(ex.short_calls[0][0:2], ("close", "PEPE/USD"))
        self.assertIsNotNone(ex.short_calls[0][2])
        executor.execute([short_trade(COVER, 0.0, close=True)], ex.ticker())
        self.assertEqual(ex.short_calls[1], ("close", "PEPE/USD", None))
        self.assertEqual(ex.shorts, {})
        self.assertEqual(self.rows[-1]["type"], SHORT_CLOSE)
        self.assertNotEqual(self.rows[-1]["realized_pnl"], "")

    def test_dry_run_sends_no_short_orders(self):
        ex = FakeExchange({"PEPE/USD": 10.0})
        (order,) = self.make(ex, dry_run=True).execute([short_trade(SHORT, 5000.0)], ex.ticker())
        self.assertEqual((ex.short_calls, order.status), ([], "DRY_RUN"))


class ValuationTest(unittest.TestCase):
    def test_short_adds_collateral_plus_pnl_to_equity_and_negative_exposure(self):
        wallet = {"USD": {"Free": 90000.0, "Lock": 0.0, "ShortCollateral": 10000.0}}
        shorts = [{"Pair": "PEPE/USD", "ShortQty": 1000.0, "EntryPrice": 10.0, "Collateral": 10000.0}]
        quotes = {"PEPE/USD": {"LastPrice": 9.0}}
        equity, cash, values = portfolio_value(wallet, quotes, shorts)
        self.assertEqual((equity, cash), (101000.0, 90000.0))   # 90k + 10k collateral + 1k profit
        self.assertEqual(values, {"PEPE/USD": -9000.0})


class LiveShortTest(unittest.TestCase):
    def test_a_cycle_with_the_sleeve_on_opens_shorts_and_remembers_them(self):
        tmp = tempfile.mkdtemp()
        try:
            series = {"BTCUSDT": zigzag(NOW, 50000.0, 0.0005), "ETHUSDT": zigzag(NOW, 2000.0, 0.0004),
                      "PEPEUSDT": zigzag(NOW, 10.0, -0.006, swing=4.0), "PAXGUSDT": zigzag(NOW, 3000.0, -0.0012)}
            prices = {p.replace("USDT", "/USD"): b[-1].close for p, b in series.items()}
            exchange = FakeExchange(prices)
            cfg = Config()
            cfg.strategy.universe = list(prices)
            cfg.strategy.short_exposure = 0.15
            bot = LiveBot(cfg, tmp, client=exchange, binance=FakeBinance(series), sleep=lambda s: None)
            bot.start()
            bot.cycle()
            self.assertIn("PEPE/USD", exchange.shorts)
            self.assertIn("PEPE/USD", bot.state.shorts)
            self.assertNotIn("PAXG/USD", exchange.shorts)
        finally:
            shutil.rmtree(tmp)


class BacktestShortTest(unittest.TestCase):
    def test_short_on_a_falling_volatile_coin_makes_money(self):
        end = NOW
        bars = {"BTC/USD": zigzag(end, 50000.0, 0.0005, hours=1600),
                "PEPE/USD": zigzag(end, 10.0, -0.006, hours=1600, swing=3.0),
                "PAXG/USD": zigzag(end, 3000.0, 0.0, hours=1600)}
        cfg = Config()
        cfg.strategy.universe = list(bars)
        cfg.strategy.short_exposure = 0.15
        start = end // HOUR_MS * HOUR_MS - 500 * HOUR_MS
        result = run_backtest(cfg, bars, start, end, 0.001, 0.0)
        shorts = [t for t in result.trades if t.side in (SHORT, COVER)]
        self.assertTrue(any(t.side == SHORT and t.pair == "PEPE/USD" for t in shorts))
        self.assertGreater(result.curve[-1][1], 0)
        cfg.strategy.short_exposure = 0.0
        without = run_backtest(cfg, bars, start, end, 0.001, 0.0)
        self.assertGreater(result.curve[-1][1], without.curve[-1][1])


class ClientShortTest(unittest.TestCase):
    def test_short_endpoints_send_only_their_documented_parameters(self):
        client, session = make_client(
            FakeResponse(200, {"Success": True, "ID": 1, "Status": "OPEN"}),
            FakeResponse(200, {"Success": True, "ClosedQty": 1.0}),
            FakeResponse(200, {"Success": True, "Positions": [{"Pair": "PEPE/USD"}]}),
        )
        client.short_open("PEPE/USD", "100.00")
        client.short_close("PEPE/USD")
        self.assertEqual(client.short_positions(), [{"Pair": "PEPE/USD"}])
        (m1, url1, body1, h1), (m2, url2, body2, h2), (m3, url3, _, h3) = session.calls
        self.assertTrue(url1.endswith("/v6/short_open"))
        self.assertTrue(body1.startswith("collateral=100.00&pair=PEPE/USD&timestamp="))
        self.assertEqual(h1["MSG-SIGNATURE"], sign(body1, DOC_SECRET))
        self.assertTrue(url2.endswith("/v6/short_close"))
        self.assertNotIn("close_qty", body2)
        self.assertIn("/v6/short_positions?timestamp=", url3)


if __name__ == "__main__":
    unittest.main()
