"""Backtest the live strategy on Binance hourly candles.

    python -m bot.backtest
    python -m bot.backtest --start 2025-10-01 --end 2026-10-01 --config my.json

The Strategy and plan_trades code is exactly what the live bot runs. Orders fill at
the hourly close, in two scenarios: every order as a taker (0.1% fee plus half a
spread of slippage), and every order as a maker (0.05%). Hourly bars cannot show
whether a resting limit order would have filled, so live costs land in between.

By default the pairs are chosen with the universe rule (universe.py) as of the start
of the window, using only data available then; --fixed-universe tests the configured
list instead. Candles are cached under data/; results are written to runs/backtest/.
"""
import argparse
import copy
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor

from bot.config import Config, load_config
from bot.market_data import HOUR_MS, Bar, BinanceClient, load_history
from bot.metrics import summarize
from bot.planner import BUY, SELL, SHORT, plan_trades
from bot.roostoo import RoostooClient
from bot.strategy import Strategy, StrategyState
from bot.universe import fetch_candidates, select_universe

log = logging.getLogger("bot.backtest")


@dataclass
class Trade:
    ts: int
    pair: str
    side: str
    quantity: float
    price: float
    fee: float
    reason: str

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class Result:
    name: str
    curve: List[Tuple[int, float]]
    trades: List[Trade] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class ShortPosition:
    """A short as Roostoo keeps it: quantity, average entry price and locked USD collateral."""
    qty: float
    entry: float
    collateral: float

    def value(self, price: float) -> float:
        """What closing it would return before the fee: collateral plus profit or loss."""
        return self.collateral + self.qty * (self.entry - price)


def run_backtest(cfg: Config, bars: Dict[str, List[Bar]], start_ms: int, end_ms: int,
                 fee: float, slippage: float, name: str = "strategy",
                 monthly_universe: bool = False,
                 slippage_by_pair: Optional[Dict[str, float]] = None,
                 halts: Optional[Dict[str, List[Tuple[int, int]]]] = None,
                 external_scores: Optional[Dict[int, Dict[str, float]]] = None) -> Result:
    """Replay the strategy hour by hour from start_ms. Bars before start_ms only warm up indicators.

    Shorts follow Roostoo's rules: opening locks the USD collateral (quantity = collateral /
    price) plus a 0.1% fee out of free cash; closing returns the collateral plus the profit or
    loss, less 0.1% of the closed value. The short fee is the same for limit and market orders.

    With monthly_universe, `bars` holds every candidate and the universe rule is re-applied on
    the first day of each month with data available then, as the live list would be refreshed.
    slippage_by_pair overrides the slippage for given pairs (for example half their spread).
    halts maps a pair to [start, end) times when the exchange refuses its orders. As in the live
    bot, the planner drops their trades, and with strategy.plan_around_halts the strategy is told.
    external_scores feeds saved model scores to the "external" rankings (research only).
    """
    cfg = copy.deepcopy(cfg)
    strategy = Strategy(cfg.strategy, pairs=list(bars) if monthly_universe else None,
                        external_scores=external_scores)
    month = None
    state = StrategyState()
    cash = cfg.backtest.initial_cash
    short_fee = cfg.backtest.short_fee
    holdings: Dict[str, float] = {}
    shorts: Dict[str, ShortPosition] = {}
    closes: Dict[str, float] = {}
    by_ts: Dict[int, List[Tuple[str, Bar]]] = {}
    for pair, series in bars.items():
        for bar in series:
            by_ts.setdefault(bar.ts, []).append((pair, bar))

    def mark() -> Tuple[float, Dict[str, float]]:
        values = {p: q * closes[p] for p, q in holdings.items()}
        equity = cash + sum(values.values()) + sum(sp.value(closes[p]) for p, sp in shorts.items())
        for p, sp in shorts.items():
            values[p] = values.get(p, 0.0) - sp.qty * closes[p]
        return equity, values

    curve: List[Tuple[int, float]] = []
    trades: List[Trade] = []
    risk_on_hours = 0
    exposure_sum = 0.0
    for ts in sorted(by_ts):
        if ts >= end_ms:
            break
        for pair, bar in by_ts[ts]:
            strategy.update(pair, bar)
            closes[pair] = bar.close
        if ts < start_ms:
            continue
        now = ts + HOUR_MS  # the decision is made when this bar closes
        if monthly_universe:
            this_month = datetime.fromtimestamp(now / 1000, tz=timezone.utc).strftime("%Y-%m")
            if this_month != month:
                month = this_month
                first = int(datetime.strptime(this_month, "%Y-%m").replace(tzinfo=timezone.utc).timestamp() * 1000)
                universe = select_universe(bars, max(first, start_ms), cfg.universe, cfg.strategy.defensive_pair)
                if cfg.strategy.regime_pair not in universe:
                    universe.insert(0, cfg.strategy.regime_pair)
                cfg.strategy.universe = universe

        equity, values = mark()
        weights = {p: v / equity for p, v in values.items()}
        frozen = {p for p, spans in (halts or {}).items() if any(a <= now < b for a, b in spans)}
        decision = strategy.decide(now, equity, weights, state,
                                   frozen if cfg.strategy.plan_around_halts else frozenset())
        planned = plan_trades(decision, weights, equity, now, state.last_fill_ts,
                              cfg.execution, cfg.strategy.min_position_weight, frozen)
        for t in planned:
            price = closes.get(t.pair)
            if not price:
                continue
            slip = slippage if slippage_by_pair is None else slippage_by_pair.get(t.pair, slippage)
            if t.side == SELL:
                held = holdings.get(t.pair, 0.0)
                quantity = held if t.close_position else min(held, t.usd / price)
                if quantity <= 0:
                    continue
                fill = price * (1.0 - slip)
                proceeds = quantity * fill
                paid = proceeds * fee
                cash += proceeds - paid
                holdings[t.pair] = held - quantity
            elif t.side == BUY:
                spend = min(t.usd, cash / (1.0 + fee))
                if spend < cfg.execution.min_trade_usd:
                    continue
                fill = price * (1.0 + slip)
                quantity = spend / fill
                paid = spend * fee
                cash -= spend + paid
                holdings[t.pair] = holdings.get(t.pair, 0.0) + quantity
            elif t.side == SHORT:
                collateral = min(t.usd, cash / (1.0 + short_fee))
                if collateral < cfg.execution.min_trade_usd:
                    continue
                fill = price * (1.0 - slip)       # a market short fills at the bid
                quantity = collateral / fill
                paid = collateral * short_fee
                cash -= collateral + paid
                old = shorts.get(t.pair)
                if old is None:
                    shorts[t.pair] = ShortPosition(quantity, fill, collateral)
                else:
                    total = old.qty + quantity
                    shorts[t.pair] = ShortPosition(total, (old.qty * old.entry + quantity * fill) / total,
                                                   old.collateral + collateral)
            else:  # COVER
                position = shorts.get(t.pair)
                if position is None:
                    continue
                fill = price * (1.0 + slip)       # a cover buys back at the ask
                quantity = position.qty if t.close_position else min(position.qty, t.usd / fill)
                if quantity <= 0:
                    continue
                share = quantity / position.qty
                pnl = max(quantity * (position.entry - fill), -position.collateral * share)
                paid = quantity * fill * short_fee
                cash += position.collateral * share + pnl - paid
                if share >= 1.0 - 1e-12:
                    del shorts[t.pair]
                else:
                    shorts[t.pair] = ShortPosition(position.qty - quantity, position.entry,
                                                   position.collateral * (1 - share))
            trades.append(Trade(now, t.pair, t.side, quantity, fill, paid, t.reason))
            state.last_fill_ts = now

        holdings = {p: q for p, q in holdings.items() if q > 1e-12}
        equity, values = mark()
        strategy.reconcile(now, {p: v / equity for p, v in values.items()}, decision, state)
        curve.append((now, equity))
        risk_on_hours += decision.risk_on
        exposure_sum += sum(abs(v) for v in values.values()) / equity

    stats = summarize(curve, cfg.backtest.initial_cash, [(t.ts, t.notional) for t in trades])
    if curve:
        stats["risk_on_share"] = risk_on_hours / len(curve)
        stats["average_exposure"] = exposure_sum / len(curve)
        stats["fees_paid"] = sum(t.fee for t in trades)
    return Result(name, curve, trades, stats)


def buy_and_hold(bars: Dict[str, List[Bar]], pairs: Sequence[str], start_ms: int, end_ms: int,
                 initial: float, name: str) -> Result:
    """Equal-weight buy and hold of `pairs` from start_ms, without fees: a reference line."""
    series = {p: {b.ts: b.close for b in bars.get(p, []) if start_ms <= b.ts < end_ms} for p in pairs}
    series = {p: s for p, s in series.items() if s}
    if not series:
        return Result(name, [])
    timeline = sorted(set().union(*[set(s) for s in series.values()]))
    quantity = {p: initial / len(series) / s[min(s)] for p, s in series.items()}
    last = {p: s[min(s)] for p, s in series.items()}
    curve = []
    for ts in timeline:
        for p, s in series.items():
            last[p] = s.get(ts, last[p])
        curve.append((ts + HOUR_MS, sum(quantity[p] * last[p] for p in series)))
    return Result(name, curve, stats=summarize(curve, initial))


# ---- reporting ---------------------------------------------------------------

REPORT_ROWS = [
    ("total_return", "Total return", "pct"),
    ("annual_return", "Annualised return", "pct"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("sharpe", "Sharpe", "num"),
    ("sortino", "Sortino", "num"),
    ("calmar", "Calmar", "num"),
    ("composite", "Composite (0.4 So + 0.3 Sh + 0.3 Ca)", "num"),
    ("window_return_median", "14-day windows: median return", "pct"),
    ("window_return_p10", "14-day windows: 10th pct return", "pct"),
    ("window_return_worst", "14-day windows: worst return", "pct"),
    ("window_positive_share", "14-day windows: share positive", "pct"),
    ("window_drawdown_worst", "14-day windows: worst drawdown", "pct"),
    ("window_composite_median", "14-day windows: median composite", "num"),
    ("trades_per_day", "Trades per day", "num"),
    ("min_trades_in_a_day", "Fewest trades in a day", "num"),
    ("active_day_share", "Days with a trade", "pct"),
    ("turnover_per_day", "Turnover per day", "pct"),
    ("fees_paid", "Fees paid (USD)", "usd"),
    ("average_exposure", "Average exposure", "pct"),
    ("risk_on_share", "Time risk-on", "pct"),
]


def format_value(value: Optional[float], kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return "%.2f%%" % (value * 100)
    if kind == "usd":
        return "{:,.0f}".format(value)
    return "%.2f" % value


def report(results: Sequence[Result], start: str, end: str) -> str:
    width = max(len(label) for _, label, _ in REPORT_ROWS) + 2
    lines = ["Backtest %s to %s (UTC), hourly bars" % (start, end), ""]
    lines.append("".ljust(width) + "".join(r.name.rjust(16) for r in results))
    for key, label, kind in REPORT_ROWS:
        cells = [format_value(r.stats.get(key), kind) for r in results]
        lines.append(label.ljust(width) + "".join(c.rjust(16) for c in cells))
    lines += ["", "Monthly returns"]
    months = monthly_returns(results)
    lines.append("".ljust(width) + "".join(r.name.rjust(16) for r in results))
    for month in sorted(months):
        cells = [format_value(months[month].get(r.name), "pct") for r in results]
        lines.append(month.ljust(width) + "".join(c.rjust(16) for c in cells))
    return "\n".join(lines)


def monthly_returns(results: Sequence[Result]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in results:
        if not r.curve:
            continue
        month_end: Dict[str, float] = {}
        for ts, value in r.curve:
            month_end[_iso(ts)[:7]] = value
        previous = r.curve[0][1]
        for month in sorted(month_end):
            out.setdefault(month, {})[r.name] = month_end[month] / previous - 1.0
            previous = month_end[month]
    return out


def write_outputs(result: Result, out_dir: str) -> None:
    path = os.path.join(out_dir, result.name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "summary.json"), "w") as f:
        json.dump(result.stats, f, indent=2, sort_keys=True)
    with open(os.path.join(path, "equity.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "equity"])
        writer.writerows((_iso(ts), "%.2f" % value) for ts, value in result.curve)
    with open(os.path.join(path, "trades.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "pair", "side", "quantity", "price", "notional", "fee", "reason"])
        for t in result.trades:
            writer.writerow([_iso(t.ts), t.pair, t.side, "%.8g" % t.quantity, "%.8g" % t.price,
                             "%.2f" % t.notional, "%.4f" % t.fee, t.reason])


def load_all(cfg: Config, pairs: Sequence[str], start_ms: int, end_ms: int) -> Dict[str, List[Bar]]:
    """Cached candles for many pairs, downloading the missing ones six at a time."""
    def one(pair: str) -> Tuple[str, List[Bar]]:
        bars = load_history(BinanceClient(cfg.live.binance_url), pair, start_ms, end_ms,
                            cfg.backtest.data_dir)
        if not bars:
            log.warning("no candles for %s", pair)
        return pair, bars

    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(one, pairs))


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _parse_date(text: str) -> int:
    day = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(day.timestamp() * 1000)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the strategy on Binance hourly candles.")
    parser.add_argument("--config", help="JSON file overriding config defaults")
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default from config)")
    parser.add_argument("--end", help="day after the last, YYYY-MM-DD (default from config)")
    parser.add_argument("--out", default=os.path.join("runs", "backtest"), help="output directory")
    parser.add_argument("--fixed-universe", action="store_true",
                        help="test strategy.universe as configured instead of applying the universe rule")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    start_ms, end_ms = _parse_date(start), _parse_date(end)
    warmup_start = start_ms - cfg.backtest.warmup_bars * HOUR_MS
    point_in_time = cfg.backtest.point_in_time_universe and not args.fixed_universe

    if point_in_time:
        roostoo = RoostooClient("", "", base_url=cfg.api.base_url)  # public endpoints only
        pairs = sorted(fetch_candidates(roostoo, cfg.universe))
    else:
        pairs = list(cfg.strategy.universe)
    bars = load_all(cfg, pairs, warmup_start, end_ms)
    monthly = point_in_time and cfg.backtest.refresh_universe_monthly
    if point_in_time:
        universe = select_universe(bars, start_ms, cfg.universe, cfg.strategy.defensive_pair)
        if cfg.strategy.regime_pair not in universe:
            universe.insert(0, cfg.strategy.regime_pair)
        cfg.strategy.universe = universe
        print("Universe: the %d most traded pairs as of %s, plus %s%s:\n  %s\n" % (
            cfg.universe.size, start, cfg.strategy.defensive_pair,
            ", refreshed on the first day of every month" if monthly else "",
            " ".join(p.split("/")[0] for p in universe)))
    if not monthly:
        bars = {p: bars.get(p, []) for p in cfg.strategy.universe}

    b = cfg.backtest
    results = [
        run_backtest(cfg, bars, start_ms, end_ms, b.taker_fee, b.taker_slippage, "taker", monthly),
        run_backtest(cfg, bars, start_ms, end_ms, b.maker_fee, 0.0, "maker", monthly),
    ]
    benchmarks = [
        buy_and_hold(bars, [cfg.strategy.regime_pair], start_ms, end_ms, b.initial_cash, "BTC hold"),
        buy_and_hold(bars, cfg.strategy.universe, start_ms, end_ms, b.initial_cash, "basket hold"),
    ]
    print(report(results + benchmarks, start, end))
    for result in results:
        write_outputs(result, args.out)
    print("\nEquity curves and trades written to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
