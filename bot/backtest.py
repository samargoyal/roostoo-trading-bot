"""Backtest the live strategy on Binance hourly candles.

    python -m bot.backtest
    python -m bot.backtest --start 2025-10-01 --end 2026-10-01 --config my.json

The Strategy and plan_trades code is exactly what the live bot runs. Orders fill at
the hourly close, in two scenarios: every order as a taker (0.1% fee plus half a
spread of slippage), and every order as a maker (0.05%). Hourly bars cannot show
whether a resting limit order would have filled, so live costs land in between.
Candles are cached under data/; results are written to runs/backtest/.
"""
import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from bot.config import Config, load_config
from bot.market_data import HOUR_MS, Bar, BinanceClient, load_history
from bot.metrics import summarize
from bot.planner import SELL, plan_trades
from bot.strategy import Strategy, StrategyState

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


def run_backtest(cfg: Config, bars: Dict[str, List[Bar]], start_ms: int, end_ms: int,
                 fee: float, slippage: float, name: str = "strategy") -> Result:
    """Replay the strategy hour by hour from start_ms. Bars before start_ms only warm up indicators."""
    strategy = Strategy(cfg.strategy)
    state = StrategyState()
    cash = cfg.backtest.initial_cash
    holdings: Dict[str, float] = {}
    closes: Dict[str, float] = {}
    by_ts: Dict[int, List[Tuple[str, Bar]]] = {}
    for pair, series in bars.items():
        for bar in series:
            by_ts.setdefault(bar.ts, []).append((pair, bar))

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

        equity = cash + sum(q * closes[p] for p, q in holdings.items())
        weights = {p: q * closes[p] / equity for p, q in holdings.items()}
        decision = strategy.decide(now, equity, weights, state)
        planned = plan_trades(decision, weights, equity, now, state.last_fill_ts,
                              cfg.execution, cfg.strategy.min_position_weight)
        for t in planned:
            price = closes.get(t.pair)
            if not price:
                continue
            if t.side == SELL:
                held = holdings.get(t.pair, 0.0)
                quantity = held if t.close_position else min(held, t.usd / price)
                if quantity <= 0:
                    continue
                fill = price * (1.0 - slippage)
                proceeds = quantity * fill
                paid = proceeds * fee
                cash += proceeds - paid
                holdings[t.pair] = held - quantity
            else:
                spend = min(t.usd, cash / (1.0 + fee))
                if spend < cfg.execution.min_trade_usd:
                    continue
                fill = price * (1.0 + slippage)
                quantity = spend / fill
                paid = spend * fee
                cash -= spend + paid
                holdings[t.pair] = holdings.get(t.pair, 0.0) + quantity
            trades.append(Trade(now, t.pair, t.side, quantity, fill, paid, t.reason))
            state.last_fill_ts = now

        holdings = {p: q for p, q in holdings.items() if q > 1e-12}
        invested = sum(q * closes[p] for p, q in holdings.items())
        equity = cash + invested
        strategy.reconcile(now, {p: q * closes[p] / equity for p, q in holdings.items()},
                           decision, state)
        curve.append((now, equity))
        risk_on_hours += decision.risk_on
        exposure_sum += invested / equity

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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    start = args.start or cfg.backtest.start
    end = args.end or cfg.backtest.end
    start_ms, end_ms = _parse_date(start), _parse_date(end)
    warmup_start = start_ms - cfg.backtest.warmup_bars * HOUR_MS

    client = BinanceClient(cfg.live.binance_url)
    bars = {}
    for pair in cfg.strategy.universe:
        bars[pair] = load_history(client, pair, warmup_start, end_ms, cfg.backtest.data_dir)
        if not bars[pair]:
            log.warning("no candles for %s", pair)

    b = cfg.backtest
    results = [
        run_backtest(cfg, bars, start_ms, end_ms, b.taker_fee, b.taker_slippage, "taker"),
        run_backtest(cfg, bars, start_ms, end_ms, b.maker_fee, 0.0, "maker"),
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
