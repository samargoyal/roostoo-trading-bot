"""The live trading loop.

    python -m bot.live --account test              # trade the testing account
    python -m bot.live --account test --dry-run    # everything except sending orders
    python -m bot.live --account test --once       # one cycle, then exit

Credentials come only from ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY; scripts/run_bot.sh
loads them from ~/.roostoo_<account>.env. Each account keeps its own logs, journal and
state under runs/<account>/ (runs/<account>-dry/ in dry-run mode).

Every hour, a minute after the candle closes, the bot:
  1. fetches the last 1000 closed hourly candles per pair from Binance and rebuilds the
     indicators, falling back to bars built from Roostoo prices if Binance is down;
  2. reads the Roostoo wallet and ticker and values the portfolio;
  3. asks the strategy for target weights and the planner for trades;
  4. executes them: limit at the touch first, market after a timeout;
  5. reconciles the strategy state with the new wallet and records everything.

On start-up, and at the start of every cycle, it cancels any order a previous run left
open, and it always works from the real wallet, so it never assumes it starts flat. It
re-reads Roostoo's trading rules every cycle and sends no orders in a pair that is halted (or
delisted), instead of failing, and the strategy plans the rest of the portfolio around it
(strategy.plan_around_halts).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import signal
import subprocess
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from bot.config import Config, load_config
from bot.execution import Executor, OrderResult, halted_pairs, parse_rules
from bot.indicators import Signal
from bot.journal import Journal, StateStore, setup_logging, utc_iso
from bot.market_data import HOUR_MS, Bar, BinanceClient, BinanceError, binance_symbol
from bot.planner import PlannedTrade, plan_trades
from bot.roostoo import RoostooAuthError, RoostooClient, RoostooError
from bot.strategy import Decision, Strategy, StrategyState

log = logging.getLogger("bot.live")


class PriceSampler:
    """Hourly bars built from Roostoo ticker samples: the fallback when Binance is unreachable."""

    def __init__(self, keep_hours: int = 48):
        self.keep_hours = keep_hours
        self.bars: Dict[str, Dict[int, List[float]]] = {}

    def add(self, quotes: Dict[str, Dict[str, float]], ts_ms: int) -> None:
        hour = ts_ms // HOUR_MS * HOUR_MS
        for pair, quote in quotes.items():
            price = float(quote.get("LastPrice") or 0.0)
            if price <= 0:
                continue
            hours = self.bars.setdefault(pair, {})
            bar = hours.get(hour)
            if bar is None:
                hours[hour] = [price, price, price, price]
            else:
                bar[1] = max(bar[1], price)
                bar[2] = min(bar[2], price)
                bar[3] = price
            for old in [h for h in hours if h < hour - self.keep_hours * HOUR_MS]:
                del hours[old]

    def closed_bars(self, pair: str, after_ts: Optional[int], now_ms: int) -> List[Bar]:
        hours = self.bars.get(pair, {})
        return [Bar(h, hours[h][0], hours[h][1], hours[h][2], hours[h][3], 0.0)
                for h in sorted(hours)
                if h + HOUR_MS <= now_ms and (after_ts is None or h > after_ts)]


def portfolio_value(wallet: Dict[str, Dict[str, float]], quotes: Dict[str, Dict[str, float]],
                    shorts: Sequence[Dict[str, float]] = ()) -> Tuple[float, float, Dict[str, float]]:
    """Equity, cash and the signed USD exposure of each position, marked at the last price.

    A short counts towards equity by what closing it would return: its collateral plus its
    unrealised profit or loss. Its exposure is negative: -quantity x price.
    """
    usd = wallet.get("USD", {})
    cash = float(usd.get("Free") or 0.0) + float(usd.get("Lock") or 0.0)
    values = {}
    for coin, balance in wallet.items():
        if coin == "USD":
            continue
        amount = float(balance.get("Free") or 0.0) + float(balance.get("Lock") or 0.0)
        if amount <= 0:
            continue
        quote = quotes.get(coin + "/USD")
        if not quote:
            log.warning("no price for %s, leaving it out of equity", coin)
            continue
        values[coin + "/USD"] = amount * float(quote["LastPrice"])
    equity = cash + sum(values.values())
    for position in shorts:
        pair, qty = position.get("Pair"), float(position.get("ShortQty") or 0.0)
        quote = quotes.get(pair)
        if not pair or qty <= 0 or not quote:
            continue
        price = float(quote["LastPrice"])
        entry = float(position.get("EntryPrice") or 0.0)
        collateral = float(position.get("Collateral") or 0.0)
        equity += collateral + qty * (entry - price)
        values[pair] = values.get(pair, 0.0) - qty * price
    return equity, cash, values


class LiveBot:
    def __init__(self, cfg: Config, run_dir: str, dry_run: bool = False,
                 client: Optional[RoostooClient] = None, binance: Optional[BinanceClient] = None,
                 sleep: Callable[[float], None] = time.sleep, funding: Optional[BinanceClient] = None):
        self.cfg = cfg
        self.dry_run = dry_run
        a = cfg.api
        self.client = client or RoostooClient.from_env(
            base_url=a.base_url, timeout_sec=a.timeout_sec, max_calls_per_minute=a.max_calls_per_minute,
            max_retries=a.max_retries, backoff_sec=a.backoff_sec)
        self.binance = binance or BinanceClient(cfg.live.binance_url)
        s = cfg.strategy
        uses_funding = (s.short_exclude_external or s.long_max_external > 0 or s.short_min_funding_rank > 0
                        or s.rotation_max_external > 0)
        self.funding = ((funding or BinanceClient(cfg.live.funding_url, timeout_sec=5.0, max_retries=1))
                        if uses_funding else None)
        self.funding_table: Dict[int, Dict[str, float]] = {}
        self.journal = Journal(os.path.join(run_dir, "journal"))
        self.store = StateStore(os.path.join(run_dir, "state.json"))
        self.sampler = PriceSampler()
        self.history: Dict[str, List[Bar]] = {}
        self.sleep = sleep
        self.state = StrategyState()
        self.executor: Optional[Executor] = None
        self.frozen: Set[str] = set()   # pairs Roostoo is not trading, as of the last check

    # ---- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self.client.sync_clock()
        info = self.client.exchange_info()
        self.executor = Executor(self.client, parse_rules(info), self.cfg.execution, self._record_order,
                                 dry_run=self.dry_run, sleep=self.sleep)
        self.frozen = halted_pairs(info, self.cfg.strategy.universe)
        if self.frozen:
            log.warning("not tradable on Roostoo now (no orders until they are): %s",
                        ", ".join(sorted(self.frozen)))
        self.state = self.store.load() or StrategyState()
        self.state.peak_equity = max(self.state.peak_equity, self.journal.peak_equity())
        wallet = self.client.balance()
        held = {coin: b for coin, b in wallet.items()
                if float(b.get("Free") or 0) + float(b.get("Lock") or 0) > 0}
        log.info("started (dry_run=%s), wallet: %s, remembered positions: %s",
                 self.dry_run, json.dumps(held, sort_keys=True), sorted(self.state.positions))

    def run_forever(self) -> None:
        while True:
            try:
                self.cycle()
            except RoostooAuthError:
                raise
            except Exception:  # noqa: BLE001 - keep trading next hour, but leave a full traceback
                log.exception("cycle failed; trying again next hour")
            self.wait_for_next_cycle()

    def wait_for_next_cycle(self) -> None:
        """Sleep until just after the next hourly close, sampling Roostoo prices meanwhile."""
        next_run = (self.client.now_ms() // HOUR_MS + 1) * HOUR_MS + self.cfg.live.bar_delay_sec * 1000
        next_sample = 0
        while True:
            now = self.client.now_ms()
            if now >= next_run:
                return
            if now >= next_sample:
                try:
                    self.sampler.add(self.client.ticker(), now)
                except RoostooError as exc:
                    log.warning("ticker sample failed: %s", exc)
                next_sample = now + self.cfg.live.ticker_sample_sec * 1000
            self.sleep(min(15.0, max((next_run - now) / 1000.0, 0.1)))

    # ---- one hourly cycle ------------------------------------------------------

    def cycle(self) -> None:
        self.client.sync_clock()
        if not self.dry_run and self.client.pending_count():
            log.warning("cancelling orders left open: %s", self.client.cancel_order())

        started = self.client.now_ms()
        strategy = Strategy(self.cfg.strategy, external_scores=self._funding_scores(started))
        universe = self.cfg.strategy.universe
        # Binance candles for ~46 pairs, a few pages each: fetched six at a time.
        with ThreadPoolExecutor(max_workers=6) as pool:
            history = dict(zip(universe, pool.map(lambda p: self._bars(p, started), universe)))
        for pair in universe:
            for bar in history[pair]:
                strategy.update(pair, bar)
        signals = strategy.signals()
        missing = [p for p in self.cfg.strategy.universe if p not in signals]
        if missing:
            log.warning("no signal for %s; their positions are held as they are", ", ".join(missing))

        quotes = self.client.ticker()
        self.sampler.add(quotes, started)
        equity, cash, values = portfolio_value(self.client.balance(), quotes, self._shorts())
        weights = {p: v / equity for p, v in values.items()} if equity > 0 else {}

        frozen = self._refresh_rules(set(universe) | set(weights))
        now = self.client.now_ms()
        decision = strategy.decide(now, equity, weights, self.state,
                                   frozen if self.cfg.strategy.plan_around_halts else frozenset())
        trades = plan_trades(decision, weights, equity, now, self.state.last_fill_ts,
                             self.cfg.execution, self.cfg.strategy.min_position_weight, frozen)
        self._record_decision(now, decision, equity, weights, trades, signals)

        results = self.executor.execute(trades, quotes) if trades else []
        if any(r.filled > 0 for r in results):
            self.state.last_fill_ts = self.client.now_ms()
        if results and not self.dry_run:
            quotes = self.client.ticker()
            equity, cash, values = portfolio_value(self.client.balance(), quotes, self._shorts())
            weights = {p: v / equity for p, v in values.items()} if equity > 0 else {}

        strategy.reconcile(now, weights, decision, self.state)
        self._record_equity(equity, cash, values, decision)
        self.store.save(self.state)

    def _funding_scores(self, now_ms: int) -> Dict[int, Dict[str, float]]:
        """{day: {pair: mean funding rate over live.funding_hours up to that day's 00:00}}, for the
        day the strategy looks up, fetched once a day. A coin Binance has no perpetual for, or
        cannot be reached for, is left out, which leaves its shorts allowed."""
        if self.funding is None:
            return {}
        day = (now_ms - HOUR_MS) // (24 * HOUR_MS) * (24 * HOUR_MS)
        if day not in self.funding_table:
            start = day - self.cfg.live.funding_hours * HOUR_MS + 1
            table: Dict[str, float] = {}
            failures = 0
            for pair in self.cfg.strategy.universe:
                try:
                    prints = self.funding.funding_rates(binance_symbol(pair), start, day)
                except BinanceError as exc:
                    log.debug("no funding rates for %s: %s", pair, exc)
                    failures += 1
                    if failures >= 3 and not table:
                        # Unreachable rather than a few coins without perpetuals: stop for today.
                        log.warning("funding rates unavailable (%s); no crowding filter today", exc)
                        break
                    continue
                if prints:
                    table[pair] = sum(rate for _, rate in prints) / len(prints)
            self.funding_table = {day: table}
            crowded = sorted(p for p, rate in table.items() if rate < 0)
            log.info("funding rates for %d pairs (mean over %dh to %s); shorts crowded, so not shorted: %s",
                     len(table), self.cfg.live.funding_hours, utc_iso(day), ", ".join(crowded) or "none")
        return self.funding_table

    def _refresh_rules(self, pairs: Iterable[str]) -> Set[str]:
        """Re-read Roostoo's trading rules; returns the given pairs it is not trading now."""
        try:
            info = self.client.exchange_info()
        except RoostooError as exc:
            log.warning("exchange info unavailable (%s); keeping the last trading rules", exc)
            return self.frozen
        if not info.get("TradePairs"):
            log.warning("exchange info lists no pairs; keeping the last trading rules")
            return self.frozen
        self.executor.rules = parse_rules(info)
        frozen = halted_pairs(info, pairs)
        if frozen != self.frozen:
            log.warning("not tradable on Roostoo now: %s", ", ".join(sorted(frozen)) or "none")
        self.frozen = frozen
        return frozen

    def _shorts(self) -> List[Dict[str, float]]:
        """Open shorts, read only when the sleeve is on or the state remembers a short."""
        s = self.cfg.strategy
        if (s.short_exposure <= 0 and s.rotation_shorts <= 0 and s.book_mode == "trend"
                and not self.state.shorts):
            return []
        return self.client.short_positions()

    def _bars(self, pair: str, now_ms: int) -> List[Bar]:
        try:
            bars = self.binance.recent_closed(binance_symbol(pair), self.cfg.live.history_bars, now_ms)
        except BinanceError as exc:
            cached = self.history.get(pair, [])
            extra = self.sampler.closed_bars(pair, cached[-1].ts if cached else None, now_ms)
            log.warning("Binance candles unavailable for %s (%s); using %d cached and %d "
                        "Roostoo-built bars", pair, exc, len(cached), len(extra))
            return cached + extra
        expected = now_ms // HOUR_MS * HOUR_MS - HOUR_MS
        if bars and bars[-1].ts < expected:
            log.warning("latest %s candle opened %s, expected %s", pair, utc_iso(bars[-1].ts),
                        utc_iso(expected))
        self.history[pair] = bars
        return bars

    # ---- records ---------------------------------------------------------------

    def _record_order(self, order: OrderResult) -> None:
        self.journal.order(order.as_row())

    def _record_decision(self, now: int, decision: Decision, equity: float,
                         weights: Dict[str, float], trades: List[PlannedTrade],
                         signals: Dict[str, Signal]) -> None:
        targets = {p: round(w, 4) for p, w in decision.targets.items() if w > 0}
        log.info("equity %.2f, %s, exposure limit %.0f%%, drawdown %.2f%%%s; targets %s; %d trade(s)",
                 equity, "risk-on" if decision.risk_on else "risk-off", decision.exposure_limit * 100,
                 decision.drawdown * 100, ", BRAKE ON" if decision.brake_on else "", targets, len(trades))
        self.journal.decision({
            "time": utc_iso(now),
            "equity": round(equity, 2),
            "risk_on": decision.risk_on,
            "exposure_limit": decision.exposure_limit,
            "drawdown": round(decision.drawdown, 5),
            "brake_on": decision.brake_on,
            "weights": {p: round(w, 5) for p, w in weights.items()},
            "targets": targets,
            "reasons": decision.reasons,
            "halted": sorted(self.frozen),
            "scores": {p: round(s, 4) for p, s in decision.scores.items()},
            "signals": {p: {"bar": utc_iso(s.ts), "close": s.close, "ema_fast": round(s.ema_fast, 6),
                            "ema_slow": round(s.ema_slow, 6), "ema_regime": round(s.ema_regime, 6),
                            "atr": round(s.atr, 6), "rsi": round(s.rsi, 2)}
                        for p, s in signals.items()},
            "trades": [{"pair": t.pair, "side": t.side, "usd": round(t.usd, 2), "reason": t.reason}
                       for t in trades],
            "dry_run": self.dry_run,
        })

    def _record_equity(self, equity: float, cash: float, values: Dict[str, float],
                       decision: Decision) -> None:
        peak = max(self.state.peak_equity, equity)
        self.journal.equity({
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "invested": round(sum(abs(v) for v in values.values()), 2),
            "drawdown": round(1.0 - equity / peak, 5) if peak > 0 else 0.0,
            "peak_equity": round(peak, 2),
            "risk_on": decision.risk_on,
            "brake_on": self.state.brake_on,
            "positions": json.dumps({p: round(v, 2) for p, v in sorted(values.items())}),
        })


def _stop(signum, frame):
    raise KeyboardInterrupt


def code_version(directory: Optional[str] = None) -> str:
    """The git commit the bot runs from, marked if files differ from it, so every log and
    decision can be traced to the exact code behind it ("unknown" outside a git checkout)."""
    directory = directory or os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=directory, capture_output=True,
                              text=True, timeout=10)
        if head.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=directory,
                               capture_output=True, text=True, timeout=10)
        return head.stdout.strip() + (" (with local changes)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def config_path(account: str, explicit: Optional[str] = None, directory: str = "config") -> Optional[str]:
    """The config file to run with: the one given, else config/<account>.json if it exists, so
    each account's settings live in the repository and a restart picks up committed changes."""
    if explicit:
        return explicit
    default = os.path.join(directory, account + ".json")
    return default if os.path.exists(default) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Roostoo trading bot.")
    parser.add_argument("--account", required=True,
                        help="label for this account's logs and state, e.g. test or comp")
    parser.add_argument("--dry-run", action="store_true", help="do everything except send orders")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--config", help="JSON file overriding config defaults (default: config/<account>.json "
                                         "if it exists)")
    args = parser.parse_args(argv)
    if not re.match(r"^[A-Za-z0-9_-]+$", args.account):
        parser.error("--account may only contain letters, digits, - and _")

    path = config_path(args.account, args.config)
    cfg = load_config(path)
    run_dir = os.path.join(cfg.live.runs_dir, args.account + ("-dry" if args.dry_run else ""))
    setup_logging(os.path.join(run_dir, "logs"))
    signal.signal(signal.SIGTERM, _stop)
    log.info("starting: account=%s dry_run=%s once=%s commit=%s config=%s", args.account, args.dry_run,
             args.once, code_version(), path or "code defaults")
    log.info("config: %s", json.dumps(cfg.to_dict(), sort_keys=True))
    try:
        bot = LiveBot(cfg, run_dir, dry_run=args.dry_run)
        bot.start()
        if args.once:
            bot.cycle()
        else:
            bot.run_forever()
    except RoostooAuthError as exc:
        log.error("authentication failed, check the API keys: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
