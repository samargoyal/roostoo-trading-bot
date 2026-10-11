"""The sub-minute scalper: a separate process beside the hourly bot, every trade closed within a minute.

    python -m bot.scalper --account test --paper   # Binance prices, Roostoo's fees, no orders
    python -m bot.scalper --account test           # real orders (needs scalper.enabled)

Every poll_sec it reads Binance's best bid and ask for its pairs in one request (free, and outside
Roostoo's limit of 30 calls a minute). When a pair's mid has risen entry_move or more over the last
lookback_sec seconds, it buys trade_usd at market and rests a limit sell `target` above the fill.
If Binance's bid falls `stop` below the fill, max_hold_sec pass, or the hourly bot's window comes
(pause_before_hour_min before the hour to pause_after_hour_min after it, when the hourly bot cancels
every open order and trades), it cancels the limit and sells at market. One scalp at a time; it only
ever sells what it bought, so the hourly bot's holdings are left alone.

Before each entry it reads Roostoo's own price for the pair (1 call) and skips the scalp unless
Roostoo is within max_gap of Binance: it trades the move, never Roostoo's delay in copying it.

Roostoo calls: the price check, a market buy and the limit sell (3), a check of the limit when
Binance trades through it (1), or a cancel, a check and a market sell (3). Its own limiter (max_calls_per_minute,
8) plus the hourly bot's 20 keep both under 30. A realised loss of max_daily_loss_usd in a UTC day
stops it until the next day.

Research H125 and a check of a year of 1-minute bars found no edge inside a minute after Roostoo's
fees (a market buy and a maker exit cost 15 basis points; the typical 1-minute move is 2-6), so
expect it to lose roughly what it pays in fees.
"""
import argparse
import csv
import json
import logging
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Callable, Deque, Dict, Optional, Tuple

from bot.config import ScalperConfig, load_config
from bot.execution import PairRules, parse_rules, round_down, round_up, to_str
from bot.journal import setup_logging, utc_iso
from bot.live import code_version, config_path
from bot.market_data import BinanceClient, BinanceError, binance_symbol
from bot.roostoo import RoostooAuthError, RoostooClient, RoostooError

log = logging.getLogger("bot.scalper")

TAKER_FEE = 0.001
MAKER_FEE = 0.0005
DAY_MS = 86_400_000
Quotes = Dict[str, Tuple[float, float]]   # pair -> (bid, ask)


@dataclass
class Scalp:
    pair: str
    qty: str                  # coins held by this scalp (a Decimal string, for the state file)
    entry: float              # average fill price
    opened_ms: int
    target: float
    spent_usd: float          # USD paid, fees included
    target_id: Optional[int] = None
    received_usd: float = 0.0  # USD back from sells so far, after fees
    sold: str = "0"
    target_filled: float = 0.0  # what the target order had filled, and its USD, when last checked
    target_usd: float = 0.0


def paused(now_ms: int, cfg: ScalperConfig) -> bool:
    """True inside the hourly bot's window around each hour."""
    s = (now_ms // 1000) % 3600
    return s >= 3600 - cfg.pause_before_hour_min * 60 or s < cfg.pause_after_hour_min * 60


class Scalper:
    def __init__(self, cfg: ScalperConfig, rules: Dict[str, PairRules], feed: Callable[[], Quotes],
                 run_dir: str, client: Optional[RoostooClient] = None, paper: bool = False,
                 roostoo_quote: Optional[Callable[[str], Tuple[float, float]]] = None):
        self.cfg = cfg
        self.roostoo_quote = roostoo_quote
        self.rules = rules
        self.feed = feed
        self.client = client
        self.paper = paper or client is None
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.state_path = os.path.join(run_dir, "scalper.json")
        self.trips_path = os.path.join(run_dir, "scalps.csv")
        self.history: Dict[str, Deque[Tuple[int, float]]] = {p: deque() for p in cfg.pairs}
        self.open: Optional[Scalp] = None
        self.cooldown: Dict[str, int] = {}
        self.day = -1
        self.day_pnl = 0.0
        self.day_trades = 0
        self.last: Quotes = {}
        self.sell_failures = 0

    # ---- start-up --------------------------------------------------------------

    def recover(self, now_ms: int) -> None:
        """A scalp left open by an earlier run is closed at market (dropped in paper mode)."""
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path, encoding="utf-8") as f:
            saved = json.load(f).get("open")
        if not saved:
            return
        scalp = Scalp(**saved)
        if self.paper:
            log.warning("dropping the paper scalp left open in %s", scalp.pair)
            self._save()
            return
        log.warning("closing the scalp left open by an earlier run: %s %s", scalp.qty, scalp.pair)
        self.open = scalp
        self._close(now_ms, "restart")

    # ---- the loop ----------------------------------------------------------------

    def step(self, now_ms: int) -> None:
        try:
            quotes = self.feed()
        except BinanceError as exc:
            log.warning("Binance prices failed: %s", exc)
            quotes = {}
        self.last.update(quotes)
        for pair, (bid, ask) in quotes.items():
            h = self.history.setdefault(pair, deque())
            h.append((now_ms, (bid + ask) / 2))
            while h and h[0][0] < now_ms - 2 * self.cfg.lookback_sec * 1000:
                h.popleft()

        day = now_ms // DAY_MS
        if day != self.day:
            if self.day >= 0:
                log.info("UTC day done: %d scalps, %+.2f USD", self.day_trades, self.day_pnl)
            self.day, self.day_pnl, self.day_trades = day, 0.0, 0

        if self.open is not None:
            self._manage(now_ms)
        elif self._may_open(now_ms):
            pair = self._signal(now_ms)
            if pair is not None:
                self._enter(now_ms, pair)

    def _may_open(self, now_ms: int) -> bool:
        c = self.cfg
        return (not paused(now_ms, c) and self.day_pnl > -c.max_daily_loss_usd
                and self.day_trades < c.max_trades_per_day)

    def _signal(self, now_ms: int) -> Optional[str]:
        """The pair whose mid rose most over lookback_sec, if by entry_move or more."""
        best, best_move = None, self.cfg.entry_move
        since = now_ms - self.cfg.lookback_sec * 1000
        for pair in self.cfg.pairs:
            h = self.history.get(pair)
            if not h or self.cooldown.get(pair, 0) > now_ms or pair not in self.rules:
                continue
            then = next((mid for ts, mid in h if ts >= since), None)
            if then is None or h[0][0] > since + 2000:   # too little history to look back that far
                continue
            move = h[-1][1] / then - 1
            if move >= best_move:
                best, best_move = pair, move
        return best

    # ---- entries -----------------------------------------------------------------

    def _enter(self, now_ms: int, pair: str) -> None:
        c, rules = self.cfg, self.rules[pair]
        bid, ask = self.last[pair]
        qty = round_down(c.trade_usd / ask, rules.amount_decimals)
        if qty <= 0 or float(qty) * ask < rules.min_order_value:
            return
        self.cooldown[pair] = now_ms + c.cooldown_sec * 1000
        if self.roostoo_quote is not None:
            # No trading on Roostoo's lag: enter only when its price already shows Binance's.
            try:
                r_bid, r_ask = self.roostoo_quote(pair)
            except RoostooError as exc:
                log.warning("Roostoo price for %s failed: %s", pair, exc)
                return
            gap = (bid + ask) / (r_bid + r_ask) - 1
            if abs(gap) > c.max_gap:
                log.info("skipped %s: Roostoo is %+.1f bp from Binance", pair, -1e4 * gap)
                return
            ask = r_ask
        if self.paper:
            filled, avg, spent = float(qty), ask, float(qty) * ask * (1 + TAKER_FEE)
            held = qty
        else:
            try:
                d = self.client.place_order(pair, "BUY", to_str(qty), "MARKET")
            except RoostooError as exc:
                log.error("scalp buy failed: %s %s: %s", to_str(qty), pair, exc)
                return
            filled = float(d.get("FilledQuantity") or 0.0)
            avg = float(d.get("FilledAverPrice") or 0.0)
            if filled <= 0 or avg <= 0:
                log.warning("scalp buy did not fill: %s", d)
                return
            fee, fee_coin = float(d.get("CommissionChargeValue") or 0.0), str(d.get("CommissionCoin") or "USD")
            coin = pair.split("/")[0]
            spent = filled * avg + (fee if fee_coin == "USD" else 0.0)
            held = round_down(filled - (fee if fee_coin == coin else 0.0), rules.amount_decimals)
        target = float(round_up(avg * (1 + c.target), rules.price_decimals))
        self.open = Scalp(pair, to_str(held), avg, now_ms, target, spent)
        log.info("scalp in: %s %s @ %s, target %s", to_str(held), pair, avg, target)
        if not self.paper:
            try:
                d = self.client.place_order(pair, "SELL", to_str(held), "LIMIT", to_str(Decimal(repr(target))))
                self.open.target_id = d.get("OrderID")
                self._take(d, maker=True)
            except RoostooError as exc:
                log.error("scalp target order failed, selling at market: %s", exc)
                self._close(now_ms, "no target")
                return
        self._save()
        if self._done():
            self._finish(now_ms, "target")

    # ---- the open scalp ----------------------------------------------------------

    def _manage(self, now_ms: int) -> None:
        s, c = self.open, self.cfg
        bid = self.last.get(s.pair, (0.0, 0.0))[0]
        if bid > s.target:
            if self.paper:
                self._paper_sell(s.target, MAKER_FEE)
            elif s.target_id is not None:
                self._refresh()
            if self._done():
                self._finish(now_ms, "target")
                return
        why = ("stop" if 0 < bid <= s.entry * (1 - c.stop) else
               "time" if now_ms - s.opened_ms >= c.max_hold_sec * 1000 else
               "pause" if paused(now_ms, c) else
               "daily loss" if self.day_pnl <= -c.max_daily_loss_usd else None)
        if why:
            self._close(now_ms, why)

    def _close(self, now_ms: int, why: str) -> None:
        """Cancel the target and sell what is left at market."""
        s = self.open
        if self.paper:
            self._paper_sell(self.last.get(s.pair, (s.entry, s.entry))[0], TAKER_FEE)
            self._finish(now_ms, why)
            return
        if s.target_id is not None:
            try:
                self.client.cancel_order(order_id=s.target_id)
            except RoostooError as exc:
                log.warning("could not cancel scalp target %s: %s", s.target_id, exc)
            self._refresh()
            s.target_id = None
        left = Decimal(s.qty) - Decimal(s.sold)
        if left > 0:
            try:
                self._take(self.client.place_order(s.pair, "SELL", to_str(left), "MARKET"), maker=False)
            except RoostooError as exc:
                self.sell_failures += 1
                log.error("scalp sell failed (%d): %s %s: %s", self.sell_failures, to_str(left), s.pair, exc)
                if self.sell_failures < 3:
                    self._save()
                    return            # try again on the next step
                log.error("giving up on selling %s %s; the hourly bot's wallet reconciliation sees it",
                          to_str(left), s.pair)
        self._finish(now_ms, why)

    def _refresh(self) -> None:
        try:
            matched = self.client.query_order(order_id=self.open.target_id)
        except RoostooError as exc:
            log.warning("could not check scalp target %s: %s", self.open.target_id, exc)
            return
        if matched:
            self._take(matched[0], maker=True, cumulative=True)

    def _take(self, d: dict, maker: bool, cumulative: bool = False) -> None:
        """Book a sell's fill. A limit order's query reports its whole fill so far (cumulative)."""
        s = self.open
        filled = float(d.get("FilledQuantity") or 0.0)
        avg = float(d.get("FilledAverPrice") or 0.0)
        fee = float(d.get("CommissionChargeValue") or 0.0)
        if str(d.get("CommissionCoin") or "USD") != "USD":
            fee *= avg
        if maker:
            # the target order: replace what was booked for it before with its total so far
            s.received_usd += (filled * avg - fee) - s.target_usd
            s.sold = to_str(Decimal(s.sold) + Decimal(repr(filled)) - Decimal(repr(s.target_filled)))
            s.target_filled, s.target_usd = filled, filled * avg - fee
        else:
            s.received_usd += filled * avg - fee
            s.sold = to_str(Decimal(s.sold) + Decimal(repr(filled)))

    def _paper_sell(self, price: float, fee: float) -> None:
        s = self.open
        left = float(Decimal(s.qty) - Decimal(s.sold))
        s.received_usd += left * price * (1 - fee)
        s.sold = s.qty

    def _done(self) -> bool:
        return self.open is not None and Decimal(self.open.sold) >= Decimal(self.open.qty)

    def _finish(self, now_ms: int, why: str) -> None:
        s = self.open
        pnl = s.received_usd - s.spent_usd
        exit_price = s.received_usd / float(s.sold) if float(s.sold) > 0 else 0.0
        self.day_pnl += pnl
        self.day_trades += 1
        self.sell_failures = 0
        self.cooldown[s.pair] = now_ms + self.cfg.cooldown_sec * 1000
        log.info("scalp out (%s): %s %s, %+.2f USD (%+.1f bp), held %.0fs; today %d scalps, %+.2f USD",
                 why, s.qty, s.pair, pnl, 1e4 * pnl / s.spent_usd, (now_ms - s.opened_ms) / 1000,
                 self.day_trades, self.day_pnl)
        new = not os.path.exists(self.trips_path)
        with open(self.trips_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["opened", "closed", "pair", "qty", "entry", "exit", "how", "pnl_usd", "pnl_bp",
                            "hold_sec", "paper"])
            w.writerow([utc_iso(s.opened_ms), utc_iso(now_ms), s.pair, s.qty, s.entry, round(exit_price, 8), why,
                        round(pnl, 4), round(1e4 * pnl / s.spent_usd, 2), round((now_ms - s.opened_ms) / 1000, 1),
                        self.paper])
        self.open = None
        self._save()

    def _save(self) -> None:
        data = asdict(self.open) if self.open is not None else None
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"open": data}, f)
        os.replace(tmp, self.state_path)


def _stop(signum, frame):
    raise KeyboardInterrupt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the sub-minute scalper beside the hourly bot.")
    parser.add_argument("--account", required=True, help="label for this account's logs, e.g. test or comp")
    parser.add_argument("--paper", action="store_true", help="simulate fills on Binance prices; send no orders")
    parser.add_argument("--config", help="JSON file overriding config defaults (default: config/<account>.json)")
    args = parser.parse_args(argv)
    if not re.match(r"^[A-Za-z0-9_-]+$", args.account):
        parser.error("--account may only contain letters, digits, - and _")
    path = config_path(args.account, args.config)
    cfg = load_config(path)
    sc = cfg.scalper
    run_dir = os.path.join(cfg.live.runs_dir, args.account + "-scalper" + ("-paper" if args.paper else ""))
    setup_logging(os.path.join(run_dir, "logs"))
    signal.signal(signal.SIGTERM, _stop)
    log.info("starting scalper: account=%s paper=%s commit=%s config=%s settings=%s", args.account, args.paper,
             code_version(), path or "code defaults", json.dumps(asdict(sc), sort_keys=True))
    if not args.paper and not sc.enabled:
        log.error("scalper.enabled is false in the config: not trading (use --paper to simulate)")
        return 1
    binance = BinanceClient(cfg.live.binance_url, timeout_sec=3.0, max_retries=0)
    symbols = {binance_symbol(p): p for p in sc.pairs}

    def feed() -> Quotes:
        return {symbols[s]: q for s, q in binance.book_tickers(list(symbols)).items() if s in symbols}

    try:
        a = cfg.api
        settings = dict(base_url=a.base_url, timeout_sec=a.timeout_sec, max_calls_per_minute=sc.max_calls_per_minute,
                        max_retries=a.max_retries, backoff_sec=a.backoff_sec)
        # paper mode needs only public calls, so it runs without keys
        client = (RoostooClient("", "", **settings) if args.paper else RoostooClient.from_env(**settings))
        client.sync_clock()
        rules = parse_rules(client.exchange_info())
        def roostoo_quote(pair: str) -> Tuple[float, float]:
            q = client.ticker(pair)[pair]
            return float(q["MaxBid"]), float(q["MinAsk"])

        scalper = Scalper(sc, rules, feed, run_dir, None if args.paper else client, paper=args.paper,
                          roostoo_quote=roostoo_quote)
        scalper.recover(client.now_ms())
        synced = time.time()
        while True:
            started = time.time()
            if started - synced > 3600:       # signed requests fail beyond 60s of clock skew
                client.sync_clock()
                synced = started
            try:
                scalper.step(client.now_ms())
            except RoostooAuthError:
                raise
            except Exception:  # noqa: BLE001 - keep going, but leave a full traceback
                log.exception("scalper step failed")
            time.sleep(max(0.05, sc.poll_sec - (time.time() - started)))
    except RoostooAuthError as exc:
        log.error("authentication failed, check the API keys: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
