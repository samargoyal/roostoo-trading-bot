"""Which pairs to trade: the most actively traded crypto pairs on Roostoo.

The rule: among Roostoo pairs whose AssetType is crypto, whose bid-ask spread is at most
max_spread, and which have enough Binance history to warm up the indicators, keep the
`size` pairs with the highest USD trading volume over the last `volume_days` days. The
defensive pair (PAXG) is always added.

The rule uses only information available when it is applied, so the backtest can apply
it as of the start of its test window. Picking coins by how well they did in the past
would flatter any backtest: the project brief's hand-picked list owed its whole profit
in 2025-26 to one coin (ZEC) that had already rallied.

    python -m bot.universe     # rank today's candidates; paste the result into config.py
"""
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from bot.config import UniverseConfig, load_config
from bot.market_data import HOUR_MS, Bar, BinanceClient, BinanceError, binance_symbol
from bot.roostoo import RoostooClient

log = logging.getLogger(__name__)
DAY_MS = 24 * HOUR_MS


def candidate_pairs(exchange_info: Dict, quotes: Dict[str, Dict[str, float]],
                    cfg: UniverseConfig) -> Dict[str, float]:
    """Tradable pairs of the right asset type with a tight enough spread, with that spread."""
    out = {}
    for pair, info in exchange_info.get("TradePairs", {}).items():
        quote = quotes.get(pair)
        if not quote or not info.get("CanTrade", True) or info.get("AssetType") != cfg.asset_type:
            continue
        bid, ask = float(quote.get("MaxBid") or 0), float(quote.get("MinAsk") or 0)
        if bid <= 0 or ask <= 0:
            continue
        spread = (ask - bid) / ((ask + bid) / 2)
        if spread <= cfg.max_spread:
            out[pair] = spread
    return out


def usd_volume(bars: Sequence[Bar], start_ms: int, end_ms: int) -> float:
    return sum(b.volume * b.close for b in bars if start_ms <= b.ts < end_ms)


def rank_by_volume(bars: Dict[str, List[Bar]], at_ms: int,
                   cfg: UniverseConfig) -> List[Tuple[str, float]]:
    """Pairs with enough history before at_ms, ranked by USD volume over the preceding days.

    History is judged by the first candle's date rather than a candle count, so a few hours
    of exchange downtime cannot knock a long-listed coin out.
    """
    since = at_ms - cfg.volume_days * DAY_MS
    listed_by = at_ms - cfg.min_history_bars * HOUR_MS
    ranked = []
    for pair, series in bars.items():
        if series and series[0].ts <= listed_by:
            ranked.append((pair, usd_volume(series, since, at_ms)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def select_universe(bars: Dict[str, List[Bar]], at_ms: int, cfg: UniverseConfig,
                    defensive_pair: str) -> List[str]:
    """The `size` most traded pairs as of at_ms, plus the defensive pair."""
    ranked = [p for p, _ in rank_by_volume(bars, at_ms, cfg) if p != defensive_pair]
    chosen = ranked[:cfg.size]
    if defensive_pair in bars:
        chosen.append(defensive_pair)
    return chosen


def fetch_candidates(client: RoostooClient, cfg: UniverseConfig) -> Dict[str, float]:
    return candidate_pairs(client.exchange_info(), client.ticker(), cfg)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rank Roostoo pairs by the universe rule.")
    parser.add_argument("--config", help="JSON file overriding config defaults")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    cfg = load_config(args.config)
    u = cfg.universe

    roostoo = RoostooClient("", "", base_url=cfg.api.base_url)  # public endpoints only
    spreads = fetch_candidates(roostoo, u)
    binance = BinanceClient(cfg.live.binance_url)

    def history(pair: str) -> Tuple[str, List[Bar]]:
        try:
            return pair, binance.recent_closed(binance_symbol(pair), u.min_history_bars)
        except BinanceError as exc:
            log.warning("skipping %s: %s", pair, exc)
            return pair, []

    with ThreadPoolExecutor(max_workers=6) as pool:
        bars = dict(pool.map(history, sorted(spreads)))
    now = roostoo.server_time()
    ranked = rank_by_volume(bars, now, u)
    chosen = select_universe(bars, now, u, cfg.strategy.defensive_pair)

    print("%d candidates: %s pairs, spread <= %.2f%%, %d+ hours of history\n"
          % (len(ranked), u.asset_type, u.max_spread * 100, u.min_history_bars))
    print("%4s  %-14s %16s %9s" % ("rank", "pair", "%dd volume, $M" % u.volume_days, "spread"))
    for i, (pair, volume) in enumerate(ranked, 1):
        mark = "  <- chosen" if pair in chosen else ""
        print("%4d  %-14s %16.0f %8.3f%%%s" % (i, pair, volume / 1e6, spreads[pair] * 100, mark))
    print("\nuniverse = %r" % chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
