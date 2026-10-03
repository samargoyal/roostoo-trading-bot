"""Every tunable number in one place, shared by the backtester and the live bot.

The defaults are the baseline strategy. A JSON file passed with --config overrides
any subset of them, for example:

    {"strategy": {"max_positions_risk_on": 5}, "execution": {"limit_timeout_sec": 600}}

Unknown keys are rejected, so a typo cannot silently fall back to a default.
"""
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

# The 20 most traded crypto pairs on Roostoo over the 30 days to 2 October 2026, plus PAXG
# as the defensive asset. Chosen by the rule in universe.py; regenerate with
# `python -m bot.universe`.
DEFAULT_UNIVERSE = [
    "BTC/USD", "ETH/USD", "ZEC/USD", "SOL/USD", "XRP/USD", "NEAR/USD", "BNB/USD",
    "SUI/USD", "DOGE/USD", "UNI/USD", "ENA/USD", "AVAX/USD", "WLD/USD", "LINK/USD",
    "ADA/USD", "ARB/USD", "TAO/USD", "PUMP/USD", "LTC/USD", "TRX/USD",
    "PAXG/USD",
]


@dataclass
class StrategyConfig:
    universe: List[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))

    # Market regime: risk-on while the regime pair closes above its long EMA.
    regime_pair: str = "BTC/USD"
    regime_ema: int = 200
    risk_on_exposure: float = 0.75       # max fraction of equity invested when risk-on
    risk_off_exposure: float = 0.25      # ... and when risk-off
    max_positions_risk_on: int = 4
    max_positions_risk_off: int = 3

    # Defensive asset: always held as a small core, and first in line when risk-off.
    defensive_pair: str = "PAXG/USD"
    core_weight: float = 0.05

    # Trend filter and trend exit.
    fast_ema: int = 50
    slow_ema: int = 200

    # Ranking. "low_volatility" (default) prefers the coins with the calmest hourly returns
    # over volatility_window; "momentum" uses the two-horizon volatility-adjusted momentum
    # below. research/ shows why low volatility won (README: Strategy research).
    ranking: str = "low_volatility"
    momentum_short: int = 72
    momentum_long: int = 168
    momentum_short_weight: float = 0.5  # the long horizon gets 1 - this
    volatility_window: int = 168        # hours of log returns behind the volatility estimate

    # Entry timing.
    rsi_period: int = 14
    rsi_max_entry: float = 70.0

    # Sizing and trailing stop.
    atr_period: int = 14
    max_weight: float = 0.15            # per-coin cap; for the defensive pair it includes the core
    stop_atr_multiple: float = 8.0      # exit below the highest close since entry minus this many ATRs
    stop_cooldown_hours: int = 24       # no re-entry into a coin for this long after a stop

    # Portfolio drawdown brake.
    brake_drawdown: float = 0.04         # engage when equity is this far below its peak
    brake_release_drawdown: float = 0.02  # release once the drawdown is back under this
    brake_factor: float = 0.5            # scale trend positions by this while engaged

    # A holding below this fraction of equity counts as no position.
    min_position_weight: float = 0.005

    # Short sleeve, betting against beta: short the most volatile coins in the universe,
    # whatever their trend, sized by inverse ATR. 0 switches it off. Research (H13) found
    # 10-20% with an 8-12 ATR stop worked in both years and in the rally that followed;
    # 0.15 with a 10 ATR stop is the centre of that range. Off by default until Roostoo's
    # /v6 short endpoints have been tried on the testing account.
    short_exposure: float = 0.0
    max_shorts: int = 3
    short_rsi_min: float = 30.0          # no new shorts into oversold coins
    short_stop_atr_multiple: float = 10.0  # cover above the lowest close since entry plus this many ATRs

    # Momentum rotation sleeve (research H15-H16): this share of equity holds the coins with
    # the strongest positive return over rotation_lookback hours, equally weighted, chosen
    # again every rotation_rebalance_hours at 00:00 UTC. It is in the market only while the
    # regime pair's rotation_trend_fast EMA is above its rotation_trend_slow EMA, and leaves
    # at once when that fails; an empty slot goes to the defensive pair if its own return is
    # positive. The rest of equity runs the strategy above. 0 switches the sleeve off.
    rotation_weight: float = 0.4
    rotation_lookback: int = 336
    rotation_top: int = 2
    rotation_rebalance_hours: int = 24
    rotation_trend_fast: int = 168
    rotation_trend_slow: int = 672


@dataclass
class UniverseConfig:
    """How `python -m bot.universe` and the backtest choose the pairs to trade (see universe.py)."""
    size: int = 20                      # pairs picked by trading volume; the defensive pair is added
    volume_days: int = 30               # trading volume is measured over this many days
    max_spread: float = 0.001           # skip pairs whose bid-ask spread is wider (coarse tick sizes)
    min_history_bars: int = 1000        # hourly candles needed to warm up the indicators
    asset_type: str = "crypto"          # Roostoo's AssetType; tokenised stocks are excluded


@dataclass
class ExecutionConfig:
    rebalance_threshold: float = 0.04   # resize a held position only when this far off target (fraction of equity)
    min_trade_usd: float = 10.0         # never send an order smaller than this
    use_limit_orders: bool = True       # try a maker order at the touch before paying the taker fee
    limit_timeout_sec: int = 300        # then cancel it and send the remainder at market
    poll_interval_sec: int = 30
    market_on_stop: bool = True         # stop-loss exits go straight to market
    cash_buffer: float = 0.01           # leave this fraction of free cash unspent, for fees and rounding

    # Activity rule: make sure at least one order fills in every UTC-aligned block of this
    # many hours. Blocks of 8 hours guarantee two or more trades in any calendar day,
    # whatever time zone the organisers count days in.
    activity_block_hours: int = 8
    activity_trigger_hours_left: float = 2.0   # act when this little of the block remains
    activity_min_usd: float = 25.0


@dataclass
class ApiConfig:
    base_url: str = "https://mock-api.roostoo.com"
    timeout_sec: float = 10.0
    max_calls_per_minute: int = 20      # the organisers' limit is 30 per minute, counting every call
    max_retries: int = 3
    backoff_sec: float = 1.0


@dataclass
class LiveConfig:
    binance_url: str = "https://data-api.binance.vision"
    history_bars: int = 2500            # hourly candles behind the indicators (the 672h EMA needs plenty)
    bar_delay_sec: int = 60             # run this long after the hour, once the candle is final
    ticker_sample_sec: int = 300        # Roostoo price samples, the fallback if Binance is unreachable
    runs_dir: str = "runs"


@dataclass
class BacktestConfig:
    start: str = "2025-10-01"
    end: str = "2026-10-01"
    initial_cash: float = 100000.0
    taker_fee: float = 0.001
    maker_fee: float = 0.0005
    taker_slippage: float = 0.0002      # half-spread paid by a market order
    short_fee: float = 0.001            # Roostoo charges 0.1% to open and to close a short
    warmup_bars: int = 2500
    data_dir: str = "data"
    # Choose the universe with the universe rule as of the start of the test window, using
    # only data available then. False backtests strategy.universe as configured, which
    # flatters the result if that list was picked with hindsight.
    point_in_time_universe: bool = True
    refresh_universe_monthly: bool = True   # re-apply the rule on the first day of every month


@dataclass
class Config:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None) -> Config:
    """Defaults, overridden by the JSON file at `path` if one is given."""
    cfg = Config()
    if path:
        with open(path, encoding="utf-8") as f:
            apply_overrides(cfg, json.load(f))
    return cfg


def apply_overrides(target: Any, overrides: Dict[str, Any], prefix: str = "") -> None:
    names = {f.name for f in fields(target)}
    for key, value in overrides.items():
        if key not in names:
            raise ValueError("unknown config key: " + prefix + key)
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError("config section %s%s must be an object" % (prefix, key))
            apply_overrides(current, value, prefix + key + ".")
        else:
            setattr(target, key, value)
