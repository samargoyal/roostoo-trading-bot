"""Every tunable number in one place, shared by the backtester and the live bot.

The defaults are the baseline strategy. A JSON file passed with --config overrides
any subset of them, for example:

    {"strategy": {"max_positions_risk_on": 5}, "execution": {"limit_timeout_sec": 600}}

Unknown keys are rejected, so a typo cannot silently fall back to a default.
"""
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

# The 45 most traded crypto pairs on Roostoo over the 30 days to 3 October 2026 (spread at most
# 0.1%, 1000+ hours of history), plus PAXG as the defensive asset. Chosen by the rule in
# universe.py; regenerate with `python -m bot.universe`. Six-year tests found the rotation
# sleeve needs this wide a net to catch the leaders (research/folds.py --universe).
DEFAULT_UNIVERSE = [
    "BTC/USD", "ETH/USD", "ZEC/USD", "SOL/USD", "XRP/USD", "NEAR/USD", "BNB/USD", "SUI/USD",
    "DOGE/USD", "UNI/USD", "ENA/USD", "AVAX/USD", "WLD/USD", "LINK/USD", "ADA/USD", "TAO/USD",
    "ARB/USD", "PUMP/USD", "LTC/USD", "TRX/USD", "ONDO/USD", "XLM/USD", "TRUMP/USD", "HBAR/USD",
    "AAVE/USD", "FIL/USD", "XPL/USD", "FET/USD", "ASTER/USD", "PENGU/USD", "DOT/USD", "APT/USD",
    "ICP/USD", "POL/USD", "CAKE/USD", "SEI/USD", "ZEN/USD", "TUT/USD", "VIRTUAL/USD",
    "PENDLE/USD", "CRV/USD", "FLOKI/USD", "EIGEN/USD", "WIF/USD", "PLUME/USD",
    "PAXG/USD",
]


@dataclass
class StrategyConfig:
    """The strategy. The first groups are what the bot trades with: the defaults run the
    rotation beside the defensive book, and config/comp.json switches the competition account's
    book to the long-short trend book and ranks its rotation on several horizons (the research
    options rotation_ranking "multi", rotation_horizons and rotation_horizon_weights; its
    previous settings are config/comp_r54b.json). Everything else under "Research options" is
    off by default and used only by research/ (see docs/research.md)."""
    universe: List[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))

    # ---- Both books ------------------------------------------------------------------
    # The regime pair (BTC) drives the trend filters; the defensive pair (PAXG) is the safe
    # haven. Positions under min_position_weight of equity count as none. Pairs Roostoo halts
    # are never traded; with plan_around_halts the strategy holds them as they are and plans
    # the rest around them (round 11).
    regime_pair: str = "BTC/USD"
    defensive_pair: str = "PAXG/USD"
    volatility_window: int = 168        # hours of log returns behind the volatility estimate
    atr_period: int = 14
    stop_cooldown_hours: int = 24       # no re-entry into a coin for this long after a stop
    min_position_weight: float = 0.005
    plan_around_halts: bool = True

    # ---- The momentum rotation: rotation_weight of equity -----------------------------
    # Holds the rotation_top coins with the strongest positive return over rotation_lookback
    # hours, equally weighted, chosen again every rotation_rebalance_hours at 00:00 UTC, while
    # the regime pair's rotation_trend_fast EMA is above its rotation_trend_slow EMA; it leaves
    # at once when that fails. An empty slot goes to the defensive pair if its own return is
    # positive, else cash. 70% since 2026-10-04, the user's choice of risk after the
    # optimisation in research/rotation_weight.py (which, with a 50% drawdown limit, gives 50%).
    rotation_weight: float = 0.7
    rotation_lookback: int = 336
    rotation_top: int = 2
    rotation_rebalance_hours: int = 24
    rotation_trend_fast: int = 168
    rotation_trend_slow: int = 672

    # ---- The book: the rest of equity -------------------------------------------------
    book_mode: str = "trend"              # "trend": the defensive book below; "long_short": the long-short
                                          # trend book (the competition account since 5 October 2026);
                                          # research: "hybrid" (the defensive book, but in a confirmed bear
                                          # market every coin in its own downtrend, short) and "overlay"
                                          # (the defensive book, its idle cash shorting downtrends)

    # The long-short trend book (book_mode "long_short", research rounds 50-64): every coin
    # long while its ls_trend fast EMA is above the slow one and short while below, each its
    # inverse-volatility share of all coins (shorts are 1x on Roostoo, so longs plus shorts never
    # exceed the book's share), shorts covered by a trailing stop and none where shorts are
    # crowded. config/comp.json: EMAs 240h/960h, 10-ATR stops, the funding filter.
    ls_trend: List[int] = field(default_factory=lambda: [168, 672])
    short_stop_atr: float = 0.0           # > 0: the book covers a short this many ATRs above its lowest close
                                          # since entry, and leaves the coin for stop_cooldown_hours
    short_exclude_external: bool = False  # no short (book or rotation basket) on a coin whose external score
                                          # is negative: live, its perpetual funding rate over
                                          # live.funding_hours (crowded shorts, round 54)

    # The defensive trend book (book_mode "trend", the default): low-volatility coins in
    # uptrends (fast EMA above slow, close above slow, RSI not above rsi_max_entry), sized by
    # equal risk contribution and capped at max_weight, with an ATR trailing stop, a market
    # regime (risk-on while the regime pair closes above its regime_ema EMA), a PAXG core and a
    # drawdown brake on the book's own value.
    regime_ema: int = 200
    risk_on_exposure: float = 0.75       # max fraction of equity invested when risk-on
    risk_off_exposure: float = 0.25      # ... and when risk-off
    max_positions_risk_on: int = 8
    max_positions_risk_off: int = 3
    core_weight: float = 0.05
    fast_ema: int = 50
    slow_ema: int = 200
    ranking: str = "low_volatility"
    momentum_short: int = 72
    momentum_long: int = 168
    momentum_short_weight: float = 0.5  # the long horizon gets 1 - this
    rsi_period: int = 14
    rsi_max_entry: float = 70.0
    max_weight: float = 0.15            # per-coin cap; for the defensive pair it includes the core
    sizing: str = "erc"                 # "erc" (equal risk contribution), "min_variance" or "inverse_atr";
                                        # the first two use rotation_cov_hours of returns, all capped at
                                        # max_weight. Six-year folds: erc with 8 positions (round 6)
    stop_atr_multiple: float = 8.0      # exit below the highest close since entry minus this many ATRs
    brake_drawdown: float = 0.04         # engage when equity is this far below its peak
    brake_release_drawdown: float = 0.02  # release once the drawdown is back under this
    brake_factor: float = 0.5            # scale trend positions by this while engaged

    # ==== Research options, all off by default (research/rounds.py, docs/research.md) ====

    # The short sleeve, betting against beta (H13): short the most volatile coins, sized by
    # inverse ATR, with a trailing stop. config/shorts.json turns it on.
    short_exposure: float = 0.0
    max_shorts: int = 3
    short_rsi_min: float = 30.0          # no new shorts into oversold coins
    short_stop_atr_multiple: float = 10.0  # cover above the lowest close since entry plus this many ATRs

    # Long-short book variants (rounds 50-64).
    ls_ensemble: List[List[int]] = field(default_factory=list)  # more [fast, slow] EMA pairs: each coin's
                                          # position is the mean of its trend signs over ls_trend and these
                                          # (a vote from -1 to 1), so mixed signals mean smaller positions
    ls_regime_aligned: bool = False       # longs only while the rotation's BTC filter is on, shorts only
                                          # while it is off
    ls_pairs: str = "all"                 # "btc": the regime pair alone
    ls_sides: str = "both"                # "long" or "short": that side alone (research diagnostics)
    ls_short_scale: float = 1.0           # the book's shorts at this fraction of their weight, the rest in cash
    ls_long_stop_atr: float = 0.0         # > 0: the book sells a long this many ATRs below its highest close
                                          # since entry and leaves the coin for stop_cooldown_hours (round 64)
    ls_short_trend: List[int] = field(default_factory=list)  # [fast, slow]: shorts follow this (faster) EMA
                                          # pair, longs ls_trend; flat while the two disagree (round 64)
    ls_absorb_rotation: float = 0.0       # > 0: while the rotation's BTC filter is off, this fraction of its
                                          # share runs the long-short book instead of PAXG or cash (round 61)
    ls_absorb_sma_hours: int = 0          # > 0: ... only while BTC is also below its simple average over this
                                          # many hours (a confirmed bear market, round 62)
    ls_idle_horizon: int = 0              # > 0: the long-short book's unused share goes to the defensive pair
                                          # while its return over this many hours is positive (round 60)
    ls_band: float = 0.0                  # > 0: a neutral zone, no position while the EMAs are closer than this
    ls_full_gap: float = 0.0              # > 0: size by trend strength, full size once the EMA gap reaches this
                                          # (the book then holds cash while trends are weak)
    short_vol_ratio: List[int] = field(default_factory=list)  # [recent, base] hours: the book's shorts shrink
                                          # by base / recent volatility of BTC when the recent one is higher
    short_entry_channel: int = 0          # > 0: the book shorts a coin only on a close below its lowest low of
                                          # this many hours, and covers above its highest high of
    short_exit_channel: int = 0           # this many (turtle rules); untimed slots stay in cash
    short_top_volume: int = 0             # > 0: the book shorts only the most traded this many coins
    short_entry_min_funding: float = 0.0  # > 0: a new short needs its external score (funding) at least this
                                          # high; one already open stays until it turns negative (round 63)
    short_entry_rsi_min: float = 0.0      # > 0: no new short while the coin's RSI is below this (round 63)
    short_min_funding_rank: float = 0.0   # > 0: the book shorts only coins whose external score (funding) ranks
                                          # at least this high among all coins: crowded longs (round 57)
    long_max_external: float = 0.0        # > 0: the long-short book holds no long whose external score (its
                                          # recent funding rate) is above this: crowded longs (round 56)
    short_regime_hours: int = 0           # > 0: shorts (long-short book, hybrid book, rotation basket) only in
                                          # a confirmed bear market: the rotation's BTC filter off and BTC
                                          # below its simple average over this many hours (4800: 200 days;
                                          # backtest.warmup_bars and live.history_bars must exceed it)

    # Defensive book variants.
    slow_filter: List[int] = field(default_factory=list)  # [fast, slow] EMA spans: a second, slower BTC
                                          # trend filter the sleeve must also pass (research round 49)
    regime_slow_filter: bool = False      # the book's risk-on also needs that slow filter (round 49)
    regime_band: float = 0.0              # > 0: hysteresis on the book's regime: off only below EMA x (1 - b),
                                          # on only above EMA x (1 + b) (research round 33)
    trend_exit_band: float = 0.0          # > 0: trend exit only when EMA fast < EMA slow x (1 - b) (round 33)
    regime_breadth: float = 0.0           # > 0: the book is risk-on while breadth is at least this (round 24)
    book_donchian_exit: int = 0           # > 0: the book exits below the previous N hours' low (42)
    book_excludes_rotation: bool = False  # the book does not enter coins the sleeve holds (round 36)

    # Rotation variants (rounds 20-64).
    rotation_attention_filter: bool = False  # research (round 65): the rotation skips a coin whose saved attention
    rotation_attention_floor: float = 0.0  # score ("ATT:" + pair, log of 7-day over 90-day Wikipedia views) is
                                          # below this floor; coins without one are never skipped
    rotation_attention_key: str = "ATT"   # round 66: "SHR" filters on the coin's share of all crypto attention
    rotation_attention_rank: float = 0.0  # > 0: rank candidates by the normal score of their return plus this
                                          # times that of their attention (none: 0); round 66
    rotation_exit_attention: bool = False  # round 66: a held pick leaves when its attention falls below
    rotation_exit_attention_floor: float = -0.51  # this (log; -0.51: 40% below normal), cooling down a day
    rotation_exit_spike: float = 0.0      # > 0: a held pick leaves on a day of this many times its usual views
    rotation_market_attention: bool = False  # round 66: the rotation is in the market only while attention to
    rotation_market_attention_floor: float = -0.22  # crypto as a whole is above this (log; -0.22: 20% below)
    rotation_exhaustion: bool = False     # round 69: skip a candidate whose trend shows exhaustion: within
    exhaustion_near_high: float = 0.97    # this fraction of its 7-day high with RSI below exhaustion_rsi
    exhaustion_rsi: float = 60.0          # (weak highs), or up over 3 days on less than exhaustion_volume of
    exhaustion_volume: float = 0.7        # the previous 3 days' dollar volume (volume divergence)
    rotation_btc_dip_hours: int = 0       # > 0: while BTC's return over this many hours to the last 00:00 UTC
    rotation_btc_dip_share: float = 0.5   # close is below zero, the picks keep this share of their weight (round 70)
    rotation_take_profit: float = 0.0     # > 0: once a pick is this far above its entry, keep only
    rotation_take_profit_keep: float = 0.5  # this share of it until it leaves the picks (round 68)
    rotation_profit_trail_after: float = 0.0  # > 0: once a pick has gained this much, a trailing stop this far
    rotation_profit_trail: float = 0.15   # below its high since entry protects the gain (round 68)
    rotation_dd_scale: bool = False       # round 68: the sleeve shrinks as the account falls from its high of
    rotation_dd_hours: int = 720          # the last this many hours: full size until rotation_dd_start below
    rotation_dd_start: float = 0.05       # it, then in proportion down to rotation_dd_min of its size at
    rotation_dd_full: float = 0.20        # rotation_dd_full below it
    rotation_dd_min: float = 0.25
    short_attention_max: float = 99.0     # < 99: the long-short book does not short a coin whose attention is
                                          # above this (log): no shorts into retail hype (round 66)
    rotation_max_external: float = 0.0    # > 0: the rotation skips a pick whose external score (funding) is
                                          # above this: crowded longs (round 57)
    rotation_core_share: float = 0.0      # share of the sleeve kept in the regime pair (BTC) while the
                                          # trend filter is on; the momentum slots share the rest
    rotation_weighting: str = "equal"     # how the picks share the sleeve: "equal", "inverse_vol",
                                          # "erc" (equal risk contribution) or "min_variance"
    rotation_max_weight: float = 1.0      # cap per pick, as a share of the filled sleeve
    rotation_cov_hours: int = 336         # hourly returns behind the covariance for erc / min_variance
    rotation_ranking: str = "return"      # "return", "residual": the return left after removing
                                          # the coin's beta to the regime pair (BTC) over the lookback,
                                          # "kalman": the Kalman trend slope over its standard deviation,
                                          # or "external" (research only): saved model scores
    rotation_exclude_external: bool = False  # research only: skip coins whose saved external score is
                                          # negative (research H30, funding-rate crowding)
    rotation_adaptive_lookbacks: List[int] = field(default_factory=list)  # [high-vol, normal]: the
                                          # lookback while BTC's 30-day volatility is above, or not, its
                                          # median over the last 60 days (research round 44)
    rotation_donchian_entry: int = 0      # > 0: picks must close at or above their highest high of the
                                          # previous N hours, a breakout (research round 42)
    rotation_donchian_exit: int = 0       # > 0: drop a pick that closes below its lowest low of the
                                          # previous N hours (round 42)
    rotation_donchian_hold: bool = False  # keep held picks until that exit instead of re-ranking (42)
    rotation_external_scale: bool = False  # research only: scale the sleeve by the saved "__scale__"
                                          # score of the day (round 41)
    rotation_regime_pair: str = ""        # set: the sleeve's trend filter reads this pair instead (round 38)
    rotation_min_age_hours: int = 0       # > 0: picks need this many hourly bars of history (round 39)
    rotation_top_volume: int = 0          # > 0: picks must rank in this many by 30-day dollar volume (40)
    rotation_corr_lambda: float = 0.0     # > 0: the second pick, among the top 5, maximises the normal score
                                          # of its return minus this x its correlation with the first (36)
    rotation_min_tstat: float = 0.0       # > 0: a pick's lookback log return over its volatility x sqrt(hours)
                                          # must reach this (round 36)
    rotation_trim_ratio: float = 0.0      # > 0: a held pick is not trimmed until it is this multiple of its
                                          # target weight (round 37)
    rotation_entry_every: int = 0         # > 0: when the filter turns on, re-plan at the next hour divisible
                                          # by this instead of waiting for the daily rebalance (round 35)
    rotation_filter_band: float = 0.0     # > 0: hysteresis on the sleeve's trend filter (round 33)
    rotation_ensemble: List[str] = field(default_factory=list)  # research rounds 31+: sub-sleeves, each
                                          # an equal share picking its own top coins: "ret:H" ranks by the
                                          # H-hour return (which must be positive), "multi:H1,H2,.." by the
                                          # summed normal scores of those returns (336h return positive)
    rotation_vs_btc: str = ""             # "btc" or "cash": a pick must beat BTC's return over the lookback
                                          # by rotation_btc_margin; empty slots go to BTC ("btc") or as
                                          # usual to PAXG or cash ("cash") (research round 21)
    rotation_btc_margin: float = 0.0
    rotation_horizons: List[int] = field(default_factory=list)  # rotation_ranking "multi": the
                                          # horizons (hours) whose return ranks are averaged
    rotation_horizon_weights: List[float] = field(default_factory=list)  # their weights (empty: equal);
                                          # config/comp.json: 7, 14 and 21 days weighted 2/1/1 (round 73)
    rotation_concentrate: float = 0.0     # > 0: one pick takes the whole sleeve when its return is at least
                                          # this multiple of the second's (research round 27)
    rotation_euphoria: float = 0.0        # > 0: halve the sleeve while BTC's lookback return is above this
                                          # (research round 28)
    rotation_pick_trend: str = ""         # "both", "close" or "ema": picks must be in their own uptrend
                                          # (close above, and/or fast EMA above, the slow EMA; round 29)
    rotation_exit_sma: int = 0            # > 0: the sleeve is also out while BTC closes below its simple
                                          # average over this many hours (research round 25)
    rotation_reentry_hours: int = 0       # > 0: the filter must have been on this long to re-enter (round 25)
    rotation_breadth: float = 0.0         # > 0: market breadth (share of coins with EMA fast > slow) needed
    rotation_breadth_mode: str = "and"    # for the sleeve: "and" with BTC's filter, or "only" (round 24)
    rotation_brake_drawdown: float = 0.0  # > 0: halve the sleeve while the account is this far below its
    rotation_brake_release: float = 0.0   # peak, until it is back within the release (research round 23)
    rotation_equity_ma_hours: int = 0     # > 0: sleeve only while the account is above its average over
                                          # this many hours (research round 23)
    rotation_buffer: int = 0              # > rotation_top: keep a held pick while it ranks in this many
                                          # (research round 22)
    rotation_min_hold_hours: int = 0      # > 0: keep a pick at least this long unless its return turns
                                          # negative or the filter exits (research round 22)
    rotation_stop_atr: float = 0.0        # > 0: drop a pick that closes this many ATRs below its highest
                                          # close since it was picked (research round 20)
    rotation_stop_pct: float = 0.0        # > 0: drop a pick that falls this fraction below that high
    rotation_shorts: int = 0              # > 0: while the trend filter is off, the sleeve shorts this many
                                          # coins instead (research H31; off: Roostoo /v6 untested)
    rotation_short_ranking: str = "return"  # "return": the weakest 2-week returns; "volatility": the wildest;
                                          # "trend_basket": every coin whose own 168h EMA is below its
                                          # 672h EMA, by inverse volatility (research H50)
    rotation_short_cap: float = 0.2       # "trend_basket": cap per coin, as a share of the sleeve
    rotation_vol_forecast: str = ""       # "har" or "ewma": scale the sleeve down when BTC's forecast daily
                                          # volatility is above its 60-day median (research H31)
    rotation_max_z: float = 0.0           # > 0: skip a pick whose close is more than this many standard
                                          # deviations above its mean over rotation_z_hours
    rotation_z_hours: int = 168
    rotation_cvar_limit: float = 0.0      # > 0: at each rebalance, scale the sleeve down so its 1-day
                                          # 95% CVaR (from the picks' last rotation_cov_hours of hourly
                                          # returns) is at most this share of total equity

    # Securing profits within the competition window (research/h70_secure_profits.py).
    # Once the account's gain since the window began exceeds secure_k times the portfolio's
    # daily volatility times the square root of the days left, every position is scaled to
    # secure_scale for the rest of the window: the gain is then more than a normal loss over
    # the remaining time could erase, so the score has more to lose than to win. The volatility
    # comes from the positions' last secure_vol_hours of hourly returns, measured once a day.
    # Windows of window_days start at window_start_ms (ms, UTC) and repeat, so a backtest
    # scores every 14 days; the first counts from window_start_equity when set (the
    # competition's $100,000), later ones from the equity at their start. Off while secure_k
    # is 0.
    secure_k: float = 0.0
    secure_mode: str = "half"           # "half": every position scaled to secure_scale, the rest in cash;
                                        # "half_gold": the same, the rest in the defensive pair; "refresh": the
                                        # rotation's coins are sold into the defensive pair and barred for the
                                        # rest of the window, so its next pick buys other coins. The others keep
                                        # the capital invested, moving the rotation's coins into the book
                                        # ("book"), BTC ("btc"), the defensive pair ("gold") or its top secure_top
                                        # ("spread")
    secure_scale: float = 0.5
    secure_top: int = 5
    secure_vol_hours: int = 720
    window_start_ms: int = 0
    window_days: int = 14
    window_start_equity: float = 0.0


@dataclass
class UniverseConfig:
    """How `python -m bot.universe` and the backtest choose the pairs to trade (see universe.py)."""
    size: int = 45                      # pairs picked by trading volume; the defensive pair is added
    volume_days: int = 30               # trading volume is measured over this many days
    max_spread: float = 0.001           # skip pairs whose bid-ask spread is wider (coarse tick sizes)
    min_history_bars: int = 1000        # hourly candles needed to warm up the indicators
    asset_type: str = "crypto"          # Roostoo's AssetType ("any" for all); tokenised stocks are excluded


@dataclass
class ExecutionConfig:
    rebalance_threshold: float = 0.04   # resize a held position only when this far off target (fraction of equity)
    rebalance_fraction: float = 1.0       # < 1: a plain rebalance moves only this share of the way to its
                                          # target; entries and exits go all the way (research round 43)
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
    funding_url: str = "https://fapi.binance.com"   # Binance USD-M futures, for funding rates
    funding_hours: int = 72             # with strategy.short_exclude_external: no short on a coin whose
                                        # funding averaged below zero over this many hours (round 54)


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
