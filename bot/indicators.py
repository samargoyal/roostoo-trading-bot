"""Technical indicators, updated one closed bar at a time.

Each indicator returns None until it has seen enough data. The backtester and the
live bot feed the same bars through the same objects, so their signals match.
"""
import math
from collections import deque
from typing import Deque, NamedTuple, Optional

from bot.market_data import Bar


class EMA:
    """Exponential moving average, seeded with the simple average of the first `period` values."""

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1)
        self.value: Optional[float] = None
        self._seed = []

    def update(self, x: float) -> Optional[float]:
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
                self._seed = []
        else:
            self.value += self.alpha * (x - self.value)
        return self.value


class WilderAverage:
    """Wilder's smoothing (alpha = 1/period), seeded with a simple average. Used by ATR and RSI."""

    def __init__(self, period: int):
        self.period = period
        self.value: Optional[float] = None
        self._seed = []

    def update(self, x: float) -> Optional[float]:
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
                self._seed = []
        else:
            self.value += (x - self.value) / self.period
        return self.value


class ATR:
    """Average true range."""

    def __init__(self, period: int):
        self.average = WilderAverage(period)
        self.prev_close: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        return self.average.value

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self.prev_close is None:
            true_range = high - low
        else:
            true_range = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.prev_close = close
        return self.average.update(true_range)


class RSI:
    """Relative strength index (Wilder)."""

    def __init__(self, period: int):
        self.gains = WilderAverage(period)
        self.losses = WilderAverage(period)
        self.prev_close: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, close: float) -> Optional[float]:
        if self.prev_close is not None:
            change = close - self.prev_close
            gain = self.gains.update(max(change, 0.0))
            loss = self.losses.update(max(-change, 0.0))
            if gain is not None and loss is not None:
                if loss == 0:
                    self.value = 100.0 if gain > 0 else 50.0
                else:
                    self.value = 100.0 - 100.0 / (1.0 + gain / loss)
        self.prev_close = close
        return self.value


class RollingStd:
    """Sample standard deviation of the last `window` values."""

    def __init__(self, window: int):
        self.window = window
        self.values: Deque[float] = deque()
        self.total = 0.0
        self.total_sq = 0.0

    @property
    def value(self) -> Optional[float]:
        n = len(self.values)
        if n < self.window or n < 2:
            return None
        variance = (self.total_sq - self.total * self.total / n) / (n - 1)
        return math.sqrt(variance) if variance > 0 else 0.0

    def update(self, x: float) -> Optional[float]:
        self.values.append(x)
        self.total += x
        self.total_sq += x * x
        if len(self.values) > self.window:
            old = self.values.popleft()
            self.total -= old
            self.total_sq -= old * old
        return self.value


class KalmanTrend:
    """Local linear trend Kalman filter on log price: level and slope as hidden states.

    The observation noise r is the variance of hourly log returns over `var_window` bars; the
    level and slope noises are r / level_hours^2 and r / slope_hours^4, which make the slope
    follow trends over roughly slope_hours. The value is the slope over its standard deviation,
    a trend measured against its own uncertainty (research H20).
    """

    def __init__(self, level_hours: float = 24.0, slope_hours: float = 336.0, var_window: int = 720):
        self.level_hours = level_hours
        self.slope_hours = slope_hours
        self.returns = RollingStd(var_window)
        self.prev: Optional[float] = None
        self.level: Optional[float] = None
        self.slope = 0.0
        self.p11 = self.p12 = self.p22 = 0.0
        self.value: Optional[float] = None

    def update(self, close: float) -> Optional[float]:
        y = math.log(close)
        if self.prev is not None:
            self.returns.update(y - self.prev)
        self.prev = y
        sd = self.returns.value
        if sd is None or sd <= 0:
            return self.value
        r = sd * sd
        if self.level is None:
            self.level, self.slope = y, 0.0
            self.p11, self.p12, self.p22 = r, 0.0, r / self.slope_hours ** 2
            return self.value
        a11 = self.p11 + 2 * self.p12 + self.p22 + r / self.level_hours ** 2
        a12 = self.p12 + self.p22
        a22 = self.p22 + r / self.slope_hours ** 4
        s = a11 + r
        k1, k2 = a11 / s, a12 / s
        innovation = y - (self.level + self.slope)
        self.level += self.slope + k1 * innovation
        self.slope += k2 * innovation
        self.p11, self.p12, self.p22 = (1 - k1) * a11, (1 - k1) * a12, a22 - k2 * a12
        self.value = self.slope / math.sqrt(self.p22) if self.p22 > 0 else None
        return self.value


class SpreadEstimate:
    """Corwin-Schultz (2012) bid-ask spread estimate from consecutive bars' highs and lows,
    averaged over the last `window` bars. A liquidity measure (research H20)."""

    def __init__(self, window: int = 168):
        self.window = window
        self.prev: Optional[tuple] = None
        self.values: Deque[float] = deque()
        self.total = 0.0

    @property
    def value(self) -> Optional[float]:
        return self.total / len(self.values) if len(self.values) >= self.window else None

    def update(self, high: float, low: float) -> Optional[float]:
        if self.prev is not None and low > 0 and self.prev[1] > 0:
            ph, pl = self.prev
            beta = math.log(high / low) ** 2 + math.log(ph / pl) ** 2
            gamma = math.log(max(high, ph) / min(low, pl)) ** 2
            k = 3 - 2 * math.sqrt(2)
            alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
            spread = max(2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha)), 0.0)
            self.values.append(spread)
            self.total += spread
            if len(self.values) > self.window:
                self.total -= self.values.popleft()
        self.prev = (high, low)
        return self.value


class Signal(NamedTuple):
    """Indicator values for one pair at the latest closed bar."""
    ts: int               # open time of the latest bar
    close: float
    ema_fast: float
    ema_slow: float
    ema_regime: float
    atr: float
    rsi: float
    return_short: float   # close / close N bars ago - 1
    return_long: float
    volatility: float     # standard deviation of hourly log returns
    return_rotation: float = 0.0   # close / close rotation_lookback bars ago - 1
    ema_trend_fast: float = 0.0    # the slow trend filter used by the rotation sleeve
    ema_trend_slow: float = 0.0
    trend_strength: float = 0.0    # Kalman trend slope over its standard deviation
    spread: float = 0.0            # Corwin-Schultz spread estimate over the last week


class IndicatorSet:
    """All the indicators the strategy needs for one pair."""

    def __init__(self, fast_ema: int, slow_ema: int, regime_ema: int, atr_period: int,
                 rsi_period: int, momentum_short: int, momentum_long: int, volatility_window: int,
                 rotation_lookback: int = 336, trend_fast: int = 168, trend_slow: int = 672):
        self.fast = EMA(fast_ema)
        self.slow = EMA(slow_ema)
        self.regime = EMA(regime_ema)
        self.atr = ATR(atr_period)
        self.rsi = RSI(rsi_period)
        self.volatility = RollingStd(volatility_window)
        self.trend_fast = EMA(trend_fast)
        self.trend_slow = EMA(trend_slow)
        self.kalman = KalmanTrend()
        self.spread = SpreadEstimate()
        self.momentum_short = momentum_short
        self.momentum_long = momentum_long
        self.rotation_lookback = rotation_lookback
        self.closes: Deque[float] = deque(maxlen=max(momentum_short, momentum_long, rotation_lookback) + 1)
        self.returns: Deque[float] = deque(maxlen=2200)   # hourly log returns, for volatility forecasts
        self.last_ts: Optional[int] = None

    def update(self, bar: Bar) -> None:
        if self.last_ts is not None and bar.ts <= self.last_ts:
            return  # already seen
        if self.closes:
            r = math.log(bar.close / self.closes[-1])
            self.volatility.update(r)
            self.returns.append(r)
        self.closes.append(bar.close)
        self.fast.update(bar.close)
        self.slow.update(bar.close)
        self.regime.update(bar.close)
        self.trend_fast.update(bar.close)
        self.trend_slow.update(bar.close)
        self.atr.update(bar.high, bar.low, bar.close)
        self.rsi.update(bar.close)
        self.kalman.update(bar.close)
        self.spread.update(bar.high, bar.low)
        self.last_ts = bar.ts

    def signal(self) -> Optional[Signal]:
        """Latest values, or None while any indicator is still warming up."""
        values = (self.fast.value, self.slow.value, self.regime.value, self.atr.value,
                  self.rsi.value, self.volatility.value, self.trend_fast.value, self.trend_slow.value)
        if any(v is None for v in values) or len(self.closes) < self.closes.maxlen:
            return None
        close = self.closes[-1]
        return Signal(
            ts=self.last_ts,
            close=close,
            ema_fast=self.fast.value,
            ema_slow=self.slow.value,
            ema_regime=self.regime.value,
            atr=self.atr.value,
            rsi=self.rsi.value,
            return_short=close / self.closes[-1 - self.momentum_short] - 1.0,
            return_long=close / self.closes[-1 - self.momentum_long] - 1.0,
            volatility=self.volatility.value,
            return_rotation=close / self.closes[-1 - self.rotation_lookback] - 1.0,
            ema_trend_fast=self.trend_fast.value,
            ema_trend_slow=self.trend_slow.value,
            trend_strength=self.kalman.value if self.kalman.value is not None else 0.0,
            spread=self.spread.value if self.spread.value is not None else 0.0,
        )
