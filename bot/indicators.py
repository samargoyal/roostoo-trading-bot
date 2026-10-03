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
        self.momentum_short = momentum_short
        self.momentum_long = momentum_long
        self.rotation_lookback = rotation_lookback
        self.closes: Deque[float] = deque(maxlen=max(momentum_short, momentum_long, rotation_lookback) + 1)
        self.last_ts: Optional[int] = None

    def update(self, bar: Bar) -> None:
        if self.last_ts is not None and bar.ts <= self.last_ts:
            return  # already seen
        if self.closes:
            self.volatility.update(math.log(bar.close / self.closes[-1]))
        self.closes.append(bar.close)
        self.fast.update(bar.close)
        self.slow.update(bar.close)
        self.regime.update(bar.close)
        self.trend_fast.update(bar.close)
        self.trend_slow.update(bar.close)
        self.atr.update(bar.high, bar.low, bar.close)
        self.rsi.update(bar.close)
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
        )
