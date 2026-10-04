"""H33: TradingView's built-in indicators, screened for anything the strategy could use.

About 65 indicators (most from the `ta` library, which implements TradingView's built-ins, plus
SuperTrend, Hull MA, Choppiness, Elder Ray, Chande Momentum, Coppock, Balance of Power and
linear-regression slope and R^2), each on the 1-hour chart and on the daily chart, every one
turned scale-free (price levels become distance from price, cumulative volume lines become
changes over the indicator's window), for every coin in the month's universe.

Written down before running (the H20 rules):
  Ranking  the daily rank IC with the next 24 and 168 hours across coins: same sign in at least
           5 of 6 folds, mean |IC| at least 0.02, and with low volatility partialled out the
           same sign in at least 4.
  Timing   the indicator on BTC at 00:00 against BTC's next 7 days (Spearman per fold): same
           sign in at least 5 of 6 folds and mean |correlation| at least 0.05.
With about 130 series, several will pass by luck; the strategy tests (H34) decide, under the
round-31 rule and the 2018-2020 holdout.

    python -m research.h33_indicators
"""
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict

import numpy as np
import pandas as pd
import ta

from bot.config import UniverseConfig, load_config
from bot.market_data import binance_symbol
from research.folds import FOLDS, candidates
from research.h20_screen import in_fold, rank_ic
from research.panel import DEFENSIVE, load_panel, monthly_universe

OUT = os.path.join("runs", "research", "h33_screen.csv")


def _w(n: float, k: float) -> int:
    return max(2, int(round(n * k)))


def _wma(s: pd.Series, n: int) -> pd.Series:
    weights = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: float(np.dot(x, weights) / weights.sum()), raw=True)


def _supertrend(h, l, c, n, mult) -> pd.Series:
    atr = ta.volatility.AverageTrueRange(h, l, c, window=n).average_true_range().values
    hl2 = ((h + l) / 2).values
    up, dn = hl2 - mult * atr, hl2 + mult * atr
    cv = c.values
    direction = np.zeros(len(cv))
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(cv)):
        fu[i] = max(up[i], fu[i - 1]) if cv[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if cv[i - 1] < fd[i - 1] else dn[i]
        if cv[i] > fd[i - 1]:
            direction[i] = 1
        elif cv[i] < fu[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=c.index)


def _linreg(logp: pd.Series, n: int):
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()

    def slope(y):
        return float(((x - xm) * (y - y.mean())).sum() / sxx)

    def r2(y):
        b = ((x - xm) * (y - y.mean())).sum() / sxx
        fit = y.mean() + b * (x - xm)
        tot = ((y - y.mean()) ** 2).sum()
        return float(1 - ((y - fit) ** 2).sum() / tot) if tot > 0 else 0.0
    sd = logp.diff().rolling(n).std()
    return logp.rolling(n).apply(slope, raw=True) / sd, logp.rolling(n).apply(r2, raw=True)


# name -> (family, function(o, h, l, c, v, k) -> Series), k scales every window (1 = default)
FEATURES: Dict[str, tuple] = {}


def feature(name: str, family: str):
    def register(fn: Callable):
        FEATURES[name] = (family, fn)
        return fn
    return register


@feature("RSI", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.RSIIndicator(c, _w(14, k)).rsi()
@feature("Stochastic %K", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.StochasticOscillator(h, l, c, _w(14, k), _w(3, k)).stoch()
@feature("Stochastic %D", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.StochasticOscillator(h, l, c, _w(14, k), _w(3, k)).stoch_signal()
@feature("Stochastic RSI", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.StochRSIIndicator(c, _w(14, k), _w(3, k), _w(3, k)).stochrsi()
@feature("Williams %R", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.WilliamsRIndicator(h, l, c, _w(14, k)).williams_r()
@feature("CCI", "momentum")
def _(o, h, l, c, v, k): return ta.trend.CCIIndicator(h, l, c, _w(20, k)).cci()
@feature("Awesome Oscillator", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.AwesomeOscillatorIndicator(h, l, _w(5, k), _w(34, k)).awesome_oscillator() / c
@feature("Ultimate Oscillator", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.UltimateOscillator(h, l, c, _w(7, k), _w(14, k), _w(28, k)).ultimate_oscillator()
@feature("True Strength Index", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.TSIIndicator(c, _w(25, k), _w(13, k)).tsi()
@feature("Rate of Change", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.ROCIndicator(c, _w(12, k)).roc()
@feature("Momentum (10)", "momentum")
def _(o, h, l, c, v, k): return c / c.shift(_w(10, k)) - 1
@feature("KAMA distance", "momentum")
def _(o, h, l, c, v, k): return c / ta.momentum.KAMAIndicator(c, _w(10, k), 2, 30).kama() - 1
@feature("PPO", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.PercentagePriceOscillator(c, _w(26, k), _w(12, k), _w(9, k)).ppo()
@feature("PPO histogram", "momentum")
def _(o, h, l, c, v, k): return ta.momentum.PercentagePriceOscillator(c, _w(26, k), _w(12, k), _w(9, k)).ppo_hist()
@feature("MACD histogram", "momentum")
def _(o, h, l, c, v, k): return ta.trend.MACD(c, _w(26, k), _w(12, k), _w(9, k)).macd_diff() / c
@feature("Chande Momentum", "momentum")
def _(o, h, l, c, v, k):
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(_w(9, k)).sum(), (-d).clip(lower=0).rolling(_w(9, k)).sum()
    return 100 * (up - dn) / (up + dn)
@feature("TRIX", "momentum")
def _(o, h, l, c, v, k): return ta.trend.TRIXIndicator(c, _w(15, k)).trix()
@feature("KST minus signal", "momentum")
def _(o, h, l, c, v, k):
    return ta.trend.KSTIndicator(c, _w(10, k), _w(15, k), _w(20, k), _w(30, k), _w(10, k), _w(10, k),
                                 _w(10, k), _w(15, k), _w(9, k)).kst_diff()
@feature("Detrended Price Oscillator", "momentum")
def _(o, h, l, c, v, k): return ta.trend.DPOIndicator(c, _w(20, k)).dpo() / c
@feature("Coppock Curve", "momentum")
def _(o, h, l, c, v, k):
    roc = (c / c.shift(_w(14, k)) - 1) + (c / c.shift(_w(11, k)) - 1)
    return roc.ewm(span=_w(10, k), adjust=False).mean()
@feature("Elder bull power", "momentum")
def _(o, h, l, c, v, k): return (h - c.ewm(span=_w(13, k), adjust=False).mean()) / c
@feature("Elder bear power", "momentum")
def _(o, h, l, c, v, k): return (l - c.ewm(span=_w(13, k), adjust=False).mean()) / c
@feature("Balance of Power", "momentum")
def _(o, h, l, c, v, k): return ((c - o) / (h - l).replace(0, np.nan)).rolling(_w(14, k)).mean()

@feature("ADX", "trend")
def _(o, h, l, c, v, k): return ta.trend.ADXIndicator(h, l, c, _w(14, k)).adx()
@feature("DI+ minus DI-", "trend")
def _(o, h, l, c, v, k):
    x = ta.trend.ADXIndicator(h, l, c, _w(14, k))
    return x.adx_pos() - x.adx_neg()
@feature("Aroon oscillator", "trend")
def _(o, h, l, c, v, k): return ta.trend.AroonIndicator(h, l, _w(25, k)).aroon_indicator()
@feature("Vortex VI+ minus VI-", "trend")
def _(o, h, l, c, v, k): return ta.trend.VortexIndicator(h, l, c, _w(14, k)).vortex_indicator_diff()
@feature("Mass Index", "trend")
def _(o, h, l, c, v, k): return ta.trend.MassIndex(h, l, _w(9, k), _w(25, k)).mass_index()
@feature("Ichimoku: distance from base line", "trend")
def _(o, h, l, c, v, k): return c / ta.trend.IchimokuIndicator(h, l, _w(9, k), _w(26, k), _w(52, k)).ichimoku_base_line() - 1
@feature("Ichimoku: distance above the cloud", "trend")
def _(o, h, l, c, v, k):
    x = ta.trend.IchimokuIndicator(h, l, _w(9, k), _w(26, k), _w(52, k))
    return c / np.maximum(x.ichimoku_a(), x.ichimoku_b()) - 1
@feature("Parabolic SAR distance", "trend")
def _(o, h, l, c, v, k): return c / ta.trend.PSARIndicator(h, l, c, 0.02 * k, 0.2).psar() - 1
@feature("Schaff Trend Cycle", "trend")
def _(o, h, l, c, v, k): return ta.trend.STCIndicator(c, _w(50, k), _w(23, k), _w(10, k)).stc()
@feature("SMA(50) distance", "trend")
def _(o, h, l, c, v, k): return c / c.rolling(_w(50, k)).mean() - 1
@feature("SMA(200) distance", "trend")
def _(o, h, l, c, v, k): return c / c.rolling(_w(200, k)).mean() - 1
@feature("EMA(20) distance", "trend")
def _(o, h, l, c, v, k): return c / c.ewm(span=_w(20, k), adjust=False).mean() - 1
@feature("WMA(20) distance", "trend")
def _(o, h, l, c, v, k): return c / _wma(c, _w(20, k)) - 1
@feature("Hull MA distance", "trend")
def _(o, h, l, c, v, k):
    n = _w(20, k)
    raw = 2 * _wma(c, max(2, n // 2)) - _wma(c, n)
    return c / _wma(raw, max(2, int(math.sqrt(n)))) - 1
@feature("Linear regression slope", "trend")
def _(o, h, l, c, v, k): return _linreg(np.log(c), _w(20, k))[0]
@feature("Linear regression R2", "trend")
def _(o, h, l, c, v, k): return _linreg(np.log(c), _w(20, k))[1]
@feature("SuperTrend direction", "trend")
def _(o, h, l, c, v, k): return _supertrend(h, l, c, _w(10, k), 3.0)

@feature("ATR %", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.AverageTrueRange(h, l, c, _w(14, k)).average_true_range() / c
@feature("Bollinger %B", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.BollingerBands(c, _w(20, k), 2).bollinger_pband()
@feature("Bollinger bandwidth", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.BollingerBands(c, _w(20, k), 2).bollinger_wband()
@feature("Keltner %", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.KeltnerChannel(h, l, c, _w(20, k), _w(10, k), original_version=False).keltner_channel_pband()
@feature("Keltner width", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.KeltnerChannel(h, l, c, _w(20, k), _w(10, k), original_version=False).keltner_channel_wband()
@feature("Donchian %", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.DonchianChannel(h, l, c, _w(20, k)).donchian_channel_pband()
@feature("Donchian width", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.DonchianChannel(h, l, c, _w(20, k)).donchian_channel_wband()
@feature("Ulcer Index", "volatility")
def _(o, h, l, c, v, k): return ta.volatility.UlcerIndex(c, _w(14, k)).ulcer_index()
@feature("Historical volatility", "volatility")
def _(o, h, l, c, v, k): return np.log(c).diff().rolling(_w(20, k)).std()
@feature("Choppiness Index", "volatility")
def _(o, h, l, c, v, k):
    n = _w(14, k)
    tr = ta.volatility.AverageTrueRange(h, l, c, 1).average_true_range()
    rng = h.rolling(n).max() - l.rolling(n).min()
    return 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)

@feature("OBV change", "volume")
def _(o, h, l, c, v, k):
    n = _w(20, k)
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    return (obv - obv.shift(n)) / v.rolling(n).sum()
@feature("Accumulation/Distribution change", "volume")
def _(o, h, l, c, v, k):
    n = _w(20, k)
    adi = ta.volume.AccDistIndexIndicator(h, l, c, v).acc_dist_index()
    return (adi - adi.shift(n)) / v.rolling(n).sum()
@feature("Chaikin Money Flow", "volume")
def _(o, h, l, c, v, k): return ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, _w(20, k)).chaikin_money_flow()
@feature("Chaikin Oscillator", "volume")
def _(o, h, l, c, v, k):
    adi = ta.volume.AccDistIndexIndicator(h, l, c, v).acc_dist_index()
    return (adi.ewm(span=_w(3, k), adjust=False).mean() - adi.ewm(span=_w(10, k), adjust=False).mean()) / v.rolling(_w(10, k)).sum()
@feature("Force Index", "volume")
def _(o, h, l, c, v, k):
    n = _w(13, k)
    return ta.volume.ForceIndexIndicator(c, v, n).force_index() / (c * v.rolling(n).mean())
@feature("Ease of Movement", "volume")
def _(o, h, l, c, v, k):
    n = _w(14, k)
    return ta.volume.EaseOfMovementIndicator(h, l, v, n).sma_ease_of_movement() * v.rolling(n).mean() / c / 1e8
@feature("Volume Price Trend change", "volume")
def _(o, h, l, c, v, k):
    n = _w(20, k)
    vpt = ta.volume.VolumePriceTrendIndicator(c, v).volume_price_trend()
    return (vpt - vpt.shift(n)) / v.rolling(n).sum()
@feature("Negative Volume Index change", "volume")
def _(o, h, l, c, v, k):
    nvi = ta.volume.NegativeVolumeIndexIndicator(c, v).negative_volume_index()
    return nvi / nvi.shift(_w(20, k)) - 1
@feature("Money Flow Index", "volume")
def _(o, h, l, c, v, k): return ta.volume.MFIIndicator(h, l, c, v, _w(14, k)).money_flow_index()
@feature("VWAP distance", "volume")
def _(o, h, l, c, v, k): return c / ta.volume.VolumeWeightedAveragePrice(h, l, c, v, _w(24, k)).volume_weighted_average_price() - 1
@feature("Relative volume", "volume")
def _(o, h, l, c, v, k): return v / v.rolling(_w(20, k)).mean()
@feature("Percentage Volume Oscillator", "volume")
def _(o, h, l, c, v, k): return ta.momentum.PercentageVolumeOscillator(v, _w(26, k), _w(12, k), _w(9, k)).pvo()


def coin_features(job):
    """{(feature, chart): Series on 00:00 rows} for one coin; the daily chart's value for a day is
    placed on the next day's 00:00 row (when that day has closed)."""
    pair, k, names = job
    warnings.filterwarnings("ignore")
    df = pd.read_csv(os.path.join("data", "binance", binance_symbol(pair) + "_1h.csv"))
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[df.index >= pd.Timestamp("2018-01-01", tz="UTC")]
    out = {}
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    for chart, frame in (("1h", df), ("1d", daily)):
        o, h, l, c, v = (frame[x] for x in ("open", "high", "low", "close", "volume"))
        for name in names:
            try:
                s = FEATURES[name][1](o, h, l, c, v, k).replace([np.inf, -np.inf], np.nan)
            except Exception:
                continue
            if chart == "1h":
                s = s[s.index.hour == 0]
            else:
                s.index = s.index + pd.Timedelta(days=1)
            out[(name, chart)] = s
    return pair, out


def compute(pairs, k: float = 1.0, names=None):
    names = names or list(FEATURES)
    with ProcessPoolExecutor(max_workers=12) as pool:
        per_coin = dict(pool.map(coin_features, [(p, k, names) for p in pairs]))
    frames = {}
    for key in {key for d in per_coin.values() for key in d}:
        frames[key] = pd.DataFrame({p: d[key] for p, d in per_coin.items() if key in d})
    return frames


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pairs = sorted(p for p in candidates(cfg))
    panel = load_panel(pairs)
    close = panel["close"]
    mask = monthly_universe(panel, UniverseConfig(), list(close.columns))
    mask[DEFENSIVE] = False
    days = close.index[(close.index.hour == 0) & (close.index >= pd.Timestamp("2020-06-01", tz="UTC"))]
    inside = mask.loc[days]
    logp = np.log(close)
    fwd = {h: (logp.shift(-h) - logp).loc[days].where(inside) for h in (24, 168)}
    low_vol = (-logp.diff().rolling(168, min_periods=150).std()).loc[days].where(inside)
    daily_close = close.loc[days, "BTC/USD"]
    btc_fwd7 = np.log(close["BTC/USD"]).shift(-168) - np.log(close["BTC/USD"])
    btc_fwd7 = btc_fwd7.loc[days]

    frames = compute([p for p in pairs if p != DEFENSIVE])
    rows = []
    for (name, chart), frame in sorted(frames.items()):
        family = FEATURES[name][0]
        f = frame.reindex(index=days, columns=close.columns).where(inside)
        row = {"indicator": name, "chart": chart, "family": family}
        best = None
        for h in (24, 168):
            ic, partial = rank_ic(f, fwd[h]), rank_ic(f, fwd[h], low_vol)
            means = [in_fold(ic, s, e).mean() for s, e in FOLDS]
            pmeans = [in_fold(partial, s, e).mean() for s, e in FOLDS]
            m = float(np.nanmean(means))
            same = sum(np.sign(x) == np.sign(m) for x in means)
            psame = sum(np.sign(x) == np.sign(m) for x in pmeans)
            ok = same >= 5 and abs(m) >= 0.02 and psame >= 4
            row["IC %dh" % h] = round(m, 3)
            row["same %dh" % h] = same
            row["pass %dh" % h] = ok
        btc = frame.get("BTC/USD")
        if btc is not None:
            b = btc.reindex(days)
            corrs = []
            for s, e in FOLDS:
                both = pd.concat([b, btc_fwd7], axis=1).dropna()
                both = both[(both.index >= pd.Timestamp(s, tz="UTC")) & (both.index < pd.Timestamp(e, tz="UTC"))]
                corrs.append(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman") if len(both) > 20 else np.nan)
            m = float(np.nanmean(corrs))
            row["BTC timing corr"] = round(m, 3)
            row["timing same"] = sum(np.sign(x) == np.sign(m) for x in corrs)
            row["pass timing"] = row["timing same"] >= 5 and abs(m) >= 0.05
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    n = len(table)
    print("%d indicator series screened" % n)
    for col in ("pass 24h", "pass 168h", "pass timing"):
        print("  %-12s %d pass" % (col, int(table[col].fillna(False).sum())))
    passed = table[table[["pass 24h", "pass 168h", "pass timing"]].fillna(False).any(axis=1)]
    print(passed.to_string(index=False))


if __name__ == "__main__":
    main()
