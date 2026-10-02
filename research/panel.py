"""Hour-by-pair price panels, point-in-time universes and candidate features.

Everything here looks only backwards from each hour, except the forward-return targets,
which are clearly named fwd_*. Data comes from the CSV cache that `python -m
bot.backtest` and `python -m bot.universe` fill under data/binance/.
"""
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from bot.config import UniverseConfig
from bot.market_data import binance_symbol

HOUR = pd.Timedelta(hours=1)
DEFENSIVE = "PAXG/USD"
MOMENTUM_HORIZONS = (1, 4, 12, 24, 72, 168, 336, 720)


def load_panel(pairs: List[str], data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """{"close": hour x pair, "high": ..., "low": ..., "volume": ...} from the candle cache."""
    frames = {}
    for pair in pairs:
        path = os.path.join(data_dir, "binance", binance_symbol(pair) + "_1h.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        frames[pair] = df
    fields = {}
    for field in ("open", "high", "low", "close", "volume"):
        fields[field] = pd.DataFrame({p: f[field] for p, f in frames.items()}).sort_index()
    full = pd.date_range(fields["close"].index[0], fields["close"].index[-1], freq="h")
    return {k: v.reindex(full) for k, v in fields.items()}


def cached_pairs(data_dir: str = "data") -> List[str]:
    """Every pair with a candle cache, as Roostoo names ("BTCUSDT" -> "BTC/USD")."""
    names = [os.path.basename(p)[:-len("USDT_1h.csv")]
             for p in glob.glob(os.path.join(data_dir, "binance", "*USDT_1h.csv"))]
    return sorted(n + "/USD" for n in names)


def monthly_universe(panel: Dict[str, pd.DataFrame], cfg: UniverseConfig,
                     candidates: List[str]) -> pd.DataFrame:
    """Boolean hour x pair mask: the universe rule applied on the first hour of each month.

    Mirrors bot.universe.select_universe: top `size` by USD volume over `volume_days`,
    among pairs listed at least `min_history_bars` hours earlier, plus PAXG.
    """
    close, volume = panel["close"][candidates], panel["volume"][candidates]
    dollar = (close * volume).rolling(cfg.volume_days * 24, min_periods=1).sum()
    first_seen = close.apply(lambda s: s.first_valid_index())
    mask = pd.DataFrame(False, index=close.index, columns=candidates)
    months = pd.date_range(close.index[0].ceil("D") + pd.offsets.MonthBegin(0), close.index[-1], freq="MS")
    for start, end in zip(months, list(months[1:]) + [close.index[-1] + HOUR]):
        at = start - HOUR  # volume known when the month starts
        if at not in dollar.index:
            continue
        listed = [p for p in candidates
                  if first_seen[p] is not None and first_seen[p] <= start - cfg.min_history_bars * HOUR]
        ranked = dollar.loc[at, listed].drop(DEFENSIVE, errors="ignore").sort_values(ascending=False)
        chosen = list(ranked.index[:cfg.size]) + ([DEFENSIVE] if DEFENSIVE in listed else [])
        mask.loc[start:end - HOUR, chosen] = True
    return mask


def features(panel: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Candidate predictors, each an hour x pair frame known at the close of that hour."""
    close, high, low, volume = panel["close"], panel["high"], panel["low"], panel["volume"]
    logp = np.log(close)
    ret = logp.diff()
    sig24 = ret.rolling(24, min_periods=20).std()
    sig168 = ret.rolling(168, min_periods=150).std()
    out = {}
    for k in MOMENTUM_HORIZONS:
        out["mom_%d" % k] = (logp - logp.shift(k)) / (sig168 * np.sqrt(k))
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    out["trend_50_200"] = (ema50 / ema200 - 1) / (sig168 * np.sqrt(168))
    out["dist_ema200"] = (close / ema200 - 1) / (sig168 * np.sqrt(168))
    gain = ret.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-ret).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss)
    out["vol_ratio"] = np.log(sig24 / sig168)
    out["vol_168"] = np.log(sig168)
    out["rel_volume"] = np.log((volume.rolling(24).sum() + 1e-12) / (volume.rolling(168).sum() / 7 + 1e-12))
    hi168, lo168 = high.rolling(168).max(), low.rolling(168).min()
    out["range_pos_168"] = (close - lo168) / (hi168 - lo168)
    out["drawdown_168"] = close / hi168 - 1
    out["liquidity"] = np.log((close * volume).rolling(720, min_periods=100).sum())
    # Market-wide context, the same value for every pair at a given hour.
    btc = "BTC/USD"
    for k in (24, 168):
        out["btc_mom_%d" % k] = _broadcast(out["mom_%d" % k][btc], close)
    out["btc_trend"] = _broadcast(out["trend_50_200"][btc], close)
    out["breadth"] = _broadcast((out["trend_50_200"] > 0).where(close.notna()).mean(axis=1), close)
    return out


def targets(panel: Dict[str, pd.DataFrame], horizon: int = 24) -> Dict[str, pd.DataFrame]:
    """Forward log return over `horizon` hours, raw and divided by trailing volatility."""
    logp = np.log(panel["close"])
    fwd = logp.shift(-horizon) - logp
    sig168 = logp.diff().rolling(168, min_periods=150).std()
    return {"fwd": fwd, "fwd_risk_adj": fwd / (sig168 * np.sqrt(horizon))}


def _broadcast(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.repeat(series.values[:, None], like.shape[1], axis=1),
                        index=like.index, columns=like.columns).where(like.notna())
