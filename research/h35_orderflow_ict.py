"""H35: order flow (volume delta) and ICT concepts, screened like the H33 indicators.

Order flow, from Binance's taker-buy volume (delta = taker buys - taker sells, per bar):
  CVD slope             cumulative delta over the window, as a share of volume
  CVD divergence        price's move minus the delta's, both standardised: price outrunning the
                        buying behind it
  Absorption            share of bars whose delta and price moved opposite ways
  Delta z-score         the last day's delta share against its past month
  Buying climax         the largest aggressive-buying share times relative volume in the window
ICT concepts, from OHLC:
  Liquidity sweeps      bars whose wick broke the prior swing low (high) and closed back inside:
                        bullish, bearish, and net
  Fair value gaps       net count of 3-bar imbalances (low above the high two bars back, or the
                        reverse)
  Market structure      direction of the latest break of structure (close beyond the prior
                        swing high or low), if recent
  Change of character   a break against the previous one, recently
Each on the 1-hour chart (swing 48 bars, recent 24) and the daily chart (swing 20 days, recent
3), for every coin in the month's universe. Gamma exposure cannot be tested: there is no free
history of options open interest by strike. Order-book heatmaps exist from 2023 only.

Written down before running: the H33 rules (ranking IC at 24h and 168h, partial on low
volatility; BTC timing). Strategy tests in H36 under the round-31 rule and the holdout.

    python -m research.h35_orderflow_ict
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict

import numpy as np
import pandas as pd

from bot.config import UniverseConfig, load_config
from research import fullbars
from research.folds import FOLDS, candidates
from research.h20_screen import in_fold, rank_ic
from research.panel import DEFENSIVE, load_panel, monthly_universe

OUT = os.path.join("runs", "research", "h35_screen.csv")
FEATURES: Dict[str, tuple] = {}


def feature(name: str, family: str):
    def register(fn: Callable):
        FEATURES[name] = (family, fn)
        return fn
    return register


def _w(n, k):
    return max(2, int(round(n * k)))


def _windows(chart):
    return (48, 24, 72) if chart == "1h" else (20, 3, 10)     # swing, recent, structure


def _delta(df):
    return 2 * df["taker_buy_base"] - df["volume"]


@feature("CVD slope (recent)", "order flow")
def _(df, chart, k):
    n = _w(_windows(chart)[1], k)
    return _delta(df).rolling(n).sum() / df["volume"].rolling(n).sum()
@feature("CVD slope (structure)", "order flow")
def _(df, chart, k):
    n = _w(_windows(chart)[2] * (7 if chart == "1h" else 1), k)
    return _delta(df).rolling(n).sum() / df["volume"].rolling(n).sum()
@feature("CVD divergence", "order flow")
def _(df, chart, k):
    n = _w(_windows(chart)[2], k)
    c = df["close"]
    ret = np.log(c).diff()
    pz = np.log(c / c.shift(n)) / (ret.rolling(_w(n * 3, 1)).std() * np.sqrt(n))
    share = _delta(df).rolling(n).sum() / df["volume"].rolling(n).sum()
    dz = (share - share.rolling(_w(n * 10, 1)).mean()) / share.rolling(_w(n * 10, 1)).std()
    return pz - dz
@feature("Absorption", "order flow")
def _(df, chart, k):
    n = _w(_windows(chart)[1], k)
    opposite = (np.sign(_delta(df)) * np.sign(df["close"] - df["open"]) < 0).astype(float)
    return opposite.rolling(n).mean()
@feature("Delta z-score", "order flow")
def _(df, chart, k):
    n = _w(_windows(chart)[1], k)
    share = _delta(df).rolling(n).sum() / df["volume"].rolling(n).sum()
    long = _w(n * 30 if chart == "1h" else n * 10, 1)
    return (share - share.rolling(long).mean()) / share.rolling(long).std()
@feature("Buying climax", "order flow")
def _(df, chart, k):
    n = _w(_windows(chart)[1], k)
    rel = df["volume"] / df["volume"].rolling(_w(n * 7, 1)).mean()
    return ((_delta(df) / df["volume"]) * rel).rolling(n).max()


def _sweeps(df, chart, k):
    swing, recent, _ = _windows(chart)
    s, n = _w(swing, k), _w(recent, k)
    prior_low = df["low"].shift(1).rolling(s).min()
    prior_high = df["high"].shift(1).rolling(s).max()
    bull = ((df["low"] < prior_low) & (df["close"] > prior_low)).astype(float).rolling(n).sum()
    bear = ((df["high"] > prior_high) & (df["close"] < prior_high)).astype(float).rolling(n).sum()
    return bull, bear


@feature("Bullish liquidity sweeps", "ICT")
def _(df, chart, k): return _sweeps(df, chart, k)[0]
@feature("Bearish liquidity sweeps", "ICT")
def _(df, chart, k): return _sweeps(df, chart, k)[1]
@feature("Net liquidity sweeps", "ICT")
def _(df, chart, k):
    bull, bear = _sweeps(df, chart, k)
    return bull - bear
@feature("Net fair value gaps", "ICT")
def _(df, chart, k):
    n = _w(_windows(chart)[1], k)
    bull = (df["low"] > df["high"].shift(2)).astype(float)
    bear = (df["high"] < df["low"].shift(2)).astype(float)
    return (bull - bear).rolling(n).sum() / n


def _breaks(df, chart, k):
    swing, recent, structure = _windows(chart)
    s = _w(swing, k)
    up = df["close"] > df["high"].shift(1).rolling(s).max()
    dn = df["close"] < df["low"].shift(1).rolling(s).min()
    return pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=df.index), _w(recent, k), _w(structure, k)


@feature("Market structure", "ICT")
def _(df, chart, k):
    ev, _, structure = _breaks(df, chart, k)
    last = ev.replace(0.0, np.nan).ffill()
    pos = pd.Series(np.arange(len(ev)), index=ev.index)
    since = pos - pos.where(ev != 0).ffill()
    return last.where(since <= structure, 0.0)
@feature("Change of character", "ICT")
def _(df, chart, k):
    ev, recent, _ = _breaks(df, chart, k)
    events = ev[ev != 0]
    choch = (events != events.shift(1)) & events.shift(1).notna()
    flips = (events * choch).reindex(ev.index).fillna(0.0)
    return flips.rolling(recent).sum().clip(-1, 1)


def coin_features(job):
    pair, k, names = job
    warnings.filterwarnings("ignore")
    df = pd.read_csv(fullbars.path(pair))
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                   "volume": "sum", "taker_buy_base": "sum"}).dropna()
    out = {}
    for chart, frame in (("1h", df), ("1d", daily)):
        for name in names:
            try:
                s = FEATURES[name][1](frame, chart, k).replace([np.inf, -np.inf], np.nan)
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
    pairs = [p for p in pairs if os.path.exists(fullbars.path(p))]
    with ProcessPoolExecutor(max_workers=12) as pool:
        per_coin = dict(pool.map(coin_features, [(p, k, names) for p in pairs]))
    keys = {key for d in per_coin.values() for key in d}
    return {key: pd.DataFrame({p: d[key] for p, d in per_coin.items() if key in d}) for key in keys}


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pairs = sorted(candidates(cfg))
    panel = load_panel(pairs)
    close = panel["close"]
    mask = monthly_universe(panel, UniverseConfig(), list(close.columns))
    mask[DEFENSIVE] = False
    days = close.index[(close.index.hour == 0) & (close.index >= pd.Timestamp("2020-06-01", tz="UTC"))]
    inside = mask.loc[days]
    logp = np.log(close)
    fwd = {h: (logp.shift(-h) - logp).loc[days].where(inside) for h in (24, 168)}
    low_vol = (-logp.diff().rolling(168, min_periods=150).std()).loc[days].where(inside)
    btc_fwd7 = (np.log(close["BTC/USD"]).shift(-168) - np.log(close["BTC/USD"])).loc[days]
    frames = compute([p for p in pairs if p != DEFENSIVE])
    rows = []
    for (name, chart), frame in sorted(frames.items()):
        f = frame.reindex(index=days, columns=close.columns).where(inside)
        row = {"indicator": name, "chart": chart, "family": FEATURES[name][0]}
        for h in (24, 168):
            ic, partial = rank_ic(f, fwd[h]), rank_ic(f, fwd[h], low_vol)
            means = [in_fold(ic, s, e).mean() for s, e in FOLDS]
            pmeans = [in_fold(partial, s, e).mean() for s, e in FOLDS]
            m = float(np.nanmean(means))
            row["IC %dh" % h] = round(m, 3)
            row["same %dh" % h] = sum(np.sign(x) == np.sign(m) for x in means)
            row["pass %dh" % h] = (row["same %dh" % h] >= 5 and abs(m) >= 0.02
                                   and sum(np.sign(x) == np.sign(m) for x in pmeans) >= 4)
        b = frame.get("BTC/USD")
        if b is not None:
            corrs = []
            for s, e in FOLDS:
                both = pd.concat([b.reindex(days), btc_fwd7], axis=1).dropna()
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
    print("%d series screened" % len(table))
    for col in ("pass 24h", "pass 168h", "pass timing"):
        print("  %-12s %d pass" % (col, int(table[col].fillna(False).sum())))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
