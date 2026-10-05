"""H60: 51 swing and medium-frequency strategies, each from a stated hypothesis.

Written down before running. Swing strategies hold for hours to days and run on hourly bars over
the bot's 45 coins plus the four meme coins its spread limit leaves out (PEPE, BONK, SHIB,
1000CHEEMS). Medium-frequency (MFT) strategies hold for minutes to hours and run on 5-minute
bars (research/klines5m.py) over 16 coins, the majors and the meme coins.

Every strategy:
  - trades at the close of the bar its signal comes from (the position earns from the next bar);
  - pays Roostoo's taker fee (0.1%) plus half the pair's real Roostoo spread on every change of
    position (research/candidates.csv); shorts are 1x, as on Roostoo;
  - puts at most a quarter of its capital in one coin (signals / max(active coins, 4));
  - only trades a coin after 30 days of history (no listing-day chaos).

Judged on the six one-year folds (October 2020 to October 2026), as everything else:
  Stage A, an edge after costs: a positive net return and a positive composite score in at
    least 5 of the 6 folds (or in all but one of the folds it had data for, at least 3).
  Stage B, worth adding to the live bot: 20% of equity in the strategy, the rest in the bot
    (R54b), rebalanced daily, beats the bot alone on the yearly composite in at least 5 of 6
    folds with a worst drawdown at most 2 points deeper; then its two neighbouring settings must
    pass Stage A, and it must beat the bot in the 2018-2020 holdout years where data allow.
With 51 strategies tested, a few could pass Stage A by luck; Stage B, the neighbours and the
holdout are there for that.

    python -m research.h60_swing_mft            # Stage A for all 51
    python -m research.h60_swing_mft --stage-b  # Stage B for the survivors
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from bot.config import DEFAULT_UNIVERSE
from bot.metrics import summarize
from research.folds import FOLDS, candidate_table, funding_table
from research.fullbars import load as load_full

OUT = os.path.join("runs", "research", "h60")
FEE = 0.001
MAX_SHARE = 4                      # at most 1/4 of the capital per coin

ALL_1H = [p for p in DEFAULT_UNIVERSE if p != "PAXG/USD"] + ["PEPE/USD", "BONK/USD", "SHIB/USD", "1000CHEEMS/USD"]
MEMES_1H = ["DOGE/USD", "SHIB/USD", "PEPE/USD", "BONK/USD", "FLOKI/USD", "WIF/USD", "PENGU/USD", "TRUMP/USD",
            "1000CHEEMS/USD", "TUT/USD"]
MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD"]
ALL_5M = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "BNB/USD", "SUI/USD", "DOGE/USD", "SHIB/USD", "PEPE/USD",
          "BONK/USD", "FLOKI/USD", "WIF/USD", "PENGU/USD", "TRUMP/USD", "1000CHEEMS/USD", "PUMP/USD"]
MEMES_5M = ["DOGE/USD", "SHIB/USD", "PEPE/USD", "BONK/USD", "FLOKI/USD", "WIF/USD", "PENGU/USD", "TRUMP/USD",
            "1000CHEEMS/USD"]
PAIRS = [("DOGE/USD", "SHIB/USD"), ("PEPE/USD", "BONK/USD"), ("PEPE/USD", "FLOKI/USD"), ("WIF/USD", "BONK/USD")]


# ---- data ------------------------------------------------------------------------------

class Data:
    """Bars on a full index, one column per pair, and what the strategies derive from them."""

    def __init__(self, fields, bars_per_hour):
        self.c, self.h, self.l = fields["close"], fields["high"], fields["low"]
        self.o = fields["open"]
        self.qv = fields["quote_volume"]
        self.vol = fields["volume"]
        self.taker = fields["taker_buy_base"]
        self.trades = fields["trades"]
        self.bph = bars_per_hour
        self.r = np.log(self.c).diff()
        age = self.c.notna().cumsum()
        self.ok = (age >= 720 * bars_per_hour) & self.c.notna()
        self.idx = self.c.index

    def hours(self, n):
        return int(round(n * self.bph))

    def sigma(self, hours=24):
        n = self.hours(hours)
        return self.r.rolling(n, min_periods=n // 2).std()


def load_1h():
    f = load_full(ALL_1H)
    f = {k: v.loc["2019-09-01":"2026-10-01"] for k, v in f.items()}
    return Data(f, 1)


def load_5m():
    frames = {}
    for pair in ALL_5M:
        path = os.path.join("data", "binance_5m", pair.replace("/USD", "USDT") + ".parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
            frames[pair] = df
    idx = pd.date_range("2020-10-01", "2026-10-01", freq="5min", tz="UTC", inclusive="left")
    fields = {}
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"):
        fields[col] = pd.DataFrame({p: f[col] for p, f in frames.items()}).reindex(idx).astype("float64")
    return Data(fields, 12)


def costs(pairs):
    spread = {r["pair"]: r["spread"] for r in candidate_table()}
    return pd.Series({p: FEE + max(0.0002, spread.get(p, 0.002) / 2) for p in pairs})


# ---- building blocks -------------------------------------------------------------------

def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def atr(D, n):
    prev = D.c.shift(1)
    tr = pd.concat([(D.h - D.l), (D.h - prev).abs(), (D.l - prev).abs()]).groupby(level=0).max()
    return tr.reindex(D.idx).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def held(entry, bars):
    """In a position for `bars` bars after each entry signal (a new signal restarts the clock)."""
    return entry.astype(float).rolling(bars, min_periods=1).max().fillna(0) > 0


def state(entry, exit_):
    """In from an entry until an exit (an exit on the same bar wins)."""
    s = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
    s[entry.fillna(False)] = 1.0
    s[exit_.fillna(False)] = 0.0
    return s.ffill().fillna(0.0) > 0


def at_hours(D, hours):
    """Boolean index: bars whose close falls in the given UTC hours."""
    close_time = D.idx + pd.Timedelta(minutes=60 / D.bph)
    return pd.Series(close_time.hour.isin(hours), index=D.idx)


def every(D, hours, offset=0):
    """Bars that close on a multiple of `hours` (e.g. daily at 00:00 UTC)."""
    close_time = D.idx + pd.Timedelta(minutes=60 / D.bph)
    first = (close_time.minute == 0) & ((close_time.hour - offset) % hours == 0)
    return pd.Series(first, index=D.idx)


def rank_pick(score, mask, k, lowest=False):
    s = score.where(mask)
    r = s.rank(axis=1, ascending=lowest, method="first")
    return (r <= k) & s.notna()


def basket(D, pairs):
    cols = [p for p in pairs if p in D.c.columns]
    return cols


def only(sig, cols):
    out = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    out[cols] = sig[cols].astype(float)
    return out


def daily_series(path, col, D):
    df = pd.read_csv(path)
    s = pd.Series(df[col].values, index=pd.to_datetime(df["ts"], unit="s", utc=True)).sort_index()
    s = s[~s.index.duplicated()]
    return s.shift(1).reindex(D.idx, method="ffill")     # the previous day's value: known by then


# ---- swing strategies (hourly bars) ----------------------------------------------------

def sw01(D, entry_h=72, exit_h=24):
    """Breakout continuation: a close above the 3-day high draws in momentum traders (an
    attention cascade), so new highs keep rising; out below the 1-day low (Turtle rules)."""
    e = D.c > D.h.shift(1).rolling(entry_h).max()
    x = D.c < D.l.shift(1).rolling(exit_h).min()
    return state(e, x)


def sw02(D, look=24):
    """Short-horizon time-series momentum: a strong 1-day move (over 1 sigma of its usual size)
    tends to continue (Liu and Tsyvinski 2021); held until the move fades below zero."""
    z = (np.log(D.c / D.c.shift(look))) / (D.sigma(168) * np.sqrt(look))
    return state(z > 1, z < 0)


def sw03(D, mult=3.0):
    """Volume-confirmed breakout: a new 2-day high on 3x normal volume is informed or attention
    flow, not noise (the high-volume return premium, Gervais et al. 2001); held 1 day."""
    e = (D.c > D.h.shift(1).rolling(48).max()) & (D.qv > mult * D.qv.rolling(168).median())
    return held(e, 24)


def sw04(D, q=0.2):
    """Volatility squeeze: calm periods precede big moves (volatility clusters), and a breakout
    from a squeeze tends to run; long above the upper band after a squeeze, out at the mean."""
    ma, sd = D.c.rolling(20).mean(), D.c.rolling(20).std()
    width = 4 * sd / ma
    squeezed = (width < width.rolling(720).quantile(q)).astype(float).rolling(24).max() > 0
    return state(squeezed & (D.c > ma + 2 * sd), D.c < ma)


def sw05(D, thresh=0.05):
    """Meme waves: meme coins move together in retail hype waves (herding), so when the basket
    is up 5% in a day, hold all of them until the basket's day turns negative."""
    cols = basket(D, MEMES_1H)
    b = np.log(D.c[cols] / D.c[cols].shift(24)).where(D.ok[cols]).mean(axis=1)
    on = state(pd.DataFrame({"x": b > thresh}), pd.DataFrame({"x": b < 0}))["x"]
    sig = pd.DataFrame(False, index=D.idx, columns=D.c.columns)
    sig[cols] = pd.DataFrame(np.repeat(on.values[:, None], len(cols), axis=1), index=D.idx, columns=cols)
    return sig


def sw06(D, thresh=0.05, k=2):
    """Laggard catch-up: hype spreads from the leading meme coins to the rest with a delay (the
    lead-lag of related assets, Hou 2007), so in a meme wave buy the 2 that moved least; 12h."""
    cols = basket(D, MEMES_1H)
    r24 = np.log(D.c / D.c.shift(24))
    b = r24[cols].where(D.ok[cols]).mean(axis=1)
    mask = D.ok.copy()
    mask[[c for c in D.c.columns if c not in cols]] = False
    pick = rank_pick(r24, mask, k, lowest=True) & (b > thresh).values[:, None]
    return held(pick, 12)


def sw07(D, z=2.0):
    """BTC leads, altcoins follow: after a strong 6-hour BTC move, the coins that have not yet
    followed tend to catch up (slow diffusion of market-wide news); held 12h."""
    r6 = np.log(D.c / D.c.shift(6))
    btc = r6["BTC/USD"] / (D.sigma(168)["BTC/USD"] * np.sqrt(6))
    lag = r6.lt(0.5 * r6["BTC/USD"], axis=0) & D.ok
    lag["BTC/USD"] = False
    return held(lag & (btc > z).values[:, None], 12)


def sw08(D, hi=0.55):
    """Order-flow pressure: when takers (aggressive buyers) dominate for 12 hours in a rising
    coin, the buying persists (order flow is autocorrelated); out when buyers fade."""
    share = (D.taker * D.c).rolling(12).sum() / D.qv.rolling(12).sum()
    return state((share > hi) & (D.c > D.c.rolling(24).mean()), share < 0.5)


def sw09(D):
    """Short squeeze fuel: a coin rising while its perpetual funding is negative has crowded
    shorts who must buy back (short covering); long while both hold."""
    table = funding_table(72)
    f = pd.DataFrame.from_dict(table, orient="index")
    f.index = pd.to_datetime(f.index, unit="ms", utc=True)
    f = f.sort_index().reindex(D.idx, method="ffill").reindex(columns=D.c.columns)
    up = D.c > ema(D.c, 72)
    return state((f < 0) & up, (f > 0) | ~up)


def sw10(D):
    """Weekend retail: meme coins are retail assets, and retail trades most at weekends while
    institutions are away; hold memes from Friday 20:00 to Monday 00:00 UTC."""
    close_time = D.idx + pd.Timedelta(hours=1)
    wk = pd.Series(((close_time.dayofweek == 4) & (close_time.hour >= 20)) | close_time.dayofweek.isin([5, 6]),
                   index=D.idx)
    return only(pd.DataFrame(np.repeat(wk.values[:, None], len(D.c.columns), axis=1), index=D.idx,
                             columns=D.c.columns), basket(D, MEMES_1H)) > 0


def sw11(D):
    """US-session flows: US retail drives meme coins, so their gains come in the US session;
    hold memes 13:00-21:00 UTC only."""
    on = at_hours(D, range(14, 22))
    return only(pd.DataFrame(np.repeat(on.values[:, None], len(D.c.columns), axis=1), index=D.idx,
                             columns=D.c.columns), basket(D, MEMES_1H)) > 0


def sw12(D):
    """Asian-session drift: majors drift up in Asian hours as Asian flows meet thin liquidity;
    hold the majors 00:00-08:00 UTC only."""
    on = at_hours(D, range(1, 9))
    return only(pd.DataFrame(np.repeat(on.values[:, None], len(D.c.columns), axis=1), index=D.idx,
                             columns=D.c.columns), MAJORS) > 0


def sw13(D, level=10):
    """Liquidity provision in uptrends: a sharp 3-hour drop in a coin above its 200-hour EMA is
    an overreaction that reverts (Connors' RSI-2); out once back above the 5-hour mean."""
    return state((rsi(D.c, 3) < level) & (D.c > ema(D.c, 200)), D.c > D.c.rolling(5).mean())


def sw14(D, z=4.0):
    """Capitulation: a 4-sigma hourly drop on 5x volume is forced selling (liquidations), which
    pushes the price below value; it reverts over the next day (Brunnermeier-Pedersen)."""
    zs = D.r / D.sigma(168)
    return held((zs < -z) & (D.qv > 5 * D.qv.rolling(168).median()), 24)


def sw15(D, pump=0.25):
    """Post-pump fade: a coin up 25% in a day has overshot on hype and gives some back (short-
    horizon overreaction); short for a day."""
    return -held(D.c / D.c.shift(24) - 1 > pump, 24).astype(float)


def sw16(D, drop=0.08):
    """Buy the dip in leaders: a 1-day drop of 8% in a coin with top-quintile 2-week momentum is
    a liquidity shock in a strong trend, which resumes; held 2 days."""
    m = D.c / D.c.shift(336) - 1
    leader = m.where(D.ok).rank(axis=1, pct=True) >= 0.8
    return held(leader & (D.c / D.c.shift(24) - 1 < -drop), 48)


def pairs_signal(D, z_in, z_out, window):
    sig = pd.DataFrame(0.0, index=D.idx, columns=D.c.columns)
    n = 0
    for a, b in PAIRS:
        if a not in D.c.columns or b not in D.c.columns:
            continue
        s = np.log(D.c[a]) - np.log(D.c[b])
        z = (s - s.rolling(window).mean()) / s.rolling(window).std()
        ok = (D.ok[a] & D.ok[b])
        long_a = state(pd.DataFrame({"x": (z < -z_in) & ok}), pd.DataFrame({"x": (z > -z_out) | ~ok}))["x"]
        long_b = state(pd.DataFrame({"x": (z > z_in) & ok}), pd.DataFrame({"x": (z < z_out) | ~ok}))["x"]
        leg = long_a.astype(float) - long_b.astype(float)       # +1: long a, short b
        sig[a] += 0.5 * leg
        sig[b] -= 0.5 * leg
        n += 1
    return sig


def sw17(D):
    """Meme pairs: coins riding the same narrative are substitutes, so when one runs 2 sigma
    ahead of its pair over a week, the gap closes; long the laggard, short the leader."""
    return pairs_signal(D, 2.0, 0.5, 168)


def sw18(D, k=3):
    """Weekly reversal: the week's biggest losers among coins still in a 30-day uptrend bounce
    back (our own finding that losers bounce too hard to short, used from the long side);
    picked daily, held 3 days."""
    w = D.c / D.c.shift(168) - 1
    pick = rank_pick(w, D.ok & (D.c > ema(D.c, 720)), k, lowest=True) & every(D, 24).values[:, None]
    return held(pick, 72)


def sw19(D, k=3):
    """Daily reversal: the day's 3 biggest losers among the 20 most traded coins revert the next
    day (liquidity provision, Lehmann 1990); picked at 00:00 UTC, held 24h."""
    liquid = D.qv.rolling(720).sum().rank(axis=1, ascending=False) <= 20
    pick = rank_pick(D.c / D.c.shift(24) - 1, D.ok & liquid, k, lowest=True) & every(D, 24).values[:, None]
    return held(pick, 24)


def sw20(D):
    """Volatility-managed memes: volatility does not predict returns, so holding the meme
    basket at less size when it is wild raises the Sharpe ratio (Moreira and Muir 2017)."""
    cols = basket(D, MEMES_1H)
    b = D.r[cols].where(D.ok[cols]).mean(axis=1)
    scale = (b.rolling(720).std() / b.rolling(168).std()).clip(upper=1.0).fillna(0)
    sig = pd.DataFrame(0.0, index=D.idx, columns=D.c.columns)
    sig[cols] = D.ok[cols].astype(float).mul(scale, axis=0)
    return sig


def sw21(D):
    """State-dependent markets: in volatile markets moves continue, in calm ones they revert;
    use the breakout (SW01) when BTC's weekly volatility is above its monthly median, else
    the dip-buyer (SW13)."""
    v = D.sigma(168)["BTC/USD"]
    wild = (v > v.rolling(720).median()).values[:, None]
    return (sw01(D) & wild) | (sw13(D) & ~wild)


def sw22(D):
    """Sentiment extremes: extreme fear marks capitulation and precedes rebounds (contrarian
    sentiment); hold the majors from a Fear & Greed reading of 20 or less until it reaches 50."""
    fg = daily_series("data/fear_greed.csv", "value", D)
    on = state(pd.DataFrame({"x": fg <= 20}), pd.DataFrame({"x": fg >= 50}))["x"]
    return only(pd.DataFrame(np.repeat(on.values[:, None], len(D.c.columns), axis=1), index=D.idx,
                             columns=D.c.columns), MAJORS) > 0


def sw23(D):
    """New money: stablecoin issuance is fresh buying power entering crypto; hold the majors
    while total stablecoin supply grew over the last week."""
    s = daily_series("data/stablecoins.csv", "usd", D)
    on = s / s.shift(168) - 1 > 0
    return only(pd.DataFrame(np.repeat(on.values[:, None], len(D.c.columns), axis=1), index=D.idx,
                             columns=D.c.columns), MAJORS) > 0


# ---- medium-frequency strategies (5-minute bars) ---------------------------------------

def mf01(D, bars=12):
    """Intraday momentum: a strong 1-hour move (1.5 sigma) continues for the next hour as news
    diffuses and traders herd (intraday time-series momentum)."""
    z = D.r.rolling(bars).sum() / (D.sigma(24) * np.sqrt(bars))
    return held(z > 1.5, 12)


def mf02(D):
    """US opening range breakout: price discovery after the US open sets the day's direction;
    long above the 13:30-14:00 UTC range, out at 20:00 UTC."""
    t = D.idx + pd.Timedelta(minutes=5)
    minutes = t.hour * 60 + t.minute
    in_range = pd.Series((minutes > 13 * 60 + 30) & (minutes <= 14 * 60), index=D.idx)
    day = pd.Series(t.normalize(), index=D.idx)
    hi = D.h.where(np.broadcast_to(in_range.values[:, None], D.h.shape)).groupby(day.values).transform("max")
    window = pd.Series((minutes > 14 * 60) & (minutes <= 20 * 60), index=D.idx)
    e = (D.c > hi) & window.values[:, None]
    return state(e, ~pd.DataFrame(np.repeat(window.values[:, None], len(D.c.columns), axis=1),
                                  index=D.idx, columns=D.c.columns))


def mf03(D, mult=5.0):
    """Attention spikes: a 5-minute bar on 5x normal volume closing at a 2-hour high is a burst
    of new buyers that carries on briefly (momentum ignition); held 30 minutes."""
    e = (D.qv > mult * D.qv.rolling(288).median()) & (D.c >= D.h.shift(1).rolling(24).max())
    return held(e, 6)


def mf04(D, entry_bars=48, exit_bars=12):
    """Intraday breakout: a break of the 4-hour range starts an intraday trend; out below the
    last hour's low."""
    return state(D.c > D.h.shift(1).rolling(entry_bars).max(), D.c < D.l.shift(1).rolling(exit_bars).min())


def mf05(D, fast=24, slow=63):
    """Micro-trends in memes: meme coins trend for hours on hype, so follow a fast EMA cross
    (8/21 on 15-minute bars)."""
    sig = ema(D.c, fast) > ema(D.c, slow)
    return only(sig, basket(D, MEMES_5M)) > 0


def mf06(D):
    """Momentum ignition: three rising 5-minute bars on rising volume mark buyers arriving in
    waves; ride it for 15 minutes."""
    up = (D.r > 0) & (D.r.shift(1) > 0) & (D.r.shift(2) > 0)
    vup = (D.qv > D.qv.shift(1)) & (D.qv.shift(1) > D.qv.shift(2))
    return only(held(up & vup, 3), basket(D, MEMES_5M)) > 0


def mf07(D, thresh=0.03):
    """Intraday meme waves: when the meme basket gains 3% in an hour, retail herding carries
    the whole group further; hold all memes for an hour."""
    cols = basket(D, MEMES_5M)
    b = D.r[cols].rolling(12).sum().where(D.ok[cols]).mean(axis=1)
    sig = pd.DataFrame(False, index=D.idx, columns=D.c.columns)
    sig[cols] = held(pd.DataFrame(np.repeat((b > thresh).values[:, None], len(cols), axis=1),
                                  index=D.idx, columns=cols), 12)
    return sig


def mf08(D, z=2.0):
    """DOGE leads memes: the biggest meme coin moves first and attention spills over to the
    smaller ones minutes later; after a 2-sigma 15-minute DOGE jump, hold the others 30 min."""
    r3 = D.r.rolling(3).sum()
    dz = r3["DOGE/USD"] / (D.sigma(24)["DOGE/USD"] * np.sqrt(3))
    cols = [c for c in basket(D, MEMES_5M) if c != "DOGE/USD"]
    sig = pd.DataFrame(False, index=D.idx, columns=D.c.columns)
    sig[cols] = held(pd.DataFrame(np.repeat((dz > z).values[:, None], len(cols), axis=1), index=D.idx,
                                  columns=cols), 6)
    return sig


def mf09(D, z=3.0):
    """BTC-to-altcoin lag: arbitrage bots carry BTC moves to altcoins with a short delay; after a
    3-sigma 5-minute BTC jump, buy the altcoins that moved less than half as much; 15 minutes."""
    btc = D.r["BTC/USD"] / D.sigma(24)["BTC/USD"]
    lag = D.r.lt(0.5 * D.r["BTC/USD"], axis=0) & D.ok
    lag["BTC/USD"] = False
    return held(lag & (btc > z).values[:, None], 3)


def mf10(D, z=4.0):
    """Price pressure: a 4-sigma 5-minute move on ordinary volume is one large order pushing
    the price, not news, so it reverts (microstructure price pressure); fade it 15 minutes."""
    zs = D.r / D.sigma(24)
    quiet = D.qv < 2 * D.qv.rolling(288).median()
    up, down = held((zs > z) & quiet, 3), held((zs < -z) & quiet, 3)
    return down.astype(float) - up.astype(float)


def mf11(D, z=2.0):
    """VWAP reversion: intraday traders anchor on the volume-weighted price, so a 2-sigma
    discount to the 4-hour VWAP closes; out at the VWAP."""
    vw = (D.c * D.vol).rolling(48).sum() / D.vol.rolling(48).sum()
    dev = D.c / vw - 1
    zs = dev / dev.rolling(288).std()
    return state(zs < -z, zs > 0)


def mf12(D):
    """Pullbacks in intraday uptrends: a very oversold 15-minute RSI in a coin above its 2-day
    EMA is a dip that bounces (RSI-2 on 15-minute bars)."""
    return state((rsi(D.c, 6) < 10) & (D.c > ema(D.c, 576)), rsi(D.c, 6) > 60)


def adx(D, n):
    up, down = D.h.diff(), -D.l.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    a = atr(D, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def mf13(D):
    """Ranges revert: when there is no trend (ADX below 20), a close 2.5 sigma below the 5-hour
    mean bounces back to it (range trading)."""
    ma, sd = D.c.rolling(60).mean(), D.c.rolling(60).std()
    return state((adx(D, 42) < 20) & (D.c < ma - 2.5 * sd), D.c > ma)


def mf14(D):
    """Failed breakouts: a new 4-hour high that falls back below the old high within 15 minutes
    was a stop run, and the trapped buyers' selling takes it lower; short for an hour."""
    level = D.h.shift(4).rolling(48).max()
    broke = (D.c.shift(3) > level) | (D.c.shift(2) > level) | (D.c.shift(1) > level)
    failed = broke & (D.c < level)
    return -held(failed, 12).astype(float)


def mf15(D):
    """Meme pairs intraday: same-narrative coins drift apart for minutes on one-off orders and
    close the gap within hours; 2.5-sigma divergences over 4 hours."""
    return pairs_signal(D, 2.5, 0.5, 48)


def mf16(D, k=2):
    """Hourly reversal in memes: retail noise overshoots, so the hour's 2 worst memes beat the
    rest the next hour (liquidity provision); rebalanced hourly."""
    cols = basket(D, MEMES_5M)
    mask = D.ok.copy()
    mask[[c for c in D.c.columns if c not in cols]] = False
    pick = rank_pick(D.r.rolling(12).sum(), mask, k, lowest=True) & every(D, 1).values[:, None]
    return held(pick, 12)


def mf17(D, k=2):
    """Hourly momentum in memes: the opposite view, that the hour's 2 best memes keep leading
    the next hour (herding); rebalanced hourly."""
    cols = basket(D, MEMES_5M)
    mask = D.ok.copy()
    mask[[c for c in D.c.columns if c not in cols]] = False
    pick = rank_pick(D.r.rolling(12).sum(), mask, k) & every(D, 1).values[:, None]
    return held(pick, 12)


def taker_share(D, bars):
    return D.taker.rolling(bars).sum() / D.vol.rolling(bars).sum()


def mf18(D, hi=0.6):
    """Order-flow imbalance: 15 minutes of mostly aggressive buying in a rising coin predicts
    the next minutes' returns (Cont, Kukanov and Stoikov 2014); held 30 minutes."""
    return held((taker_share(D, 3) > hi) & (D.c > D.c.shift(3)), 6)


def mf19(D):
    """Buying climax: extreme aggressive buying (75%) after a sharp hour up is buyers exhausting
    themselves; it reverses; short 30 minutes."""
    z = D.r.rolling(12).sum() / (D.sigma(24) * np.sqrt(12))
    return -held((taker_share(D, 3) > 0.75) & (z > 2.5), 6).astype(float)


def mf20(D, mult=5.0):
    """Retail bursts: a 5-minute bar with 5x the usual number of trades is a burst of small
    traders arriving, and they keep pushing the same way for 15 minutes."""
    burst = D.trades > mult * D.trades.rolling(288).median()
    up, down = held(burst & (D.r > 0), 3), held(burst & (D.r < 0), 3)
    return up.astype(float) - down.astype(float)


def mf21(D):
    """Liquidity premium: when the effective spread (Corwin-Schultz) triples after a drop,
    liquidity providers are scarce, and those who step in are paid as it recovers; 1 hour."""
    beta = (np.log(D.h / D.l) ** 2) + (np.log(D.h.shift(1) / D.l.shift(1)) ** 2)
    hi2, lo2 = pd.concat([D.h, D.h.shift(1)]).groupby(level=0).max(), pd.concat([D.l, D.l.shift(1)]).groupby(level=0).min()
    gamma = np.log(hi2.reindex(D.idx) / lo2.reindex(D.idx)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).rolling(12).mean()
    e = (spread > 3 * spread.rolling(288).median()) & (D.r.rolling(12).sum() < 0)
    return held(e, 12)


def mf22(D, hot=0.0002):
    """Funding avoidance: longs close before paying a high funding rate at 00, 08 and 16 UTC,
    pushing prices down in the hour before; short that hour when funding is high."""
    table = funding_table(72)
    f = pd.DataFrame.from_dict(table, orient="index")
    f.index = pd.to_datetime(f.index, unit="ms", utc=True)
    f = f.sort_index().reindex(D.idx, method="ffill").reindex(columns=D.c.columns)
    before = at_hours(D, [23, 7, 15]) | pd.Series(((D.idx + pd.Timedelta(minutes=5)).hour.isin([0, 8, 16])) &
                                                  ((D.idx + pd.Timedelta(minutes=5)).minute == 0), index=D.idx)
    return -((f > hot) & before.values[:, None]).astype(float)


def mf23(D, k=0.5):
    """Volatility breakout (Larry Williams): a day that rises half of yesterday's range above
    its open is a trending day; long from the break to the day's end."""
    t = D.idx + pd.Timedelta(minutes=5)
    day = t.normalize()
    g = D.c.groupby(day.values)
    d_open = D.o.groupby(day.values).transform("first")
    d_hi, d_lo = D.h.groupby(day.values).max(), D.l.groupby(day.values).min()
    rng = (d_hi - d_lo).shift(1)
    rng = rng.reindex(day.values).set_axis(D.idx)
    e = D.c > d_open + k * rng
    last = pd.Series((t.hour == 0) & (t.minute == 0), index=D.idx)
    del g
    return state(e & ~last.values[:, None], pd.DataFrame(np.repeat(last.values[:, None], len(D.c.columns), axis=1),
                                                         index=D.idx, columns=D.c.columns))


def mf24(D):
    """Buy pullbacks in trends: in a 2-day uptrend, a dip below the 5-hour EMA is a pause, not
    a reversal; long until 1% above that EMA or the trend breaks."""
    up = ema(D.c, 576) > ema(D.c, 1728)
    e5 = ema(D.c, 63)
    return state(up & (D.c < 0.995 * e5), (D.c > 1.01 * e5) | ~up)


def mf25(D):
    """State-dependent intraday markets: in wild days moves continue (MF01), in calm days they
    revert (MF11); switch on the coin's daily volatility against its 30-day median."""
    v = D.sigma(24)
    wild = v > v.rolling(8640).median()
    return (mf01(D) & wild) | (mf11(D) & ~wild)


def mf26(D, t_min=2.0):
    """Denoised micro-trend: a 2-hour drift that is statistically clear (a t-statistic over 2)
    is a real trend, not noise; long while it lasts."""
    m = D.r.ewm(span=24, adjust=False).mean()
    s = D.r.ewm(span=24, adjust=False).std()
    t = m / (s / np.sqrt(24))
    return state(t > t_min, t < 0)


def mf27(D, z=4.0):
    """Jumps carry news: a 4-sigma jump on 3x volume is information arriving, and prices drift
    after it as it spreads (post-jump drift); follow it for an hour. (MF10 tests the opposite
    view for jumps without volume.)"""
    bip = (D.r.abs() * D.r.abs().shift(1)).rolling(288).mean() * (np.pi / 2)
    zs = D.r / np.sqrt(bip)
    loud = D.qv > 3 * D.qv.rolling(288).median()
    up, down = held((zs > z) & loud, 12), held((zs < -z) & loud, 12)
    return up.astype(float) - down.astype(float)


def mf28(D):
    """Thin weekends: liquidity is thinner at weekends, so price pressure from single orders is
    larger and reverts more; fade 3-sigma 5-minute moves on Saturdays and Sundays (15 min)."""
    wk = pd.Series((D.idx + pd.Timedelta(minutes=5)).dayofweek.isin([5, 6]), index=D.idx).values[:, None]
    zs = D.r / D.sigma(24)
    up, down = held((zs > 3) & wk, 3), held((zs < -3) & wk, 3)
    return down.astype(float) - up.astype(float)


# Each: id, freq, function, settings fixed in advance, two neighbours.
REGISTRY = [
    ("SW01 breakout continuation", "1h", sw01, {}, [{"entry_h": 48, "exit_h": 16}, {"entry_h": 120, "exit_h": 36}]),
    ("SW02 short-horizon time-series momentum", "1h", sw02, {}, [{"look": 12}, {"look": 48}]),
    ("SW03 volume-confirmed breakout", "1h", sw03, {}, [{"mult": 2.0}, {"mult": 4.0}]),
    ("SW04 volatility squeeze breakout", "1h", sw04, {}, [{"q": 0.1}, {"q": 0.3}]),
    ("SW05 meme waves", "1h", sw05, {}, [{"thresh": 0.03}, {"thresh": 0.08}]),
    ("SW06 meme laggard catch-up", "1h", sw06, {}, [{"thresh": 0.03}, {"thresh": 0.08}]),
    ("SW07 BTC leads altcoins", "1h", sw07, {}, [{"z": 1.5}, {"z": 2.5}]),
    ("SW08 order-flow pressure", "1h", sw08, {}, [{"hi": 0.53}, {"hi": 0.58}]),
    ("SW09 short-squeeze fuel (funding)", "1h", sw09, {}, [{}, {}]),
    ("SW10 weekend retail (memes)", "1h", sw10, {}, [{}, {}]),
    ("SW11 US-session memes", "1h", sw11, {}, [{}, {}]),
    ("SW12 Asian-session majors", "1h", sw12, {}, [{}, {}]),
    ("SW13 dips in uptrends (RSI-3)", "1h", sw13, {}, [{"level": 5}, {"level": 15}]),
    ("SW14 capitulation reversal", "1h", sw14, {}, [{"z": 3.0}, {"z": 5.0}]),
    ("SW15 post-pump fade (short)", "1h", sw15, {}, [{"pump": 0.2}, {"pump": 0.35}]),
    ("SW16 dips in leaders", "1h", sw16, {}, [{"drop": 0.06}, {"drop": 0.10}]),
    ("SW17 meme pairs reversion", "1h", sw17, {}, [{}, {}]),
    ("SW18 weekly reversal (long losers in uptrends)", "1h", sw18, {}, [{"k": 2}, {"k": 5}]),
    ("SW19 daily reversal", "1h", sw19, {}, [{"k": 2}, {"k": 5}]),
    ("SW20 volatility-managed memes", "1h", sw20, {}, [{}, {}]),
    ("SW21 volatility regime switch", "1h", sw21, {}, [{}, {}]),
    ("SW22 extreme fear contrarian", "1h", sw22, {}, [{}, {}]),
    ("SW23 stablecoin inflows", "1h", sw23, {}, [{}, {}]),
    ("MF01 intraday momentum", "5m", mf01, {}, [{"bars": 6}, {"bars": 24}]),
    ("MF02 US opening range breakout", "5m", mf02, {}, [{}, {}]),
    ("MF03 volume-spike breakout", "5m", mf03, {}, [{"mult": 4.0}, {"mult": 7.0}]),
    ("MF04 4-hour range breakout", "5m", mf04, {}, [{"entry_bars": 36, "exit_bars": 9}, {"entry_bars": 72, "exit_bars": 18}]),
    ("MF05 meme micro-trend (EMA 8/21 on 15m)", "5m", mf05, {}, [{"fast": 12, "slow": 36}, {"fast": 48, "slow": 126}]),
    ("MF06 momentum ignition (memes)", "5m", mf06, {}, [{}, {}]),
    ("MF07 intraday meme waves", "5m", mf07, {}, [{"thresh": 0.02}, {"thresh": 0.05}]),
    ("MF08 DOGE leads memes", "5m", mf08, {}, [{"z": 1.5}, {"z": 2.5}]),
    ("MF09 BTC-to-altcoin lag", "5m", mf09, {}, [{"z": 2.5}, {"z": 4.0}]),
    ("MF10 price-pressure fade", "5m", mf10, {}, [{"z": 3.0}, {"z": 5.0}]),
    ("MF11 VWAP reversion", "5m", mf11, {}, [{"z": 1.5}, {"z": 2.5}]),
    ("MF12 intraday RSI-2 pullbacks", "5m", mf12, {}, [{}, {}]),
    ("MF13 range reversion (low ADX)", "5m", mf13, {}, [{}, {}]),
    ("MF14 failed breakout short", "5m", mf14, {}, [{}, {}]),
    ("MF15 meme pairs intraday", "5m", mf15, {}, [{}, {}]),
    ("MF16 hourly reversal in memes", "5m", mf16, {}, [{"k": 1}, {"k": 3}]),
    ("MF17 hourly momentum in memes", "5m", mf17, {}, [{"k": 1}, {"k": 3}]),
    ("MF18 order-flow imbalance", "5m", mf18, {}, [{"hi": 0.57}, {"hi": 0.65}]),
    ("MF19 buying climax fade (short)", "5m", mf19, {}, [{}, {}]),
    ("MF20 retail trade bursts", "5m", mf20, {}, [{"mult": 4.0}, {"mult": 7.0}]),
    ("MF21 liquidity premium (spread spike)", "5m", mf21, {}, [{}, {}]),
    ("MF22 funding avoidance (short)", "5m", mf22, {}, [{"hot": 0.00015}, {"hot": 0.0003}]),
    ("MF23 volatility breakout (Larry Williams)", "5m", mf23, {}, [{"k": 0.4}, {"k": 0.6}]),
    ("MF24 pullbacks in intraday trends", "5m", mf24, {}, [{}, {}]),
    ("MF25 intraday regime switch", "5m", mf25, {}, [{}, {}]),
    ("MF26 denoised micro-trend", "5m", mf26, {}, [{"t_min": 1.5}, {"t_min": 2.5}]),
    ("MF27 post-jump drift", "5m", mf27, {}, [{"z": 3.0}, {"z": 5.0}]),
    ("MF28 thin-weekend fade", "5m", mf28, {}, [{}, {}]),
]


# ---- simulation and judging ------------------------------------------------------------

def weights(sig, D):
    s = sig.astype(float).where(D.ok, 0.0).fillna(0.0)
    active = (s.abs() > 0).sum(axis=1).clip(lower=MAX_SHARE)
    return s.div(active, axis=0)


def simulate(sig, D, cost):
    w = weights(sig, D)
    hold = w.shift(1).fillna(0.0)
    r = (D.c / D.c.shift(1) - 1).fillna(0.0)
    gross = (hold * r).sum(axis=1)
    trade = hold.diff().abs().fillna(0.0)
    fees = (trade * cost.reindex(w.columns).fillna(FEE + 0.001)).sum(axis=1)
    exposure = hold.abs().sum(axis=1)
    entries = ((hold.abs() > 0) & (hold.shift(1).abs() == 0)).sum(axis=1)
    return gross, fees, exposure, entries


def fold_stats(gross, fees, exposure, entries, start, end, bph):
    sl = slice(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=1))
    g, f = gross.loc[sl], fees.loc[sl]
    if len(g) == 0 or exposure.loc[sl].sum() == 0:
        return None
    net = (1 + g - f).cumprod()
    hourly = net[(net.index + pd.Timedelta(minutes=60 / bph)).minute == 0]
    pts = [(int((ts + pd.Timedelta(minutes=60 / bph)).value // 10 ** 6), float(v)) for ts, v in hourly.items()]
    st = summarize(pts, 1.0)
    n_entries = float(entries.loc[sl].sum())
    in_pos = float((exposure.loc[sl] > 0).sum())
    return {"ret": st["total_return"], "gross": float((1 + g).prod() - 1), "fees": float(f.sum()),
            "mdd": st["max_drawdown"], "comp": st["composite"], "w14": st.get("window_composite_median", 0.0),
            "exposure": float(exposure.loc[sl].mean()), "entries": n_entries,
            "hold_hours": in_pos / max(n_entries, 1) / bph}


def stage_a(by_fold):
    have = [y for y, s in by_fold.items() if s is not None]
    good = [y for y in have if by_fold[y]["ret"] > 0 and by_fold[y]["comp"] > 0]
    need = 5 if len(have) == 6 else max(len(have) - 1, 3)
    return len(have) >= 3 and len(good) >= need, len(good), len(have)


def run(D, cost, fn, params, bph):
    sig = fn(D, **params)
    g, f, e, n = simulate(sig, D, cost)
    return {s[:4]: fold_stats(g, f, e, n, s, en, bph) for s, en in FOLDS}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ""
    out_path = os.path.join(OUT, "stage_a.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for freq in ("1h", "5m"):
        todo = [r for r in REGISTRY if r[1] == freq and r[0] not in results and r[0].startswith(which)]
        if not todo:
            continue
        D = load_1h() if freq == "1h" else load_5m()
        cost = costs(list(D.c.columns))
        bph = D.bph
        for name, _, fn, params, _ in todo:
            by = run(D, cost, fn, params, bph)
            ok, good, have = stage_a(by)
            results[name] = {"folds": by, "stage_a": ok, "good": good, "have": have}
            json.dump(results, open(out_path, "w"))
            row = " ".join("%+6.0f%%" % (by[y]["ret"] * 100) if by[y] else "    --" for y in sorted(by))
            gross = " ".join("%+5.0f%%" % (by[y]["gross"] * 100) if by[y] else "   --" for y in sorted(by))
            trades = sum(by[y]["entries"] for y in by if by[y])
            hold = np.median([by[y]["hold_hours"] for y in by if by[y]] or [0])
            print("%-46s net %s | gross %s | %5.0f entries, held %5.1fh | %s %d/%d" % (
                name, row, gross, trades, hold, "PASS" if ok else "fail", good, have), flush=True)
        del D


if __name__ == "__main__" and "--stage-b" not in sys.argv:
    main()


# ---- stage B: the survivors beside the live bot ----------------------------------------
# Written down before running (only SW16 passed Stage A): neighbours must pass Stage A; 20% of
# equity in the strategy and 80% in the live bot (R54b, with the funding filter), rebalanced
# daily, must beat the bot alone on the yearly composite in at least 5 of 6 folds with a worst
# drawdown at most 2 points deeper; then both holdout years (2018-2020). Also, for the record,
# the medium-frequency strategies whose gross edge was positive in every fold, re-costed as if
# every order were a limit order at the 0.05% maker fee paying no spread: a best case, since
# real limit orders miss fills exactly when the price runs away.

R54B = {"strategy": {"book_mode": "long_short", "ls_trend": [240, 960], "short_exclude_external": True,
                     "short_stop_atr": 10.0}}


def bot_curve(fold):
    from bot.backtest import run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import HOUR_MS, BinanceClient, load_history
    from research.folds import candidates, ms
    import copy
    start, end = fold
    path = os.path.join(OUT, "r54b_%s.csv" % start)
    if not os.path.exists(path):
        cfg = load_config()
        apply_overrides(cfg, copy.deepcopy(R54B))
        s, e = ms(start), ms(end)
        client = BinanceClient()
        slip = candidates(cfg)
        bars = {p: load_history(client, p, s - cfg.backtest.warmup_bars * HOUR_MS, e, cfg.backtest.data_dir) for p in slip}
        bars = {p: b for p, b in bars.items() if b}
        r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                         monthly_universe=True, slippage_by_pair=slip, external_scores=funding_table(72))
        pd.DataFrame(r.curve, columns=["ts", "equity"]).to_csv(path, index=False)
    df = pd.read_csv(path)
    return pd.Series(df["equity"].values / 100000.0, index=pd.to_datetime(df["ts"], unit="ms", utc=True))


def curve_stats(curve):
    pts = [(int(ts.value // 10 ** 6), float(v)) for ts, v in curve.items()]
    st = summarize(pts, 1.0)
    return {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"],
            "w14": st.get("window_composite_median", 0.0)}


def sleeve_curve(D, cost, fn, params, start, end):
    g, f, _, _ = simulate(fn(D, **params), D, cost)
    sl = slice(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=1))
    net = (1 + g.loc[sl] - f.loc[sl]).cumprod()
    net.index = net.index + pd.Timedelta(hours=1)          # by bar close, like the bot's curve
    return net


def stage_b():
    from concurrent.futures import ProcessPoolExecutor
    from research.h50_long_short import blend
    from research.holdout2018 import HOLDOUT
    results = json.load(open(os.path.join(OUT, "stage_a.json")))
    survivors = [r for r in REGISTRY if results.get(r[0], {}).get("stage_a")]
    with ProcessPoolExecutor(max_workers=8) as pool:
        bots = dict(zip([f[0] for f in FOLDS + HOLDOUT], pool.map(bot_curve, FOLDS + HOLDOUT)))
    full = load_full(ALL_1H)
    D = Data({k: v.loc["2017-12-01":"2026-10-01"] for k, v in full.items()}, 1)
    cost = costs(list(D.c.columns))
    for name, freq, fn, params, neighbours in survivors:
        print("\n" + name)
        for i, p in enumerate(neighbours):
            by = run(D, cost, fn, p, 1)
            ok, good, have = stage_a(by)
            print("  neighbour %s: %s  -> Stage A %s (%d/%d)" % (p, " ".join(
                "%+.0f%%" % (by[y]["ret"] * 100) if by[y] else "--" for y in sorted(by)), "holds" if ok else "breaks", good, have))
        for label, folds in (("six folds", FOLDS), ("holdout", HOLDOUT)):
            better, rows, worst_b, worst_m = 0, [], 0.0, 0.0
            for start, end in folds:
                bot = bots[start]
                sleeve = sleeve_curve(D, cost, fn, params, start, end)
                b, m = curve_stats(bot), curve_stats(blend(bot, sleeve.reindex(bot.index).ffill().fillna(1.0), 0.20))
                better += m["comp"] > b["comp"]
                worst_b, worst_m = max(worst_b, b["mdd"]), max(worst_m, m["mdd"])
                rows.append("%s bot %+.0f%% c%.2f | with 20%% %+.0f%% c%.2f" % (start[:4], b["ret"] * 100, b["comp"],
                                                                         m["ret"] * 100, m["comp"]))
            print("  %s: %s\n    composite better %d/%d, worst drawdown %.0f%% (bot %.0f%%)" % (
                label, "\n    ".join([""] + rows), better, len(folds), worst_m * 100, worst_b * 100))
    D5 = load_5m()
    cost5 = pd.Series(0.0005, index=D5.c.columns)
    print("\nMedium-frequency strategies with a gross edge in every fold, at maker cost (0.05%, no spread):")
    for name, freq, fn, params, _ in REGISTRY:
        f = results[name]["folds"]
        if freq != "5m" or not all(v is None or v["gross"] > 0 for v in f.values()):
            continue
        by = run(D5, cost5, fn, params, 12)
        ok, good, have = stage_a(by)
        print("  %-40s %s -> %s" % (name, " ".join("%+.0f%%" % (by[y]["ret"] * 100) if by[y] else "--" for y in sorted(by)),
                                    "PASS" if ok else "fail"))


if __name__ == "__main__" and "--stage-b" in sys.argv:
    stage_b()
