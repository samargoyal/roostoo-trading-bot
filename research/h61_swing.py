"""H61: 30 swing strategies, holding days, from stated hypotheses (round 2 of H60's swing tests).

H60 found that most swing ideas with an edge before costs entered thousands of positions and
paid it away, and that the one survivor bought dips in momentum leaders (SW16). Written down
before running, these strategies decide on daily or 4-hour bars, hold for days, and exit on
stops, targets, trailing stops or time, as swing traders do. Same universe (the bot's 45 coins
plus PEPE, BONK, SHIB and 1000CHEEMS), same costs (0.1% plus half each pair's Roostoo spread)
and the same Stage A as H60. Positions are long only, at most a quarter of capital per coin.

Stage B, fixed now: a strategy is worth adopting if (B1) 70% rotation plus 30% of it, in place
of the long-short book, beats the live bot (R54b) on the yearly composite in at least 5 of 6
folds with a worst drawdown at most 2 points deeper, or (B2) 20% of it beside the live bot
does; then both neighbours pass Stage A and it beats the bot in both holdout years (2018-2020).

    python -m research.h61_swing            # Stage A
    python -m research.h61_swing --stage-b  # Stage B for the survivors
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from research.folds import FOLDS, funding_table
from research.fullbars import load as load_full
from research.h60_swing_mft import (ALL_1H, MAJORS, MEMES_1H, Data, basket, costs, curve_stats, daily_series, ema,
                                    fold_stats, rsi, simulate, stage_a)

OUT = os.path.join("runs", "research", "h61")


def load():
    f = load_full(ALL_1H)
    return Data({k: v.loc["2017-12-01":"2026-10-01"] for k, v in f.items()}, 1)


# ---- daily bars and the trade engine ---------------------------------------------------

class Daily:
    """UTC-day bars from the hourly ones; a daily value is known at the day's last hourly close."""

    def __init__(self, D):
        key = D.idx.normalize()
        self.c = D.c.groupby(key).last()
        self.h = D.h.groupby(key).max()
        self.l = D.l.groupby(key).min()
        self.v = D.qv.groupby(key).sum(min_count=1)
        self.r = np.log(self.c).diff()
        self.D = D

    def atr(self, n=14):
        prev = self.c.shift(1)
        tr = np.maximum(self.h - self.l, np.maximum((self.h - prev).abs(), (self.l - prev).abs()))
        return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    def event(self, daily_bool):
        """A daily event as True on the hourly bar that closes the day (23:00 UTC)."""
        D = self.D
        out = pd.DataFrame(False, index=D.idx, columns=D.c.columns)
        last = D.idx[D.idx.hour == 23]
        vals = daily_bool.reindex(last.normalize()).reindex(columns=D.c.columns).fillna(False).values
        out.loc[last] = vals
        return out

    def level(self, daily_values):
        """A daily value carried from the day's last hourly close until the next day's."""
        D = self.D
        last = D.idx[D.idx.hour == 23]
        frame = pd.DataFrame(np.nan, index=D.idx, columns=D.c.columns)
        frame.loc[last] = daily_values.reindex(last.normalize()).reindex(columns=D.c.columns).astype(float).values
        return frame.ffill()


def engine(D, entry, stop=None, target=None, trail=None, max_hours=None, exit_when=None):
    """Long trades from entry signals, exits checked on hourly closes. `stop`, `target` and
    `trail` are fractions (frames of per-bar values, read at entry) below the entry, above the
    entry and below the highest close since entry; `exit_when` a boolean frame."""
    c = D.c.values
    ent = entry.fillna(False).values & D.ok.values
    n, m = c.shape
    held = np.zeros((n, m), dtype=bool)
    st = stop.values if isinstance(stop, pd.DataFrame) else None
    tg = target.values if isinstance(target, pd.DataFrame) else None
    tr = trail.values if isinstance(trail, pd.DataFrame) else None
    ex = exit_when.fillna(False).values if exit_when is not None else None
    for j in range(m):
        inside, e_price, hi, t0, s_d, g_d, t_d = False, 0.0, 0.0, 0, 0.0, 0.0, 0.0
        for t in range(n):
            p = c[t, j]
            if inside:
                if p != p:
                    held[t, j] = True
                    continue
                hi = max(hi, p)
                out = ((st is not None and p < e_price * (1 - s_d))
                       or (tg is not None and p > e_price * (1 + g_d))
                       or (tr is not None and p < hi * (1 - t_d))
                       or (max_hours is not None and t - t0 >= max_hours)
                       or (ex is not None and ex[t, j]))
                if out:
                    inside = False
                else:
                    held[t, j] = True
                    continue
            if ent[t, j] and p == p:
                inside, e_price, hi, t0 = True, p, p, t
                s_d = st[t, j] if st is not None else 0.0
                g_d = tg[t, j] if tg is not None else 0.0
                t_d = tr[t, j] if tr is not None else 0.0
                if (st is not None and s_d != s_d) or (tg is not None and g_d != g_d) or (tr is not None and t_d != t_d):
                    inside = False
                    continue
                held[t, j] = True
    return pd.DataFrame(held, index=D.idx, columns=D.c.columns)


def const(D, x):
    return pd.DataFrame(x, index=D.idx, columns=D.c.columns)


def atr_frac(D, Y, k, n=14):
    """k daily ATRs as a fraction of the close, carried hourly."""
    return Y.level(k * Y.atr(n) / Y.c)


def momentum_rank(Y, days=14):
    m = Y.c / Y.c.shift(days) - 1
    return m.rank(axis=1, pct=True)


# ---- the strategies --------------------------------------------------------------------

def s24(D, Y, k=2.0):
    """Trend pullbacks: in a 4-hour uptrend (EMA 50 above EMA 200 on 4-hour bars), a dip below
    the 4-hour EMA 20 that closes back above it ends a pullback; trends resume after them
    (short-term reversal inside long-term momentum). Stop 2 ATRs, trail 3 ATRs."""
    up = ema(D.c, 200) > ema(D.c, 800)
    e20 = ema(D.c, 80)
    reclaim = (D.c > e20) & (D.c.shift(1) <= e20.shift(1)) & (D.l.rolling(12).min() < e20)
    return engine(D, up & reclaim, stop=atr_frac(D, Y, k), trail=atr_frac(D, Y, 1.5 * k), max_hours=240)


def s25(D, Y, level=35):
    """Oversold in an uptrend: a 4-hour RSI below 35 while the daily close is above its 50-day
    average is a dip, not a trend change; out at RSI 65 or after 7 days, stop 8%."""
    up = Y.level(Y.c > Y.c.rolling(50).mean()).fillna(0).astype(bool)
    r4 = rsi(D.c, 56)
    return engine(D, up & (r4 < level), stop=const(D, 0.08), max_hours=168, exit_when=r4 > 65)


def s26(D, Y, depth=1.5):
    """First pullback after a breakout: a coin making a new 30-day high draws breakout buyers,
    who buy its first dip of 1.5 daily ATRs within 10 days; out on a new high trail of 2 ATRs,
    after 10 days or at a stop 1 ATR below the dip."""
    hi30 = Y.c.rolling(30).max()
    broke = (Y.c >= hi30).astype(float).rolling(10).max() > 0
    dip = (hi30 - Y.c) > depth * Y.atr()
    first = dip & ~dip.shift(1, fill_value=False) & broke
    return engine(D, Y.event(first), stop=atr_frac(D, Y, 1.0), trail=atr_frac(D, Y, 2.0), max_hours=240)


def s27(D, Y, near=0.02):
    """Breakout and retest: old resistance becomes support (orders cluster at previous highs,
    Osler 2003); after a close above the 20-day high, a return to within 2% of that level that
    holds above it is bought; stop 1 ATR below the level, trail 2 ATRs, 10 days."""
    lvl = Y.h.shift(1).rolling(20).max()
    broke_lvl = lvl.where(Y.c > lvl).ffill(limit=10)
    retest = (Y.l <= broke_lvl * (1 + near)) & (Y.c > broke_lvl) & (Y.c.shift(1) > broke_lvl * (1 + near))
    return engine(D, Y.event(retest), stop=atr_frac(D, Y, 1.0), trail=atr_frac(D, Y, 2.0), max_hours=240)


def s28(D, Y):
    """Market structure: a higher swing low followed by a break of the last swing high means
    buyers are in control (Dow theory); stop under the higher low, trail 2 ATRs, 15 days."""
    lo_piv = (Y.l < Y.l.shift(1)) & (Y.l < Y.l.shift(-1)) & (Y.l < Y.l.shift(2)) & (Y.l < Y.l.shift(-2))
    lo_piv = lo_piv.shift(2, fill_value=False)                   # a pivot is known 2 days later
    pivots = Y.l.shift(2).where(lo_piv)
    lows = pivots.ffill()
    prev_low = pd.DataFrame({c: pivots[c].dropna().shift(1).reindex(pivots.index).ffill() for c in pivots.columns})
    hi_piv = ((Y.h > Y.h.shift(1)) & (Y.h > Y.h.shift(-1)) & (Y.h > Y.h.shift(2))
              & (Y.h > Y.h.shift(-2))).shift(2, fill_value=False)
    last_high = Y.h.shift(2).where(hi_piv).ffill()
    sig = (lows > prev_low) & (Y.c > last_high) & (Y.c.shift(1) <= last_high)
    stop = Y.level(((Y.c - lows) / Y.c).clip(lower=0.02, upper=0.25))
    return engine(D, Y.event(sig), stop=stop, trail=atr_frac(D, Y, 2.0), max_hours=360)


def s29(D, Y):
    """Volatility contraction near highs (Minervini's VCP): a coin within 10% of its 60-day high
    whose daily ATR is at a 60-day low is coiling; a close above the 20-day high starts the
    next leg. Stop 1.5 ATRs, trail 2.5 ATRs, 15 days."""
    a = Y.atr()
    coil = (a <= a.rolling(60).min() * 1.05) & (Y.c >= 0.9 * Y.c.rolling(60).max())
    sig = coil.astype(float).rolling(5).max().astype(bool) & (Y.c > Y.h.shift(1).rolling(20).max())
    return engine(D, Y.event(sig), stop=atr_frac(D, Y, 1.5), trail=atr_frac(D, Y, 2.5), max_hours=360)


def s30(D, Y):
    """NR7 inside day (Crabel 1990): the narrowest range of 7 days, inside the day before, is a
    contraction that resolves in a directional move; buy a break of its high the next day,
    stop at its low, out after 3 days."""
    rng = Y.h - Y.l
    nr7 = (rng <= rng.rolling(7).min()) & (Y.h < Y.h.shift(1)) & (Y.l > Y.l.shift(1))
    level_hi = Y.level(Y.h.where(nr7))
    lvl_lo = Y.level(Y.l.where(nr7))
    fresh = Y.level(nr7.astype(float)).fillna(0) > 0
    trigger = fresh & (D.c > level_hi)
    stop = ((D.c - lvl_lo) / D.c).clip(lower=0.01, upper=0.2)
    return engine(D, trigger, stop=stop, max_hours=72)


def s31(D, Y, z=2.0):
    """Volatility-scaled dips in leaders (H60's SW16, without its fixed 8%): a daily drop of 2
    sigma in a top-quintile 14-day momentum coin is a liquidity shock in a strong trend; out
    when it recovers the pre-dip close, after 5 days, or at a stop 2 sigma lower."""
    sig_d = Y.r.rolling(30).std()
    lead = momentum_rank(Y).shift(1) >= 0.8
    dip = (Y.r < -z * sig_d) & lead
    target = Y.level((np.exp(-Y.r) - 1).where(dip))                # back to the pre-dip close
    stop = Y.level((1 - np.exp(-z * sig_d)).where(dip))
    return engine(D, Y.event(dip), stop=stop, target=target, max_hours=120)


def s32(D, Y):
    """Relative strength under stress: on a day BTC falls 3%, the top-quintile momentum coins that
    fall less than half as much show strong demand and lead the rebound (O'Neil); held 5 days,
    stop 8%."""
    btc = Y.r["BTC/USD"]
    lead = momentum_rank(Y) >= 0.8
    resilient = lead & Y.r.gt(0.5 * btc, axis=0) & (btc < np.log(0.97)).values[:, None]
    resilient["BTC/USD"] = False
    return engine(D, Y.event(resilient), stop=const(D, 0.08), max_hours=120)


def s33(D, Y, k=5):
    """Leaders on pullbacks: the rotation's top 5 by 14-day return keep leading, and entering 5%
    below their 7-day high buys them cheaper than at the top; out at a new 7-day high, after 7
    days or at a 10% stop."""
    m = Y.c / Y.c.shift(14) - 1
    top = m.rank(axis=1, ascending=False) <= k
    hi7 = Y.c.rolling(7).max()
    pull = top & (Y.c <= 0.95 * hi7) & (m > 0)
    new_high = D.c > Y.level(hi7)
    return engine(D, Y.event(pull), stop=const(D, 0.10), max_hours=168, exit_when=new_high)


def s34(D, Y, share=0.6):
    """Market-wide capitulation: when 60% of coins fall 10% in a day, forced liquidations have
    pushed the whole market below value; buy the 10 most traded coins for 3 days."""
    falls = (Y.r < np.log(0.9)).sum(axis=1) / Y.c.notna().sum(axis=1)
    liquid = Y.v.rolling(30).sum().rank(axis=1, ascending=False) <= 10
    sig = liquid & (falls > share).values[:, None]
    return engine(D, Y.event(sig), max_hours=72)


def s35(D, Y, depth=0.25):
    """Stretched below the mean in a long uptrend: 25% below the 20-day average while above the
    200-day average is an overreaction in a healthy coin; out at the 20-day average, after
    10 days or at a 15% stop."""
    ma20, ma200 = Y.c.rolling(20).mean(), Y.c.rolling(200).mean()
    sig = (Y.c < (1 - depth) * ma20) & (Y.c > ma200)
    return engine(D, Y.event(sig), stop=const(D, 0.15), max_hours=240, exit_when=D.c > Y.level(ma20))


def s36(D, Y):
    """Seller exhaustion (Connors): three falling days totalling over 10% in a coin above its
    200-day average exhaust short-term sellers; out on the first close above the 5-day
    average, after 7 days or at a 12% stop."""
    down3 = (Y.r < 0) & (Y.r.shift(1) < 0) & (Y.r.shift(2) < 0) & (Y.r.rolling(3).sum() < np.log(0.9))
    sig = down3 & (Y.c > Y.c.rolling(200).mean())
    return engine(D, Y.event(sig), stop=const(D, 0.12), max_hours=168, exit_when=D.c > Y.level(Y.c.rolling(5).mean()))


def s37(D, Y):
    """52-week highs (George and Hwang 2004): traders anchor on the year's high and under-react
    when it breaks, so new 365-day highs keep rising; out below the 50-day average."""
    sig = Y.c >= Y.c.rolling(365, min_periods=300).max()
    return engine(D, Y.event(sig), exit_when=D.c < Y.level(Y.c.rolling(50).mean()))


def s38(D, Y):
    """Darvas boxes: a 20-day consolidation within 15% that breaks out on twice the usual volume
    resolves upwards; stop at the box bottom, trail 15%."""
    top, bottom = Y.h.shift(1).rolling(20).max(), Y.l.shift(1).rolling(20).min()
    box = (top / bottom - 1) < 0.15
    sig = box & (Y.c > top) & (Y.v > 2 * Y.v.rolling(20).median())
    stop = Y.level(((Y.c - bottom) / Y.c).clip(lower=0.03, upper=0.2))
    return engine(D, Y.event(sig), stop=stop, trail=const(D, 0.15), max_hours=480)


def s39(D, Y):
    """No overhead supply: above the all-time high there are no holders waiting to sell at break
    even (the disposition effect, Grinblatt and Han 2005), so new highs since listing run;
    held 20 days or until 20% off the peak."""
    sig = Y.c >= Y.c.expanding(min_periods=200).max()
    return engine(D, Y.event(sig), trail=const(D, 0.20), max_hours=480)


def s40(D, Y, growth=0.01):
    """Liquidity inflow surges: stablecoin supply up 1% in two weeks is new buying power arriving
    fast; hold the majors and the top momentum quintile until two-week growth turns negative."""
    s = daily_series("data/stablecoins.csv", "usd", D)
    g = s / s.shift(336) - 1
    lead = Y.level(momentum_rank(Y) >= 0.8).fillna(0).astype(bool)
    names = lead.copy()
    names[[p for p in MAJORS if p in names.columns]] = True
    on = (g > growth).values[:, None]
    off = (g < 0)
    return engine(D, names & on, exit_when=pd.DataFrame(np.repeat(off.values[:, None], D.c.shape[1], axis=1),
                                                        index=D.idx, columns=D.c.columns))


def s41(D, Y):
    """Fear in a bull market: when average funding across coins turns negative while BTC is above
    its 200-day average, traders are short into an uptrend and get squeezed; hold the majors
    7 days."""
    table = funding_table(72)
    f = pd.DataFrame.from_dict(table, orient="index")
    f.index = pd.to_datetime(f.index, unit="ms", utc=True)
    avg = f.sort_index().mean(axis=1)
    avg_d = avg.reindex(Y.c.index, method="ffill")
    bull = Y.c["BTC/USD"] > Y.c["BTC/USD"].rolling(200).mean()
    sig = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    cond = (avg_d < 0) & bull
    for p in MAJORS:
        if p in sig.columns:
            sig[p] = cond
    return engine(D, Y.event(sig), max_hours=168)


def s42(D, Y):
    """Fear in a bull market, by sentiment: a Fear & Greed reading under 30 while BTC is above its
    200-day average is a buying chance (H60's SW22 had no trend filter); hold the majors until
    the reading passes 60."""
    fg = daily_series("data/fear_greed.csv", "value", D)
    bull = Y.level(pd.DataFrame({p: Y.c["BTC/USD"] > Y.c["BTC/USD"].rolling(200).mean() for p in Y.c.columns}))
    ent = bull.fillna(0).astype(bool) & (fg < 30).values[:, None]
    ent[[p for p in ent.columns if p not in MAJORS]] = False
    return engine(D, ent, exit_when=pd.DataFrame(np.repeat((fg > 60).values[:, None], D.c.shape[1], axis=1),
                                                index=D.idx, columns=D.c.columns))


def s43(D, Y):
    """Altseason rotation: once altcoins as a group start beating BTC over a week, money keeps
    rotating into them for weeks; hold the top momentum quintile of altcoins while the
    basket's 7-day return beats BTC's, else BTC."""
    alts = [p for p in Y.c.columns if p != "BTC/USD"]
    rel = Y.r[alts].rolling(7).sum().mean(axis=1) - Y.r["BTC/USD"].rolling(7).sum()
    lead = momentum_rank(Y) >= 0.8
    pick = lead & (rel > 0).values[:, None]
    pick["BTC/USD"] = (rel <= 0)
    lvl = Y.level(pick.astype(float)).fillna(0).astype(bool)
    return lvl


def s44(D, Y, drop=0.03):
    """Weekend selloffs overshoot: thin weekend liquidity exaggerates falls in the majors, which
    recover on Monday and Tuesday; buy the majors on Sunday at 20:00 UTC after a weekend fall
    of 3%, held 48 hours."""
    t = D.idx + pd.Timedelta(hours=1)
    sunday20 = pd.Series((t.dayofweek == 6) & (t.hour == 20), index=D.idx)
    wk = D.c / D.c.shift(44) - 1
    ent = (wk < -drop) & sunday20.values[:, None]
    ent[[p for p in ent.columns if p not in MAJORS]] = False
    return engine(D, ent, max_hours=48)


def s45(D, Y):
    """Attention concentrates: within the meme sector, flows chase the hottest coin, so hold the
    single strongest meme by 3-day return while the meme basket is above its 7-day average;
    re-picked daily."""
    cols = basket(D, MEMES_1H)
    m3 = Y.c[cols] / Y.c[cols].shift(3) - 1
    b = Y.r[cols].mean(axis=1).cumsum()
    up = b > b.rolling(7).mean()
    best = m3.rank(axis=1, ascending=False) <= 1
    pick = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    pick[cols] = best & up.values[:, None]
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def s46(D, Y, drop=0.15):
    """Meme dips in meme uptrends: a 15% one-day drop in a meme coin whose 7-day trend is up is
    a shakeout of weak hands; held 2 days, stop 15%."""
    cols = basket(D, MEMES_1H)
    sig = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    up7 = Y.c[cols] > Y.c[cols].shift(7)
    sig[cols] = (Y.r[cols] < np.log(1 - drop)) & up7.shift(1, fill_value=False)
    return engine(D, Y.event(sig), stop=const(D, 0.15), max_hours=48)


def s47(D, Y):
    """Sector hype phases trend: when the meme basket closes at a 10-day high, the hype phase
    has begun; hold all memes until the basket closes below its 5-day low."""
    cols = basket(D, MEMES_1H)
    b = Y.r[cols].mean(axis=1).fillna(0).cumsum()
    on = b >= b.rolling(10).max()
    off = b <= b.rolling(5).min()
    st = pd.Series(np.nan, index=b.index)
    st[on] = 1.0
    st[off] = 0.0
    st = st.ffill().fillna(0)
    pick = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    pick[cols] = pd.DataFrame(np.repeat((st > 0).values[:, None], len(cols), axis=1), index=Y.c.index, columns=cols)
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def s48(D, Y):
    """ETH against BTC: capital rotates between the two largest coins in trends that last
    weeks; hold ETH while ETH/BTC is above its 20-day average, else BTC."""
    ratio = np.log(Y.c["ETH/USD"] / Y.c["BTC/USD"])
    eth_on = ratio > ratio.rolling(20).mean()
    pick = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    pick["ETH/USD"] = eth_on
    pick["BTC/USD"] = ~eth_on
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def s49(D, Y):
    """Daily time-series momentum on the majors (Moskowitz, Ooi and Pedersen 2012): long a major
    while its 20-day return is positive, re-checked daily. Low turnover by design."""
    on = Y.c / Y.c.shift(20) - 1 > 0
    on[[p for p in on.columns if p not in MAJORS]] = False
    return Y.level(on.astype(float)).fillna(0).astype(bool)


def s50(D, Y, z=3.0):
    """Capitulation, daily: a 3-sigma daily fall on 3x volume is forced selling that reverts over
    the next days (H60's SW14 on daily bars, so it trades less); held 3 days, stop 10%."""
    sd = Y.r.rolling(30).std()
    sig = (Y.r < -z * sd) & (Y.v > 3 * Y.v.rolling(30).median())
    return engine(D, Y.event(sig), stop=const(D, 0.10), max_hours=72)


def s51(D, Y):
    """The golden cross: the 50-day average crossing above the 200-day marks a new long trend
    that lasts months; out on the death cross or 25% off the peak."""
    m50, m200 = Y.c.rolling(50).mean(), Y.c.rolling(200).mean()
    cross = (m50 > m200) & (m50.shift(1) <= m200.shift(1))
    death = Y.level(m50 < m200).fillna(0).astype(bool)
    return engine(D, Y.event(cross), trail=const(D, 0.25), exit_when=death)


def s52(D, Y):
    """The Turtles' long system: a close above the 55-day high starts a trend, a close below the
    20-day low ends it (Dennis and Eckhardt's system 2)."""
    sig = Y.c > Y.h.shift(1).rolling(55).max()
    out = D.c < Y.level(Y.l.shift(1).rolling(20).min())
    return engine(D, Y.event(sig), exit_when=out)


def s53(D, Y):
    """Calm breakouts: a 20-day high in one of the least volatile fifth of coins is a cleaner
    signal than in a wild one (the low-volatility effect meets momentum); stop 2 ATRs, trail
    3 ATRs, 15 days."""
    sd = Y.r.rolling(30).std()
    calm = sd.rank(axis=1, pct=True) <= 0.2
    sig = calm & (Y.c > Y.h.shift(1).rolling(20).max())
    return engine(D, Y.event(sig), stop=atr_frac(D, Y, 2.0), trail=atr_frac(D, Y, 3.0), max_hours=360)


REGISTRY = [
    ("SW24 trend pullbacks (4h EMA reclaim)", s24, {}, [{"k": 1.5}, {"k": 2.5}]),
    ("SW25 oversold in an uptrend (4h RSI)", s25, {}, [{"level": 30}, {"level": 40}]),
    ("SW26 first pullback after a breakout", s26, {}, [{"depth": 1.0}, {"depth": 2.0}]),
    ("SW27 breakout and retest", s27, {}, [{"near": 0.01}, {"near": 0.03}]),
    ("SW28 higher low, broken high (structure)", s28, {}, [{}, {}]),
    ("SW29 volatility contraction near highs (VCP)", s29, {}, [{}, {}]),
    ("SW30 NR7 inside-day breakout", s30, {}, [{}, {}]),
    ("SW31 volatility-scaled dips in leaders", s31, {}, [{"z": 1.5}, {"z": 2.5}]),
    ("SW32 relative strength on BTC down days", s32, {}, [{}, {}]),
    ("SW33 leaders on pullbacks (top 5)", s33, {}, [{"k": 3}, {"k": 8}]),
    ("SW34 market-wide capitulation", s34, {}, [{"share": 0.5}, {"share": 0.7}]),
    ("SW35 stretched below the mean in an uptrend", s35, {}, [{"depth": 0.2}, {"depth": 0.3}]),
    ("SW36 seller exhaustion (3 down days)", s36, {}, [{}, {}]),
    ("SW37 52-week-high breakouts", s37, {}, [{}, {}]),
    ("SW38 Darvas box breakouts", s38, {}, [{}, {}]),
    ("SW39 all-time-high breakouts", s39, {}, [{}, {}]),
    ("SW40 stablecoin inflow surges", s40, {}, [{"growth": 0.005}, {"growth": 0.02}]),
    ("SW41 negative funding in a bull market", s41, {}, [{}, {}]),
    ("SW42 fear in a bull market (sentiment)", s42, {}, [{}, {}]),
    ("SW43 altseason rotation", s43, {}, [{}, {}]),
    ("SW44 weekend selloff reversal (majors)", s44, {}, [{"drop": 0.02}, {"drop": 0.05}]),
    ("SW45 hottest meme", s45, {}, [{}, {}]),
    ("SW46 meme dips in meme uptrends", s46, {}, [{"drop": 0.12}, {"drop": 0.2}]),
    ("SW47 meme hype phases", s47, {}, [{}, {}]),
    ("SW48 ETH against BTC", s48, {}, [{}, {}]),
    ("SW49 daily time-series momentum (majors)", s49, {}, [{}, {}]),
    ("SW50 daily capitulation reversal", s50, {}, [{"z": 2.5}, {"z": 3.5}]),
    ("SW51 golden cross", s51, {}, [{}, {}]),
    ("SW52 Turtles' 55/20-day system", s52, {}, [{}, {}]),
    ("SW53 calm breakouts", s53, {}, [{}, {}]),
]


def run(D, Y, cost, fn, params):
    sig = fn(D, Y, **params)
    g, f, e, n = simulate(sig, D, cost)
    return {s[:4]: fold_stats(g, f, e, n, s, en, 1) for s, en in FOLDS}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ""
    path = os.path.join(OUT, "stage_a.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    D = load()
    Y = Daily(D)
    cost = costs(list(D.c.columns))
    for name, fn, params, _ in REGISTRY:
        if name in results or not name.startswith(which):
            continue
        by = run(D, Y, cost, fn, params)
        ok, good, have = stage_a(by)
        results[name] = {"folds": by, "stage_a": ok, "good": good, "have": have}
        json.dump(results, open(path, "w"))
        row = " ".join("%+6.0f%%" % (by[y]["ret"] * 100) if by[y] else "    --" for y in sorted(by))
        gross = " ".join("%+5.0f%%" % (by[y]["gross"] * 100) if by[y] else "   --" for y in sorted(by))
        entries = sum(by[y]["entries"] for y in by if by[y])
        hold = np.median([by[y]["hold_hours"] for y in by if by[y]] or [0])
        print("%-44s net %s | gross %s | %5.0f entries, held %5.0fh | %s %d/%d" % (
            name, row, gross, entries, hold, "PASS" if ok else "fail", good, have), flush=True)


if __name__ == "__main__" and "--stage-b" not in sys.argv:
    main()


# ---- stage B ---------------------------------------------------------------------------

def rotation_curve(fold):
    """The rotation alone (rotation_weight 0.99), hourly, for B1."""
    from research.h50_long_short import incumbent_curve
    import research.h50_long_short as h50
    h50.OUT = OUT
    _, path = incumbent_curve(("rotation_only", {"strategy": {"rotation_weight": 0.99}}, fold))
    df = pd.read_csv(path)
    return pd.Series(df["equity"].values / 100000.0, index=pd.to_datetime(df["ts"], unit="ms", utc=True))


def stage_b():
    from concurrent.futures import ProcessPoolExecutor
    from research.h50_long_short import blend
    from research.h60_swing_mft import bot_curve
    from research.holdout2018 import HOLDOUT
    os.makedirs(OUT, exist_ok=True)
    results = json.load(open(os.path.join(OUT, "stage_a.json")))
    survivors = [r for r in REGISTRY if results.get(r[0], {}).get("stage_a")]
    folds = FOLDS + HOLDOUT
    with ProcessPoolExecutor(max_workers=8) as pool:
        bots = dict(zip([f[0] for f in folds], pool.map(bot_curve, folds)))
        rots = dict(zip([f[0] for f in folds], pool.map(rotation_curve, folds)))
    D = load()
    Y = Daily(D)
    cost = costs(list(D.c.columns))
    summary = []
    for name, fn, params, neighbours in survivors:
        g, f, _, _ = simulate(fn(D, Y, **params), D, cost)
        net_all = (1 + g - f)
        nb = []
        for p in neighbours:
            ok, good, have = stage_a(run(D, Y, cost, fn, p))
            nb.append("%s %d/%d" % ("holds" if ok else "breaks", good, have))
        line = {"name": name, "neighbours": nb}
        for label, fs in (("six", FOLDS), ("holdout", HOLDOUT)):
            b1 = b2 = 0
            dd_bot = dd1 = dd2 = 0.0
            rows = []
            for start, end in fs:
                sl = slice(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=1))
                sleeve = net_all.loc[sl].cumprod()
                sleeve.index = sleeve.index + pd.Timedelta(hours=1)
                bot, rot = bots[start], rots[start]
                s_b = sleeve.reindex(bot.index).ffill().fillna(1.0)
                s_r = sleeve.reindex(rot.index).ffill().fillna(1.0)
                sb, s1, s2 = curve_stats(bot), curve_stats(blend(rot, s_r, 0.30)), curve_stats(blend(bot, s_b, 0.20))
                b1 += s1["comp"] > sb["comp"]
                b2 += s2["comp"] > sb["comp"]
                dd_bot, dd1, dd2 = max(dd_bot, sb["mdd"]), max(dd1, s1["mdd"]), max(dd2, s2["mdd"])
                rows.append((start[:4], sb["ret"], s1["ret"], s2["ret"]))
            line[label] = {"b1": b1, "b2": b2, "n": len(fs), "dd": (dd_bot, dd1, dd2), "rows": rows}
        summary.append(line)
        six, ho = line["six"], line["holdout"]
        print("\n%s  neighbours: %s" % (name, ", ".join(nb)))
        for lab, d in (("six folds", six), ("holdout", ho)):
            print("  %s: B1 (70%% rotation + 30%% this) better %d/%d, B2 (20%% beside the bot) better %d/%d; "
                  "worst DD bot %.0f%%, B1 %.0f%%, B2 %.0f%%" % (lab, d["b1"], d["n"], d["b2"], d["n"], *[x * 100 for x in d["dd"]]))
            print("    " + "  ".join("%s bot %+.0f%% B1 %+.0f%% B2 %+.0f%%" % (y, a * 100, b * 100, c * 100) for y, a, b, c in d["rows"]))
    json.dump(summary, open(os.path.join(OUT, "stage_b.json"), "w"))


if __name__ == "__main__" and "--stage-b" in sys.argv:
    stage_b()
