"""H50: long-short strategies, screened before any goes into the bot.

The bot is long only because every short tried so far failed (H5, H13, round 19): shorts
opened after the trend filter had turned, so after much of the fall, and bear-market rallies
then squeezed them; a sleeve shorting the most volatile coins lost as they rallied in
2023-24. Roostoo's shorts are 1x: a short locks USD collateral equal to its size, so a book
that is long $50k and short $50k uses all of a $100k account, and a market-neutral book earns
half its long-short spread.

Written down before running. Each sleeve is a book of at most 100% gross, rebalanced daily at
00:00 UTC on signals known at the close of the day before, over the bot's universe (the 45
most traded Roostoo coins with spreads <= 0.1% and 1000+ hours of history, chosen monthly),
paying 0.1% plus half the spread on every change of position:

  S1 cross-sectional momentum   long the 3 best 336h returns, short the 3 worst, equal
                                weights, dollar neutral (Liu, Tsyvinski and Wu 2022). The
                                rotation already holds the winners; the question is the losers.
  S2 residual momentum          the same on the 336h return left after removing each coin's
                                BTC beta (720h), legs sized to cancel their beta. Rallies lift
                                both legs by their beta, the squeeze that broke earlier shorts
                                (Blitz, Huij and Martens 2011, momentum without market crashes)
  S3 trend long/short per coin  long every coin with a positive 336h return, short every coin
                                with a negative one, inverse-volatility weights (time-series
                                momentum, Moskowitz, Ooi and Pedersen 2012)
  S4 BTC trend long/short       long BTC while its 168h EMA is above its 672h EMA (the
                                rotation's filter), short it while below. BTC's bear rallies
                                are milder than altcoins'
  S5 new-listing decay          short coins listed on Binance less than a year ago while their
                                336h return is negative, up to 5 equally; supply unlocks and
                                airdrop selling weigh on new tokens. S5h hedges with an equal
                                long in BTC
  S6 bear-market weak alts      round 19's A1 for reference: while BTC's filter is off, short
                                the 2 weakest coins
  S7 BAB long-short             long the 3 least volatile coins, short the 3 most volatile
                                (H13's idea, both legs)

Each sleeve is reported alone per fold, then blended with the bot as it is (70/30), taking
15% or 30% of equity, rebalanced daily. A sleeve goes forward to the bot's own backtester only
if a blend beats the bot on the yearly composite in at least 5 of the 6 folds, with
neighbours (lookback 168h and 504h, 2 or 5 coins a side) not breaking it. The 2018-2020
holdout is not used here.

    python -m research.h50_long_short
"""
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.config import apply_overrides, load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from bot.metrics import summarize
from research.folds import FOLDS, candidates, ms
from research.fullbars import load

OUT = os.path.join("runs", "research", "h50")
FEE = 0.001


# ---- the bot as it is, hourly, per fold ------------------------------------------------

def incumbent_curve(job):
    """The bot's hourly equity for one fold, with config overrides (none: the bot as it is)."""
    tag, overrides, (start, end) = job
    path = os.path.join(OUT, "%s_%s.csv" % (tag, start))
    if os.path.exists(path):
        return start, path
    cfg = load_config()
    apply_overrides(cfg, copy.deepcopy(overrides))
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {}
    for pair in slippage:
        series = load_history(client, pair, warm, e, cfg.backtest.data_dir)
        if series:
            bars[pair] = series
    result = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage,
                          "taker", monthly_universe=True, slippage_by_pair=slippage)
    pd.DataFrame(result.curve, columns=["ts", "equity"]).to_csv(path, index=False)
    return start, path


def incumbents(folds, tag="incumbent", overrides=None):
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        paths = dict(pool.map(incumbent_curve, [(tag, overrides or {}, f) for f in folds]))
    out = {}
    for start, path in paths.items():
        df = pd.read_csv(path)
        out[start] = pd.Series(df["equity"].values / load_config().backtest.initial_cash,
                               index=pd.to_datetime(df["ts"], unit="ms", utc=True))
    return out


# ---- market data and the point-in-time universe -----------------------------------------

def market():
    cfg = load_config()
    slip = candidates(cfg)
    data = load(sorted(slip))
    close = data["close"].ffill(limit=6)
    qvol = data["quote_volume"]
    first = close.apply(lambda s: s.first_valid_index())
    # Monthly universe: the 45 most traded over 30 days with 1000+ hours of history (PAXG apart).
    month_starts = pd.date_range(close.index[0].normalize() + pd.offsets.MonthBegin(1), close.index[-1], freq="MS")
    vol30 = qvol.rolling(720, min_periods=1).sum()
    hist = close.notna().cumsum()
    member = pd.DataFrame(False, index=close.index, columns=close.columns)
    for i, m in enumerate(month_starts):
        prev = m - pd.Timedelta(hours=1)
        if prev not in close.index:
            continue
        ok = (hist.loc[prev] >= cfg.universe.min_history_bars) & close.loc[prev].notna()
        ok["PAXG/USD"] = False
        ranked = vol30.loc[prev][ok].sort_values(ascending=False).index[:cfg.universe.size]
        until = month_starts[i + 1] if i + 1 < len(month_starts) else close.index[-1] + pd.Timedelta(hours=1)
        member.loc[m:until - pd.Timedelta(hours=1), list(ranked)] = True
    member["BTC/USD"] = member["BTC/USD"] | close["BTC/USD"].notna()
    cost = pd.Series({p: FEE + max(cfg.backtest.taker_slippage, s) for p, s in slip.items()})
    return close, member, first, cost


# ---- daily target weights for each sleeve -----------------------------------------------

def daily(df):
    """Values at the close of each UTC day (the 23:00 bar), indexed by that day's date."""
    d = df[df.index.hour == 23]
    d.index = d.index.normalize()
    return d


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


class Signals:
    def __init__(self, close, member, first):
        self.close, self.first = close, first
        self.logret = np.log(close).diff()
        self.member = daily(member)
        self.c = daily(close)
        self.btc = close["BTC/USD"]

    def ret(self, h):
        return daily(self.close / self.close.shift(h) - 1)

    def vol(self, h=336):
        return daily(self.logret.rolling(h, min_periods=h // 2).std())

    def beta(self, h=720):
        b = self.logret["BTC/USD"]
        cov = self.logret.rolling(h, min_periods=h // 2).cov(b)
        return daily(cov.div(b.rolling(h, min_periods=h // 2).var(), axis=0))

    def btc_trend(self, fast=168, slow=672):
        return daily((ema(self.btc, fast) > ema(self.btc, slow)).astype(float))

    def age_days(self):
        idx = self.c.index
        return pd.DataFrame({p: (idx - f.normalize()).days if f is not None else 10**6
                             for p, f in self.first.items()}, index=idx)


def rank_select(score, member, k, top=True):
    s = score.where(member)
    r = s.rank(axis=1, ascending=not top, method="first")
    return (r <= k) & s.notna()


def xs_momentum(sig, look=336, k=3, residual=False):
    score = sig.ret(look)
    if residual:
        lr = np.log1p(score)
        beta = sig.beta()
        score = lr.sub(beta.mul(lr["BTC/USD"], axis=0))
    m = sig.member.copy()
    if residual:
        m["BTC/USD"] = False
    longs, shorts = rank_select(score, m, k, True), rank_select(score, m, k, False)
    w = longs.astype(float) - shorts.astype(float)
    if not residual:
        return w / (2 * k)
    beta = sig.beta().clip(0.2, 3.0)
    bl = (beta.where(longs)).mean(axis=1)
    bs = (beta.where(shorts)).mean(axis=1)
    # Long notional L and short notional S with L*bl = S*bs and L + S = 1.
    L = bs / (bl + bs)
    S = 1 - L
    return longs.astype(float).mul(L / k, axis=0) - shorts.astype(float).mul(S / k, axis=0)


def ts_momentum(sig, look=336, ema_pair=None):
    if ema_pair:
        fast, slow = ema_pair
        sign = daily(np.sign(ema(sig.close, fast) - ema(sig.close, slow)))
    else:
        sign = np.sign(sig.ret(look))
    m = sig.member.copy()
    inv = 1.0 / sig.vol()
    w = (sign * inv).where(m).fillna(0.0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def btc_trend(sig, fast=168, slow=672, short_only=False):
    on = sig.btc_trend(fast, slow)
    w = pd.DataFrame(0.0, index=on.index, columns=sig.c.columns)
    w["BTC/USD"] = -(1 - on) if short_only else (2 * on - 1)
    return w


def new_listing(sig, max_age=365, look=336, k=5, hedge=False):
    age = sig.age_days()
    r = sig.ret(look)
    m = sig.member.copy()
    m["BTC/USD"] = False
    cand = m & (age < max_age) & (r < 0)
    picks = rank_select(r.where(cand), cand, k, top=False)
    side = 0.5 if hedge else 1.0
    w = -picks.astype(float) * side / k
    if hedge:
        w["BTC/USD"] = -w.sum(axis=1)
    return w


def bear_weak_alts(sig, k=2):
    off = 1 - sig.btc_trend()
    r = sig.ret(336)
    m = sig.member.copy()
    m["BTC/USD"] = False
    picks = rank_select(r.where(r < 0), m & (r < 0), k, top=False)
    return -picks.astype(float).mul(off / k, axis=0)


def bab(sig, k=3):
    v = sig.vol()
    m = sig.member.copy()
    lows, highs = rank_select(-v, m, k, True), rank_select(v, m, k, True)
    return (lows.astype(float) - highs.astype(float)) / (2 * k)


# ---- simulation ----------------------------------------------------------------------

def simulate(weights, close, cost, start, end):
    """Hourly equity of a book holding `weights` (decided at each day's close) through the next
    day, paying `cost` on every change in position. Returns a Series indexed by bar close time."""
    w = weights.reindex(columns=close.columns).fillna(0.0)
    days = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=1), freq="D")
    c = close
    equity = 1.0
    prev_w = pd.Series(0.0, index=c.columns)
    out_idx, out_val = [], []
    cost = cost.reindex(c.columns).fillna(FEE + 0.0002)
    for d in days:
        decided = d - pd.Timedelta(days=1)
        target = w.loc[decided] if decided in w.index else pd.Series(0.0, index=c.columns)
        anchor_ts = d - pd.Timedelta(hours=1)          # the 23:00 bar of the day before
        if anchor_ts not in c.index:
            continue
        anchor = c.loc[anchor_ts]
        target = target.where(anchor.notna() & (anchor > 0), 0.0)
        equity *= 1 - float((target - prev_w).abs().mul(cost).sum())
        block = c.loc[d:d + pd.Timedelta(hours=23)]
        rel = (block / anchor - 1).fillna(0.0)
        mult = 1 + rel.mul(target, axis=1).sum(axis=1)
        out_idx.extend(block.index + pd.Timedelta(hours=1))
        out_val.extend(equity * mult.values)
        end_rel = rel.iloc[-1]
        m_end = float(mult.iloc[-1])
        equity *= m_end
        # Weights drift with prices: a long's exposure grows with its price, so does a short's.
        prev_w = target * (1 + end_rel) / m_end if m_end > 0 else target * 0
    return pd.Series(out_val, index=pd.DatetimeIndex(out_idx))


def stats(curve):
    pts = [(int(ts.value // 10**6), float(v)) for ts, v in curve.items()]
    st = summarize(pts, 1.0)
    return {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"],
            "w14": st.get("window_composite_median", 0.0)}


def blend(bot, sleeve, share):
    """Daily-rebalanced mix: each day starts with `share` of equity in the sleeve. Both curves
    start from 1.0 and are indexed by bar close; a day runs from 01:00 to the next 00:00."""
    idx = bot.index.intersection(sleeve.index)
    bot, sleeve = bot.loc[idx], sleeve.loc[idx]
    day = (idx - pd.Timedelta(hours=1)).normalize()
    b_ref = bot.groupby(day).last().shift(1).fillna(1.0).reindex(day).values
    s_ref = sleeve.groupby(day).last().shift(1).fillna(1.0).reindex(day).values
    growth = (1 - share) * bot.values / b_ref + share * sleeve.values / s_ref
    g = pd.Series(growth, index=idx)
    start = g.groupby(day).last().cumprod().shift(1).fillna(1.0).reindex(day).values
    return pd.Series(start * growth, index=idx)


def sleeves(sig):
    return {
        "S1 XS momentum 336h, 3 a side": xs_momentum(sig),
        "S1 / 168h": xs_momentum(sig, look=168),
        "S1 / 504h": xs_momentum(sig, look=504),
        "S1 / 2 a side": xs_momentum(sig, k=2),
        "S1 / 5 a side": xs_momentum(sig, k=5),
        "S2 residual momentum, beta neutral": xs_momentum(sig, residual=True),
        "S2 / 168h": xs_momentum(sig, look=168, residual=True),
        "S2 / 504h": xs_momentum(sig, look=504, residual=True),
        "S3 trend long/short per coin 336h": ts_momentum(sig),
        "S3 / 168h": ts_momentum(sig, look=168),
        "S3 / 504h": ts_momentum(sig, look=504),
        "S3 / EMA 168-672": ts_momentum(sig, ema_pair=(168, 672)),
        "S4 BTC trend long/short": btc_trend(sig),
        "S4 / short only": btc_trend(sig, short_only=True),
        "S4 / EMA 120-480": btc_trend(sig, 120, 480),
        "S4 / EMA 240-960": btc_trend(sig, 240, 960),
        "S5 new listings in downtrend, short": new_listing(sig),
        "S5 / age < 180 days": new_listing(sig, max_age=180),
        "S5 / age < 540 days": new_listing(sig, max_age=540),
        "S5 / 168h trend": new_listing(sig, look=168),
        "S5h new listings short, BTC long": new_listing(sig, hedge=True),
        "S6 bear-market weak alts (round 19 A1)": bear_weak_alts(sig),
        "S7 BAB long low-vol, short high-vol": bab(sig),
        "S7 / 5 a side": bab(sig, k=5),
    }


def main() -> None:
    pd.set_option("display.width", 250)
    folds = FOLDS
    bots = incumbents(folds)
    close, member, first, cost = market()
    sig = Signals(close, member, first)
    books = sleeves(sig)
    years = [f[0][:4] for f in folds]
    base = {f[0][:4]: stats(bots[f[0]]) for f in folds}
    rows = []
    print("bot as it is: " + "  ".join("%s %+.0f%% c%.2f" % (y, base[y]["ret"] * 100, base[y]["comp"]) for y in years))
    for name, w in books.items():
        per = {}
        for start, end in folds:
            y = start[:4]
            curve = simulate(w, close, cost, start, end)
            bot = bots[start]
            alone = stats(curve)
            daily_b = bot[bot.index.hour == 0].pct_change().dropna()
            daily_s = curve[curve.index.hour == 0].pct_change().dropna()
            corr = daily_b.corr(daily_s.reindex(daily_b.index))
            mix = {s: stats(blend(bot, curve, s)) for s in (0.15, 0.30)}
            per[y] = (alone, corr, mix)
        row = {"sleeve": name,
               "alone: return by year": " ".join("%+.0f%%" % (per[y][0]["ret"] * 100) for y in years),
               "worst DD": "%.0f%%" % (max(per[y][0]["mdd"] for y in years) * 100),
               "median comp": round(float(np.median([per[y][0]["comp"] for y in years])), 2),
               "corr": round(float(np.nanmean([per[y][1] for y in years])), 2)}
        for s in (0.15, 0.30):
            better = sum(per[y][2][s]["comp"] > base[y]["comp"] for y in years)
            better14 = sum(per[y][2][s]["w14"] > base[y]["w14"] for y in years)
            row["%d%%: comp better" % (s * 100)] = "%d/6" % better
            row["%d%%: 14d better" % (s * 100)] = "%d/6" % better14
            row["%d%%: worst DD" % (s * 100)] = "%.0f%%" % (max(per[y][2][s]["mdd"] for y in years) * 100)
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False, header=len(rows) == 1), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "screen.csv"), index=False)


if __name__ == "__main__" and "--stage2" not in sys.argv and "--stage3" not in sys.argv:
    main()


# ---- stage 2: the long-short book in place of the defensive book ------------------------
# Written down after the screen above, before running: the per-coin trend book was the only
# sleeve with positive returns in most years and a low correlation with the bot, but taking
# its capital from both books diluted the rotation. Here it replaces the defensive book: 70%
# rotation (the bot run with the rotation alone) plus 30% in the sleeve, rebalanced daily,
# against the bot as it is. 70% rotation plus 30% of the defensive book run alone checks the
# method (it should come close to the bot as it is). S3r is a variant fixed now: long coins in
# uptrends only while BTC's filter is on, short coins in downtrends only while it is off.

def regime_aligned(sig, fast=168, slow=672):
    sign = daily(np.sign(ema(sig.close, fast) - ema(sig.close, slow)))
    on = sig.btc_trend(fast, slow)
    sign = sign.where(sign > 0, 0).mul(on, axis=0) + sign.where(sign < 0, 0).mul(1 - on, axis=0)
    inv = 1.0 / sig.vol()
    w = (sign * inv).where(sig.member).fillna(0.0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def stage2() -> None:
    pd.set_option("display.width", 250)
    folds = FOLDS
    years = [f[0][:4] for f in folds]
    bots = incumbents(folds)
    rot = incumbents(folds, "rotation_only", {"strategy": {"rotation_weight": 0.99}})
    book = incumbents(folds, "book_only", {"strategy": {"rotation_weight": 0.0}})
    close, member, first, cost = market()
    sig = Signals(close, member, first)
    books = {
        "defensive book (method check)": None,
        "S3 trend long/short, EMA 168-672": ts_momentum(sig, ema_pair=(168, 672)),
        "S3 / EMA 120-480": ts_momentum(sig, ema_pair=(120, 480)),
        "S3 / EMA 240-960": ts_momentum(sig, ema_pair=(240, 960)),
        "S3 / 336h return": ts_momentum(sig),
        "S3r regime-aligned, EMA 168-672": regime_aligned(sig),
        "S3r / EMA 120-480": regime_aligned(sig, 120, 480),
        "S3r / EMA 240-960": regime_aligned(sig, 240, 960),
        "S4 BTC trend long/short, EMA 240-960": btc_trend(sig, 240, 960),
    }
    base = {y: stats(bots[f[0]]) for y, f in zip(years, folds)}
    print("bot as it is: " + "  ".join("%s %+.0f%% DD %.0f%% c%.2f 14d %.2f" % (
        y, base[y]["ret"] * 100, base[y]["mdd"] * 100, base[y]["comp"], base[y]["w14"]) for y in years))
    for name, w in books.items():
        alone, mixed = {}, {}
        for (start, end), y in zip(folds, years):
            curve = book[start] if w is None else simulate(w, close, cost, start, end)
            alone[y] = stats(curve)
            mixed[y] = stats(blend(rot[start], curve, 0.30))
        better = sum(mixed[y]["comp"] > base[y]["comp"] for y in years)
        better14 = sum(mixed[y]["w14"] > base[y]["w14"] for y in years)
        print("\n%s\n  alone:          %s\n  70/30 with it:  %s\n  composite better %d/6, 14-day better %d/6, worst DD %.0f%% (bot %.0f%%)" % (
            name,
            "  ".join("%s %+.0f%%" % (y, alone[y]["ret"] * 100) for y in years),
            "  ".join("%s %+.0f%% c%.2f 14d %.2f" % (y, mixed[y]["ret"] * 100, mixed[y]["comp"], mixed[y]["w14"]) for y in years),
            better, better14, max(mixed[y]["mdd"] for y in years) * 100, max(base[y]["mdd"] for y in years) * 100), flush=True)


if __name__ == "__main__" and "--stage2" in sys.argv:
    stage2()


# ---- stage 3: relative value on BTC dominance -------------------------------------------
# Written down before running (round 56's screen): altcoins as a group trend against BTC for
# months (token supply inflation, rotation of risk appetite). Index the universe's altcoins
# (equal weights, daily), take the trend of log(alt index / BTC) with EMAs, and:
#   D1  while the ratio falls, long BTC and short the altcoins; while it rises, the reverse
#   D2  only the first leg (long BTC / short altcoins while the ratio falls), flat otherwise
# Both beta-neutral: the altcoin side's notional is set so its BTC beta (720h) matches the BTC
# side, the two summing to the sleeve's 100%. D2d: D2 dollar-neutral (50/50). EMAs 240h/960h,
# neighbours 168h/672h and 336h/1344h. Judged like stage 2: 70% rotation plus 30% of it,
# against the bot as it is.

def dominance(sig, fast=240, slow=960, both=True, beta_neutral=True):
    lr = sig.logret
    m = sig.member_hourly
    alts = [p for p in lr.columns if p not in ("BTC/USD", "PAXG/USD")]
    alt_ret = lr[alts].where(m[alts]).mean(axis=1).fillna(0.0)
    ratio = alt_ret.cumsum() - lr["BTC/USD"].fillna(0.0).cumsum()
    falling = daily(((ema(ratio, fast) < ema(ratio, slow)).astype(float)).to_frame("x"))["x"]
    beta = sig.beta().clip(0.3, 3.0)
    mem = sig.member.copy()
    mem["BTC/USD"] = False
    mem["PAXG/USD"] = False
    n = mem.sum(axis=1).replace(0, np.nan)
    b = beta.where(mem).mean(axis=1).fillna(1.0)
    if beta_neutral:
        w_alt, w_btc = 1 / (1 + b), b / (1 + b)
    else:
        w_alt, w_btc = 0.5 + 0 * b, 0.5 + 0 * b
    side = falling * 2 - 1 if both else falling          # +1: long BTC / short alts
    w = -mem.astype(float).div(n, axis=0).mul(w_alt * side, axis=0)
    w["BTC/USD"] = w_btc * side
    return w.fillna(0.0)


def stage3() -> None:
    folds = FOLDS
    years = [f[0][:4] for f in folds]
    bots = incumbents(folds)
    rot = incumbents(folds, "rotation_only", {"strategy": {"rotation_weight": 0.99}})
    close, member, first, cost = market()
    sig = Signals(close, member, first)
    sig.member_hourly = member
    books = {
        "D1 dominance both ways, 240/960": dominance(sig),
        "D1 / 168/672": dominance(sig, 168, 672),
        "D1 / 336/1344": dominance(sig, 336, 1344),
        "D2 long BTC / short alts while alts bleed, 240/960": dominance(sig, both=False),
        "D2 / 168/672": dominance(sig, 168, 672, both=False),
        "D2 / 336/1344": dominance(sig, 336, 1344, both=False),
        "D2d dollar-neutral": dominance(sig, both=False, beta_neutral=False),
    }
    base = {y: stats(bots[f[0]]) for y, f in zip(years, folds)}
    for name, w in books.items():
        alone, mixed = {}, {}
        for (start, end), y in zip(folds, years):
            curve = simulate(w, close, cost, start, end)
            alone[y] = stats(curve)
            mixed[y] = stats(blend(rot[start], curve, 0.30))
        print("\n%s\n  alone:          %s\n  70/30 with it:  %s\n  composite better %d/6, 14-day better %d/6, worst DD %.0f%%" % (
            name, "  ".join("%s %+.0f%% DD %.0f%%" % (y, alone[y]["ret"] * 100, alone[y]["mdd"] * 100) for y in years),
            "  ".join("%s %+.0f%% c%.2f" % (y, mixed[y]["ret"] * 100, mixed[y]["comp"]) for y in years),
            sum(mixed[y]["comp"] > base[y]["comp"] for y in years),
            sum(mixed[y]["w14"] > base[y]["w14"] for y in years),
            max(mixed[y]["mdd"] for y in years) * 100), flush=True)


if __name__ == "__main__" and "--stage3" in sys.argv:
    stage3()
