"""H62: retail attention, from Wikipedia page views, as a trading signal.

Coins move on social media, but tweets cannot be collected years back for free. Wikipedia page
views can (research/attention.py): the academic proxy for retail attention. Written down before
running, 12 hypotheses on daily signals over the coins that have an English article. Views for
a day are published a few hours after it ends, so every decision uses views up to the day
before. Same costs (0.1% plus half each pair's Roostoo spread), same Stage A and Stage B as H61
(B1: in place of the long-short book; B2: 20% beside the live bot), neighbours and holdout.

    python -m research.h62_attention            # Stage A
    python -m research.h62_attention --stage-b  # Stage B for the survivors
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import MAJORS, costs, curve_stats, fold_stats, simulate, stage_a
from research.h61_swing import Daily, const, engine, load

warnings.filterwarnings("ignore")
OUT = os.path.join("runs", "research", "h62")
MEMES = ["DOGE/USD", "SHIB/USD", "PEPE/USD", "BONK/USD", "TRUMP/USD"]


def views(Y):
    """Daily views per pair (and MARKET) aligned to the price days, lagged a day (published late)."""
    w = pd.read_csv(os.path.join("data", "wiki_views.csv"), index_col=0, parse_dates=True)
    w.index = pd.to_datetime(w.index, utc=True).normalize()
    return w.reindex(Y.c.index).shift(1)


def abnormal(w, short=7, long=90):
    """Log of recent views over their usual level: > 0 means attention is rising."""
    return np.log(w.rolling(short, min_periods=short).mean() / w.rolling(long, min_periods=long // 2).median())


def frame(Y, cols, values):
    out = pd.DataFrame(np.nan, index=Y.c.index, columns=Y.c.columns)
    for c in cols:
        if c in values.columns and c in out.columns:
            out[c] = values[c]
    return out


def a01(D, Y, W, thresh=np.log(1.5)):
    """Attention inflows (Da, Engelberg and Gao 2011): when a rising coin's views run 50% above
    their 90-day norm, retail buying pressure builds for about two weeks; long while attention
    stays raised and the price holds its 20-day average."""
    aa = frame(Y, Y.c.columns, abnormal(W))
    ent = (aa > thresh) & (Y.c / Y.c.shift(14) > 1)
    out = Y.level((aa < 0) | (Y.c < Y.c.rolling(20).mean())).fillna(0).astype(bool)
    return engine(D, Y.event(ent), exit_when=out)


def a02(D, Y, W, spike=4.0):
    """Attention tops (Barber and Odean 2008): a day with 4x the usual views is attention-grabbing
    retail buying at a peak, which reverses; short for 7 days."""
    s = frame(Y, Y.c.columns, W / W.rolling(30, min_periods=15).median())
    hits = Y.event(s > spike)
    held_ = hits.astype(float).rolling(168, min_periods=1).max() > 0
    return -held_.astype(float)


def a03(D, Y, W):
    """Retail tides: market-wide attention (the 'Cryptocurrency' article) above its 90-day norm
    means retail money is flowing in; hold the majors while it lasts, else cash."""
    on = abnormal(W[["MARKET"]])["MARKET"] > 0
    lvl = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    for p in MAJORS:
        if p in lvl.columns and p in W.columns:
            lvl[p] = on
    return Y.level(lvl.astype(float)).fillna(0).astype(bool)


def a04(D, Y, W, k=2):
    """Momentum when retail is arriving: the 2 strongest coins by 14-day return, held only while
    market attention is above its norm (retail inflows feed momentum); re-picked daily."""
    on = abnormal(W[["MARKET"]])["MARKET"] > 0
    m = Y.c / Y.c.shift(14) - 1
    has = frame(Y, Y.c.columns, W).notna()
    pick = (m.where(has).rank(axis=1, ascending=False) <= k) & (m > 0) & on.values[:, None]
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def a05(D, Y, W, k=3):
    """Momentum confirmed by attention (Hou, Peng and Xiong 2009): among coins whose own attention
    is rising, the 3 strongest by 14-day return; re-picked daily."""
    aa = frame(Y, Y.c.columns, abnormal(W))
    m = Y.c / Y.c.shift(14) - 1
    pick = (m.where(aa > 0).rank(axis=1, ascending=False) <= k) & (m > 0)
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def a06(D, Y, W, k=3):
    """Quiet winners (Hong, Lim and Stein 2000): strong coins nobody is talking about are under-
    priced, as the news has not spread; the 3 strongest whose attention is falling."""
    aa = frame(Y, Y.c.columns, abnormal(W))
    m = Y.c / Y.c.shift(14) - 1
    pick = (m.where(aa < 0).rank(axis=1, ascending=False) <= k) & (m > 0)
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def a07(D, Y, W, fade=np.log(0.7)):
    """Trends need an audience: hold coins above their 50-day average, except while their
    attention has fallen 30% below its norm (a trend nobody follows any more ends)."""
    aa = frame(Y, Y.c.columns, abnormal(W))
    has = frame(Y, Y.c.columns, W).notna()
    on = (Y.c > Y.c.rolling(50).mean()) & has & ~(aa < fade)
    return Y.level(on.astype(float)).fillna(0).astype(bool)


def a08(D, Y, W, spike=2.0):
    """Meme waves start with attention: when DOGE, SHIB or PEPE views double their 30-day norm,
    retail is piling into memes; hold the meme coins for 3 days."""
    cols = [c for c in MEMES if c in W.columns]
    s = (W[cols] / W[cols].rolling(30, min_periods=15).median()).max(axis=1)
    sig = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    for c in MEMES:
        if c in sig.columns:
            sig[c] = s > spike
    return engine(D, Y.event(sig), max_hours=72)


def a09(D, Y, W):
    """Attention predicts volatility (Urquhart 2018): hold the majors at full size while their
    attention is normal and scale down in proportion when it spikes, to cut the wild days."""
    ratio = frame(Y, MAJORS, np.exp(abnormal(W)))
    scale = (1.0 / ratio.clip(lower=1.0)).where(ratio.notna(), np.nan)
    lvl = Y.level(scale).fillna(0)
    return lvl


def a10(D, Y, W):
    """Panic marks bottoms: a tripling of views while the price fell 15% in 3 days is fear, not
    greed, and capitulation reverses; long for 7 days, stop 12%."""
    s = frame(Y, Y.c.columns, W / W.rolling(30, min_periods=15).median())
    sig = (s > 3) & (Y.c / Y.c.shift(3) - 1 < -0.15)
    return engine(D, Y.event(sig), stop=const(D, 0.12), max_hours=168)


def a11(D, Y, W, share=0.6):
    """Broad participation: when 60% of coins have rising attention, retail interest is broad
    (a healthy bull); hold the 3 strongest coins by 14-day return, else cash."""
    aa = frame(Y, Y.c.columns, abnormal(W))
    breadth = (aa > 0).sum(axis=1) / aa.notna().sum(axis=1).replace(0, np.nan)
    m = Y.c / Y.c.shift(14) - 1
    has = aa.notna()
    pick = (m.where(has).rank(axis=1, ascending=False) <= 3) & (m > 0) & (breadth > share).values[:, None]
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def a12(D, Y, W):
    """Retail rotates to altcoins: when attention to altcoins rises against attention to Bitcoin,
    retail is moving down the risk curve; hold the 3 strongest altcoins, else BTC."""
    alts = [c for c in W.columns if c not in ("MARKET", "BTC/USD")]
    rel = np.log(W[alts].sum(axis=1) / W["BTC/USD"])
    rising = rel.rolling(7).mean() > rel.rolling(90, min_periods=45).median()
    m = Y.c / Y.c.shift(14) - 1
    has = frame(Y, Y.c.columns, W).notna()
    has["BTC/USD"] = False
    pick = (m.where(has).rank(axis=1, ascending=False) <= 3) & (m > 0) & rising.values[:, None]
    pick["BTC/USD"] = ~rising
    return Y.level(pick.astype(float)).fillna(0).astype(bool)


def short_engine(D, entry, stop=0.15, max_hours=336, exit_when=None):
    """Short trades from entry signals: out on a close `stop` above the entry (a squeeze), after
    max_hours, or when exit_when is true; returns -1 while short."""
    c = D.c.values
    ent = entry.fillna(False).values & D.ok.values
    ex = exit_when.fillna(False).values if exit_when is not None else None
    n, m = c.shape
    pos = np.zeros((n, m))
    for j in range(m):
        inside, e_price, t0 = False, 0.0, 0
        for t in range(n):
            p = c[t, j]
            if inside:
                if p != p:
                    pos[t, j] = -1.0
                    continue
                if p > e_price * (1 + stop) or t - t0 >= max_hours or (ex is not None and ex[t, j]):
                    inside = False
                else:
                    pos[t, j] = -1.0
                    continue
            if ent[t, j] and p == p:
                inside, e_price, t0 = True, p, t
                pos[t, j] = -1.0
    return pd.DataFrame(pos, index=D.idx, columns=D.c.columns)


def collapse_shorts(D, Y, W, fade=np.log(0.7)):
    aa = frame(Y, Y.c.columns, abnormal(W))
    ma20 = Y.c.rolling(20).mean()
    ent = (aa < fade) & (Y.c / Y.c.shift(14) < 1) & (Y.c < ma20)
    out = Y.level((aa > np.log(0.85)) | (Y.c > ma20)).fillna(0).astype(bool)
    return short_engine(D, Y.event(ent), exit_when=out)


def a13(D, Y, W, fade=np.log(0.7)):
    """Attention long-short: long rising coins whose attention surges (A01, retail arriving),
    short falling coins whose attention has collapsed 30% below normal (retail gone, no buyers
    left); shorts out on a 15% squeeze, after 14 days, or once attention or price recovers."""
    longs = a01(D, Y, W).astype(float)
    shorts = collapse_shorts(D, Y, W, fade)
    return longs.where(longs > 0, shorts)


def a14(D, Y, W, fade=np.log(0.7)):
    """A13's short side alone: does a collapse in attention mark coins that keep falling?"""
    return collapse_shorts(D, Y, W, fade)


REGISTRY = [
    ("A01 attention inflows (rising coins)", a01, {}, [{"thresh": np.log(1.3)}, {"thresh": np.log(2.0)}]),
    ("A02 attention tops (short spikes)", a02, {}, [{"spike": 3.0}, {"spike": 6.0}]),
    ("A03 retail tides (market attention)", a03, {}, [{}, {}]),
    ("A04 momentum while retail arrives", a04, {}, [{"k": 1}, {"k": 3}]),
    ("A05 momentum confirmed by attention", a05, {}, [{"k": 2}, {"k": 5}]),
    ("A06 quiet winners", a06, {}, [{"k": 2}, {"k": 5}]),
    ("A07 trends need an audience", a07, {}, [{"fade": np.log(0.6)}, {"fade": np.log(0.8)}]),
    ("A08 meme waves start with attention", a08, {}, [{"spike": 1.5}, {"spike": 3.0}]),
    ("A09 attention-scaled majors", a09, {}, [{}, {}]),
    ("A10 panic marks bottoms", a10, {}, [{}, {}]),
    ("A11 broad participation", a11, {}, [{"share": 0.5}, {"share": 0.7}]),
    ("A12 retail rotates to altcoins", a12, {}, [{}, {}]),
    ("A13 attention long-short", a13, {}, [{"fade": np.log(0.6)}, {"fade": np.log(0.8)}]),
    ("A14 attention-collapse shorts alone", a14, {}, [{"fade": np.log(0.6)}, {"fade": np.log(0.8)}]),
]


def run(D, Y, W, cost, fn, params):
    sig = fn(D, Y, W, **params)
    g, f, e, n = simulate(sig, D, cost)
    return {s[:4]: fold_stats(g, f, e, n, s, en, 1) for s, en in FOLDS}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    D = load()
    Y = Daily(D)
    W = views(Y)
    print("coins with Wikipedia views: %d (%s)" % (len([c for c in W.columns if c != "MARKET"]),
                                                  ", ".join(c.split("/")[0] for c in W.columns if c != "MARKET")))
    cost = costs(list(D.c.columns))
    results = {}
    for name, fn, params, _ in REGISTRY:
        by = run(D, Y, W, cost, fn, params)
        ok, good, have = stage_a(by)
        results[name] = {"folds": by, "stage_a": ok, "good": good, "have": have}
        row = " ".join("%+6.0f%%" % (by[y]["ret"] * 100) if by[y] else "    --" for y in sorted(by))
        gross = sum(by[y]["gross"] > 0 for y in by if by[y])
        ent = sum(by[y]["entries"] for y in by if by[y])
        print("%-40s %s | gross>0 %d/6 | %4.0f entries | %s %d/%d" % (name, row, gross, ent, "PASS" if ok else "fail",
                                                                    good, have), flush=True)
    json.dump(results, open(os.path.join(OUT, "stage_a.json"), "w"))


def stage_b():
    from research.h50_long_short import blend
    from research.h60_swing_mft import bot_curve
    from research.h61_swing import rotation_curve
    from research.holdout2018 import HOLDOUT
    results = json.load(open(os.path.join(OUT, "stage_a.json")))
    survivors = [r for r in REGISTRY if results[r[0]]["stage_a"]]
    if not survivors:
        print("no survivors")
        return
    D = load()
    Y = Daily(D)
    W = views(Y)
    cost = costs(list(D.c.columns))
    for name, fn, params, neighbours in survivors:
        nb = []
        for p in neighbours:
            ok, good, have = stage_a(run(D, Y, W, cost, fn, p))
            nb.append("%s %d/%d" % ("holds" if ok else "breaks", good, have))
        g, f, _, _ = simulate(fn(D, Y, W, **params), D, cost)
        net = 1 + g - f
        print("\n%s  neighbours: %s" % (name, ", ".join(nb)))
        for label, fs in (("six folds", FOLDS), ("holdout", HOLDOUT)):
            b1 = b2 = 0
            rows = []
            for start, end in fs:
                sl = slice(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=1))
                sleeve = net.loc[sl].cumprod()
                sleeve.index = sleeve.index + pd.Timedelta(hours=1)
                bot, rot = bot_curve((start, end)), rotation_curve((start, end))
                sb = curve_stats(bot)
                s1 = curve_stats(blend(rot, sleeve.reindex(rot.index).ffill().fillna(1.0), 0.30))
                s2 = curve_stats(blend(bot, sleeve.reindex(bot.index).ffill().fillna(1.0), 0.20))
                b1 += s1["comp"] > sb["comp"]
                b2 += s2["comp"] > sb["comp"]
                rows.append("%s bot %+.0f%% B1 %+.0f%% B2 %+.0f%%" % (start[:4], sb["ret"] * 100, s1["ret"] * 100, s2["ret"] * 100))
            print("  %s: B1 better %d/%d, B2 better %d/%d | %s" % (label, b1, len(fs), b2, len(fs), "  ".join(rows)))


if __name__ == "__main__":
    stage_b() if "--stage-b" in sys.argv else main()
