"""H74: long-short swings (the user: "not only long swings, we can do long-short swings").

h61's swing strategies were long only. Written down before running: the short twin of each of
h61's Stage A survivors that has a natural mirror, and two short setups of their own, each a
hypothesis about why prices fall next:

  SS27 breakdown and retest: support becomes resistance; after a close below the 20-day low, a
       rebound to within 2% of that level that fails below it is shorted; stop 1 ATR above it,
       trail 2 ATRs, 10 days
  SS29 volatility contraction near lows: within 10% of the 60-day low with the daily ATR at a
       60-day low, a close below the 20-day low starts the next leg down; stop 1.5 ATRs, trail
       2.5 ATRs, 15 days
  SS32 relative weakness on BTC's up days: on a day BTC rises 3%, the bottom-fifth momentum
       coins that rise less than half as much lack buyers and lead the next fall; 5 days, stop 8%
  SS38 Darvas box breakdowns: a 20-day range within 15% broken downwards on twice the usual
       volume; stop at the top of the range, trail 15%
  SS39 new lows since listing: below the all-time low every holder is under water, so rallies
       meet sellers waiting to break even; 20 days or 20% off the trough
  SS41 greed in a bear market: average funding in its top tenth of the past year while BTC is
       below its 200-day average means longs are crowded into a downtrend; short the majors 7 days
  SS51 the death cross: the 50-day average crossing below the 200-day; out on the golden cross or
       25% off the trough
  SS60 failed breakouts: a close above the 20-day high that falls back below it within 3 days
       leaves trapped buyers who sell; stop 1 ATR above the high, trail 2 ATRs, 10 days
  SS61 bear-market rallies: in a downtrend (50-day average below the 200-day), a rally to the
       falling 20-day average with RSI(14) above 60 is shorted; stop 1.5 ATRs, trail 2 ATRs, 10 days

Stage A as in h61: alone, after the fee and half the spread, net return and composite positive in
at least 5 of the 6 folds. Then the long-short swing book (h61's long survivors and these short
survivors, equally weighted) beside the live bot as h73 judged the long-only one: 10% of the
account from both books, and in the rotation's idle share while BTC's filter is off; against the
model of the live bot, then both holdout years.

    python -m research.h74_long_short_swings
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS, funding_table
from research.h60_swing_mft import MAJORS, costs, simulate, stage_a
from research.h61_swing import OUT as H61, REGISTRY as LONGS, Daily, atr_frac, const, load, momentum_rank, run

OUT = os.path.join("runs", "research", "h74")


def engine_short(D, entry, stop=None, target=None, trail=None, max_hours=None, exit_when=None):
    """Short trades, the mirror of h61's engine: `stop` above the entry, `target` below it and
    `trail` above the lowest close since entry, as fractions read at entry. A -1 position frame."""
    c = D.c.values
    ent = entry.fillna(False).values & D.ok.values
    n, m = c.shape
    held = np.zeros((n, m), dtype=bool)
    st = stop.values if isinstance(stop, pd.DataFrame) else None
    tg = target.values if isinstance(target, pd.DataFrame) else None
    tr = trail.values if isinstance(trail, pd.DataFrame) else None
    ex = exit_when.fillna(False).values if exit_when is not None else None
    for j in range(m):
        inside, e_price, lo, t0, s_d, g_d, t_d = False, 0.0, 0.0, 0, 0.0, 0.0, 0.0
        for t in range(n):
            p = c[t, j]
            if inside:
                if p != p:
                    held[t, j] = True
                    continue
                lo = min(lo, p)
                out = ((st is not None and p > e_price * (1 + s_d))
                       or (tg is not None and p < e_price * (1 - g_d))
                       or (tr is not None and p > lo * (1 + t_d))
                       or (max_hours is not None and t - t0 >= max_hours)
                       or (ex is not None and ex[t, j]))
                if out:
                    inside = False
                else:
                    held[t, j] = True
                    continue
            if ent[t, j] and p == p:
                inside, e_price, lo, t0 = True, p, p, t
                s_d = st[t, j] if st is not None else 0.0
                g_d = tg[t, j] if tg is not None else 0.0
                t_d = tr[t, j] if tr is not None else 0.0
                if (st is not None and s_d != s_d) or (tg is not None and g_d != g_d) or (tr is not None and t_d != t_d):
                    inside = False
                    continue
                held[t, j] = True
    return -pd.DataFrame(held, index=D.idx, columns=D.c.columns).astype(float)


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def ss27(D, Y, near=0.02):
    lvl = Y.l.shift(1).rolling(20).min()
    broke = lvl.where(Y.c < lvl).ffill(limit=10)
    retest = (Y.h >= broke * (1 - near)) & (Y.c < broke) & (Y.c.shift(1) < broke * (1 - near))
    return engine_short(D, Y.event(retest), stop=atr_frac(D, Y, 1.0), trail=atr_frac(D, Y, 2.0), max_hours=240)


def ss29(D, Y, band=0.10):
    a = Y.atr()
    coil = (a <= a.rolling(60).min() * 1.05) & (Y.c <= (1 + band) * Y.c.rolling(60).min())
    sig = coil.astype(float).rolling(5).max().astype(bool) & (Y.c < Y.l.shift(1).rolling(20).min())
    return engine_short(D, Y.event(sig), stop=atr_frac(D, Y, 1.5), trail=atr_frac(D, Y, 2.5), max_hours=360)


def ss32(D, Y, jump=0.03):
    btc = Y.r["BTC/USD"]
    lag = momentum_rank(Y) <= 0.2
    weak = lag & Y.r.lt(0.5 * btc, axis=0) & (btc > np.log(1 + jump)).values[:, None]
    weak["BTC/USD"] = False
    return engine_short(D, Y.event(weak), stop=const(D, 0.08), max_hours=120)


def ss38(D, Y, width=0.15):
    top, bottom = Y.h.shift(1).rolling(20).max(), Y.l.shift(1).rolling(20).min()
    box = (top / bottom - 1) < width
    sig = box & (Y.c < bottom) & (Y.v > 2 * Y.v.rolling(20).median())
    stop = Y.level(((top - Y.c) / Y.c).clip(lower=0.03, upper=0.2))
    return engine_short(D, Y.event(sig), stop=stop, trail=const(D, 0.15), max_hours=480)


def ss39(D, Y, trail=0.20):
    sig = Y.c <= Y.c.expanding(min_periods=200).min()
    return engine_short(D, Y.event(sig), trail=const(D, trail), max_hours=480)


def ss41(D, Y, top=0.9):
    f = pd.DataFrame.from_dict(funding_table(72), orient="index")
    f.index = pd.to_datetime(f.index, unit="ms", utc=True)
    avg = f.sort_index().mean(axis=1).reindex(Y.c.index, method="ffill")
    crowded = avg > avg.rolling(365, min_periods=120).quantile(top)
    bear = Y.c["BTC/USD"] < Y.c["BTC/USD"].rolling(200).mean()
    sig = pd.DataFrame(False, index=Y.c.index, columns=Y.c.columns)
    for p in MAJORS:
        if p in sig.columns:
            sig[p] = crowded & bear
    return engine_short(D, Y.event(sig), max_hours=168)


def ss51(D, Y, trail=0.25):
    m50, m200 = Y.c.rolling(50).mean(), Y.c.rolling(200).mean()
    cross = (m50 < m200) & (m50.shift(1) >= m200.shift(1))
    golden = Y.level(m50 > m200).fillna(0).astype(bool)
    return engine_short(D, Y.event(cross), trail=const(D, trail), exit_when=golden)


def ss60(D, Y, days=3):
    lvl = Y.h.shift(1).rolling(20).max()
    broke = (Y.c > lvl).astype(float).shift(1).rolling(days).max().astype(bool)
    trap = broke & (Y.c < lvl.shift(1))
    return engine_short(D, Y.event(trap), stop=atr_frac(D, Y, 1.0), trail=atr_frac(D, Y, 2.0), max_hours=240)


def ss61(D, Y, level=60):
    m20, m50, m200 = Y.c.rolling(20).mean(), Y.c.rolling(50).mean(), Y.c.rolling(200).mean()
    rally = (m50 < m200) & (m20 < m20.shift(5)) & (Y.h >= m20) & (rsi(Y.c) > level)
    return engine_short(D, Y.event(rally), stop=atr_frac(D, Y, 1.5), trail=atr_frac(D, Y, 2.0), max_hours=240)


SHORTS = [
    ("SS27 breakdown and retest", ss27, {}, [{"near": 0.01}, {"near": 0.03}]),
    ("SS29 volatility contraction near lows", ss29, {}, [{"band": 0.07}, {"band": 0.15}]),
    ("SS32 relative weakness on BTC's up days", ss32, {}, [{"jump": 0.02}, {"jump": 0.04}]),
    ("SS38 Darvas box breakdowns", ss38, {}, [{"width": 0.10}, {"width": 0.20}]),
    ("SS39 new lows since listing", ss39, {}, [{"trail": 0.15}, {"trail": 0.25}]),
    ("SS41 greed in a bear market", ss41, {}, [{"top": 0.8}, {"top": 0.95}]),
    ("SS51 the death cross", ss51, {}, [{"trail": 0.20}, {"trail": 0.30}]),
    ("SS60 failed breakouts", ss60, {}, [{"days": 2}, {"days": 5}]),
    ("SS61 bear-market rallies", ss61, {}, [{"level": 55}, {"level": 65}]),
]


def net_returns(D, Y, cost, registry, names):
    out = {}
    for name, fn, params, _ in registry:
        if name in names:
            g, f, _, _ = simulate(fn(D, Y, **params), D, cost)
            out[name.split(" ")[0]] = g - f
    return pd.DataFrame(out).fillna(0.0)


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "stage_a.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    D = load()
    Y = Daily(D)
    cost = costs(list(D.c.columns))
    print("Stage A, the short setups alone (net return by fold, after the fee and half the spread):")
    for name, fn, params, neighbours in SHORTS:
        if name not in results:
            by = run(D, Y, cost, fn, params)
            ok, good, have = stage_a(by)
            nb = []
            if ok:
                for p in neighbours:
                    nok, ngood, nhave = stage_a(run(D, Y, cost, fn, p))
                    nb.append("%s %d/%d" % ("holds" if nok else "breaks", ngood, nhave))
            results[name] = {"folds": by, "stage_a": ok, "good": good, "have": have, "neighbours": nb}
            json.dump(results, open(path, "w"))
        r = results[name]
        row = " ".join("%+6.0f%%" % (r["folds"][y]["ret"] * 100) if r["folds"][y] else "    --" for y in sorted(r["folds"]))
        print("  %-42s %s | %s %d/%d%s" % (name, row, "PASS" if r["stage_a"] else "fail", r["good"], r["have"],
                                            " | neighbours: " + ", ".join(r["neighbours"]) if r["neighbours"] else ""),
              flush=True)
    longs = json.load(open(os.path.join(H61, "stage_a.json")))
    long_names = [n for n, *_ in LONGS if longs.get(n, {}).get("stage_a")]
    short_names = [n for n, *_ in SHORTS if results[n]["stage_a"] and
                   all(x.startswith("holds") for x in results[n]["neighbours"])]
    print("\nlong survivors (h61): %d; short survivors with both neighbours holding: %d (%s)" % (
        len(long_names), len(short_names), ", ".join(n.split(" ")[0] for n in short_names) or "none"))
    R = pd.concat([net_returns(D, Y, cost, LONGS, long_names), net_returns(D, Y, cost, SHORTS, short_names)], axis=1)
    R.to_pickle(os.path.join(OUT, "returns.pkl"))
    sleeves({"long-only swings (h73)": R[[n.split(" ")[0] for n in long_names]].mean(axis=1),
             "long-short swings": R.mean(axis=1)} | ({"short swings alone": R[[n.split(" ")[0] for n in short_names]].mean(axis=1)}
                                                       if short_names else {}), D)


def sleeves(books, D):
    """Each swing book beside the live bot, as h73 judged the long-only one."""
    from concurrent.futures import ProcessPoolExecutor
    from research.h60_swing_mft import bot_curve, ema
    from research.h61_swing import rotation_curve
    from research.h73_swing_sleeve import book_curve, idle, mix, scores, sleeve_curve
    from research.holdout2018 import HOLDOUT
    folds = list(FOLDS) + list(HOLDOUT)
    with ProcessPoolExecutor(max_workers=8) as pool:
        bots = dict(zip([f[0] for f in folds], pool.map(bot_curve, folds)))
        rots = dict(zip([f[0] for f in folds], pool.map(rotation_curve, folds)))
        bks = dict(zip([f[0] for f in folds], pool.map(book_curve, folds)))
    btc = D.c["BTC/USD"]
    off = ema(btc, 168) < ema(btc, 672)
    off.index = off.index + pd.Timedelta(hours=1)
    res = {}
    for start, end in folds:
        bot, rot, bk = bots[start], rots[start], bks[start]
        idx = bot.index.intersection(rot.index).intersection(bk.index)
        rot, bk = rot.loc[idx], bk.loc[idx]
        curves = {"model of the live bot": mix([rot, bk], [0.7, 0.3])}
        for name, r in books.items():
            sw = sleeve_curve(r, start, end, idx)
            curves["%s: alone" % name] = sw
            curves["%s: 10%% from both books" % name] = mix([rot, bk, sw], [0.63, 0.27, 0.1])
            curves["%s: the rotation's idle share" % name] = mix([idle(rot, sw, off, 1.0, start), bk], [0.7, 0.3])
            curves["%s: half the idle share" % name] = mix([idle(rot, sw, off, 0.5, start), bk], [0.7, 0.3])
        for n, c in curves.items():
            res.setdefault(n, {})[start[:4]] = scores(c, start)
    for label, fs in (("six folds", FOLDS), ("holdout", HOLDOUT)):
        ys = [f[0][:4] for f in fs]
        base = res["model of the live bot"]
        print("\n%s (against the model of the live bot)" % label)
        for n, by in res.items():
            better = sum(by[y][2] > base[y][2] for y in ys)
            w14 = sum(by[y][3] > base[y][3] for y in ys)
            s14 = sum(by[y][4] > base[y][4] for y in ys)
            worst, bworst = max(by[y][1] for y in ys), max(base[y][1] for y in ys)
            ok = better >= (5 if len(ys) == 6 else len(ys)) and worst <= bworst + 0.02
            print("  %-52s %s | DD %.0f%% | year better %d/%d | 14d score %d/%d, Sharpe %d/%d, mean %+.1f%%%s" % (
                n, " ".join("%+6.0f%%" % (by[y][0] * 100) for y in ys), worst * 100, better, len(ys), w14, len(ys),
                s14, len(ys), np.mean([by[y][5] for y in ys]) * 100,
                "  -> passes" if ok and not n.startswith("model") and "alone" not in n else ""))


if __name__ == "__main__":
    main()
