"""H135: meta-labeling the frequent dip buys, taking only the trades a model expects to win.

H134's hourly dip buys trade several times a day and were profitable in every historical period,
but win only about half the time. Meta-labeling keeps the signals and learns which of their trades
win: a gradient-boosted classifier, trained on development trades only, rates each candidate at
entry, and only those above a cutoff are taken.

Fixed before running:
  candidates: every hour any of H134's long signals fires (rsi2 < 10 in an uptrend; a 1- or 4-hour
      residual dip of 2 sd in an uptrend; a leader's 2 sd pullback), entered at the next hour's open,
      out at +1.5 daily moves (a limit), a stop 2 daily moves down, or after 12 hours; costs as before
  label: the trade made money after costs
  features at entry: which signals fired; the 1- and 4-hour residual in sd; RSI(2); the pair's hourly
      volatility; its 1-week and 30-day returns and distance from the 50-day average; BTC's 1-, 4-, 24-
      and 168-hour returns and volatility; the hour's volume against the day's median; the hour of
      day and day of week; beta to BTC; the spread; the share of pairs up over 24 hours; the pair's
      14-day return rank
  model: scikit-learn HistGradientBoostingClassifier (200 iterations, depth 4, learning rate 0.05)
  cutoff: from out-of-fold probabilities within development (5 consecutive folds), the lowest
      probability whose trades win 62% or more (aiming above the 60% brief); then the model is fit on
      all of development and the cutoff applied to validation, the 2018-20 holdout and the competition
After selection, one trade per pair at a time. The brief as before: a win rate of 60% or more, wins
at least 1.5 times losses, positive; reported with the number of trades a day.

    python -m research.h135_meta_labels
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from research import h130_high_win_rate as h
from research.h132_frequent_bounces import HOLDOUT
from research.h134_strong_coin_dips import COMP, rsi

TARGET, STOP, HOLD = 1.5, 2.0, 12
WIN_CUT = 0.62
POOLS = ("P1 rsi2 < 10, uptrend", "P2 residual 1h 2 sd, uptrend", "P2 residual 4h 2 sd, uptrend", "P3 leader pullback")


def dataset():
    spreads = h.pairs_and_spreads()
    frames = {p: f for p in spreads if (f := h.load_hourly(p)) is not None and len(f) >= 24 * 260}
    closes = pd.DataFrame({p: f.set_index("ts")["close"] for p, f in frames.items()})
    rank14 = (closes / closes.shift(336) - 1).rank(axis=1, pct=True)
    breadth = (closes / closes.shift(24) > 1).sum(axis=1) / closes.notna().sum(axis=1)
    btc = closes["BTC/USD"]
    lb = np.log(btc).diff()
    btc_f = pd.DataFrame({"btc_1h": lb, "btc_4h": lb.rolling(4).sum(), "btc_24h": lb.rolling(24).sum(),
                          "btc_168h": lb.rolling(168).sum(), "btc_vol": lb.rolling(168).std()})
    rows = []
    h.MAX_H = HOLD
    for pair, hr in frames.items():
        c, v, ts = hr["close"], hr["volume"], hr["ts"]
        lr = np.log(c).diff()
        sd = lr.rolling(168).std()
        unit = (c * sd * np.sqrt(24)).values
        sma = c.rolling(1200).mean()
        week = c / c.shift(168) - 1
        up = (c > sma) & (week > 0)
        r2 = rsi(c, 2)
        b = lb.reindex(ts.values).values
        lrv = lr.values
        cov = (pd.Series(lrv * b).rolling(168).mean()
               - pd.Series(lrv).rolling(168).mean() * pd.Series(b).rolling(168).mean())
        beta = (cov / pd.Series(b).rolling(168).var()).clip(-1, 4).values
        z = {}
        sig = {"P1 rsi2 < 10, uptrend": (up & (r2 < 10)).fillna(False).values}
        for n in (1, 4):
            rc = np.log(c / c.shift(n)).values
            resid = rc - beta * pd.Series(b).rolling(n).sum().values
            rsd = pd.Series(resid).rolling(168).std().shift(1).values
            z[n] = resid / rsd
            sig["P2 residual %dh 2 sd, uptrend" % n] = up.values & (resid <= -2 * rsd)
        lead = (rank14[pair].reindex(ts.values) >= 1 - 10 / rank14.notna().sum(axis=1).reindex(ts.values)).fillna(False).values
        sig["P3 leader pullback"] = lead & (lrv <= -2 * sd.shift(1).values)
        ok = (ts.shift(-1) - ts == 3_600_000).values
        any_sig = np.zeros(len(hr), dtype=bool)
        for s in sig.values():
            any_sig |= np.nan_to_num(s.astype(float)).astype(bool)
        idx = np.flatnonzero(any_sig & ok & np.isfinite(unit) & (unit > 0))
        if not len(idx):
            continue
        cn = dict(h=hr, entry_row=np.arange(len(hr)) + 1.0)
        p0, H, L, C, ets = h.paths(cn, idx)
        u = unit[idx] / p0
        half = max(spreads[pair] / 2, 0.00005)
        tgt, stp = p0 * (1 + TARGET * u), p0 * (1 - STOP * u)
        hit_t, hit_s = H >= tgt[:, None], L <= stp[:, None]
        ft = np.where(hit_t.any(1), hit_t.argmax(1), 10 ** 6)
        fs = np.where(hit_s.any(1), hit_s.argmax(1), 10 ** 6)
        last = HOLD - 1
        sf, tf = (fs <= ft) & (fs <= last), (ft < fs) & (ft <= last)
        end = C[:, last]
        done = np.isfinite(end) | sf | tf
        px = np.where(sf, stp, np.where(tf, tgt, end))
        net = np.where(tf, px, px * (1 - half)) / (p0 * (1 + half)) - 1 - h.TAKER - np.where(tf, h.MAKER, h.TAKER)
        at = np.where(sf, fs, np.where(tf, ft, last))
        bf = btc_f.reindex(ts.values[idx])
        t = pd.to_datetime(ts.values[idx], unit="ms")
        vm = (v / v.rolling(24).median().shift(1)).values
        df = pd.DataFrame({
            "pair": pair, "ts": ets, "net": net, "at": at, "done": done,
            "rsi2": r2.values[idx], "z1": z[1][idx], "z4": z[4][idx], "vol": sd.values[idx],
            "week": week.values[idx], "month": (c / c.shift(720) - 1).values[idx], "dist": (c / sma - 1).values[idx],
            "volume": vm[idx], "hour": t.hour, "dow": t.dayofweek, "beta": beta[idx], "spread": spreads[pair],
            "breadth": breadth.reindex(ts.values[idx]).values, "rank14": rank14[pair].reindex(ts.values[idx]).values,
            **{k: bf[k].values for k in btc_f.columns},
            **{"sig_%d" % i: np.nan_to_num(sig[p].astype(float))[idx] for i, p in enumerate(POOLS)}})
        rows.append(df[df["done"]])
        print("  built", pair, len(df), flush=True)
    return pd.concat(rows, ignore_index=True)


FEATURES = ["rsi2", "z1", "z4", "vol", "week", "month", "dist", "volume", "hour", "dow", "beta", "spread", "breadth",
            "rank14", "btc_1h", "btc_4h", "btc_24h", "btc_168h", "btc_vol"] + ["sig_%d" % i for i in range(len(POOLS))]


def model():
    return HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=0)


def one_per_pair(d):
    """Trades in time order, skipping any that start while the same pair is still in a trade."""
    d = d.sort_values("ts")
    ts, at, pairs = d["ts"].values, d["at"].values, d["pair"].values
    keep, free = np.zeros(len(d), dtype=bool), {}
    for i in range(len(d)):
        if ts[i] >= free.get(pairs[i], -1):
            keep[i] = True
            free[pairs[i]] = ts[i] + (at[i] + 1) * 3_600_000
    return d[keep]


def summary(d, days):
    x = d["net"].values
    if not len(x):
        return "   0 trades"
    w, l = x[x > 0], x[x <= 0]
    ratio = w.mean() / -l.mean() if len(w) and len(l) else float("inf")
    return "%5d trades (%4.1f a day) WR %4.1f%% win %+5.2f%% loss %+5.2f%% ratio %4.2f mean %+5.2f%%" % (
        len(x), len(x) / days, 100 * len(w) / len(x), 100 * (w.mean() if len(w) else 0),
        100 * (l.mean() if len(l) else 0), ratio, 100 * x.mean())


def score(data):
    """Development with out-of-fold probabilities, the cutoff, and the later periods scored by a
    model fit on all of development: {"cut": .., "dev": .., "val": .., "holdout": .., "comp": ..}."""
    data = data.copy()
    data["win"] = (data["net"] > 0).astype(int)

    def period(a, b):
        return data[(data["ts"] >= h.ms(a)) & (data["ts"] < h.ms(b))]
    dev = period(*h.DEV).sort_values("ts").reset_index(drop=True)
    # out-of-fold probabilities in development, 5 consecutive folds (each fit on the other four)
    oof = np.full(len(dev), np.nan)
    for f in np.array_split(np.arange(len(dev)), 5):
        train = np.setdiff1d(np.arange(len(dev)), f)
        m = model().fit(dev.loc[train, FEATURES], dev.loc[train, "win"])
        oof[f] = m.predict_proba(dev.loc[f, FEATURES])[:, 1]
    dev["p"] = oof
    cut = None
    for q in np.arange(0.40, 0.95, 0.01):
        sel = one_per_pair(dev[dev["p"] >= q])
        if len(sel) < 300:
            break
        if (sel["net"] > 0).mean() >= WIN_CUT:
            cut = q
            break
    final = model().fit(dev[FEATURES], dev["win"])
    out = {"cut": cut, "dev": dev}
    for label, (a, b) in (("val", h.VAL), ("holdout", HOLDOUT), ("comp", COMP)):
        d = period(a, b).copy()
        d["p"] = final.predict_proba(d[FEATURES])[:, 1]
        out[label] = d
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    sc = score(dataset())
    cut, dev = sc["cut"], sc["dev"]
    print("\nH135: %d candidate trades in development; out-of-fold cutoff for a %.0f%% win rate: %s"
          % (len(dev), 100 * WIN_CUT, "none reached" if cut is None else "%.2f" % cut))
    print("  development, all candidates (one per pair):   %s" % summary(one_per_pair(dev), 1461))
    for q in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        print("  development, out of fold, p >= %.2f:          %s" % (q, summary(one_per_pair(dev[dev["p"] >= q]), 1461)))
    for label, key, days in (("validation 2024-26", "val", 730), ("holdout 2018-20", "holdout", 731),
                             ("competition", "comp", 7.25)):
        d = sc[key]
        print("  %-20s all: %s" % (label, summary(one_per_pair(d), days)))
        for q in sorted({0.50, 0.55, 0.60, 0.65} | ({round(cut, 2)} if cut else set())):
            print("  %-20s p >= %.2f%s: %s" % (label, q, " (the cutoff)" if cut and abs(q - cut) < 1e-9 else "",
                                               summary(one_per_pair(d[d["p"] >= q]), days)))


if __name__ == "__main__":
    main()
