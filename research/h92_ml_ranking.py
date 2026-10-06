"""H92: machine learning for the rotation's ranking (the user: "anything you want: stochastic, ML,
RL, DL, neural networks").

The rotation (75% of the account) holds the 3 coins its ranking puts first. K2 ranks on 7-, 14-
and 21-day returns (normal scores weighted 2/2/1). Can a model trained on many more features rank
better? Daily, at 00:00 UTC, for each of the bot's 49 coins (hourly bars, 2019-26):

  features  log returns over 1, 3, 7, 14, 21 and 30 days, and against BTC over 7 and 14; hourly
            volatility over 7 and 30 days and their ratio; the 7-day mean daily range (Parkinson);
            dollar volume, 7 days over 30, and its level; the 7-day taker-buy share (order flow);
            the 14-day Williams %R; the 240h/960h EMA gap (the book's trend); the 14-day return's
            t-statistic; the 30-day variance ratio; the 7-day skewness of hourly returns; the
            best day of the last 30 (the MAX effect); each ranked across coins that day; and
            BTC's 7- and 30-day returns and 30-day volatility, the market state, as they are
  target    the coin's next 7-day log return, ranked across coins that day

Models, walk-forward: for each fold (a year from 1 October), trained on every day before it
less a 7-day gap (so no target overlaps the year), then used, unchanged, for the year:

  M1  ridge regression (a linear baseline)
  M2  LightGBM, gradient-boosted trees (300 trees, 15 leaves, learning rate 0.03)
  M3  a neural network (scikit-learn MLP, 64-32 ReLU units, early stopping), 3 seeds averaged
  M4  the mean of M2 and M3's ranks

Judged two ways on each fold, against K2's ranking and the 14-day return alone (R54b):
(1) the mean daily rank correlation (information coefficient) with the next 7 days' returns;
(2) a rotation run on each ranking, exactly alike: the top 3 with a positive 14-day return,
equal weights, while BTC's 168h EMA is above its 672h EMA, otherwise cash; re-picked daily,
paying the fee and half the spread on what changes. A model passes if its rotation beats K2's in
yearly return and in median 14-day return in at least 5 of the 6 folds. A pass goes into the bot.

    python -m research.h92_ml_ranking
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from research.folds import FOLDS
from research.h60_swing_mft import costs, load_1h

HORIZON = 7


def build(D):
    c, h, l = D.c, D.h, D.l
    day = c.index.normalize()
    last = c.groupby(day).last()
    dh, dl = h.groupby(day).max(), l.groupby(day).min()
    dollar = D.qv.groupby(day).sum(min_count=1)
    taker = (D.taker.groupby(day).sum(min_count=1) / D.vol.groupby(day).sum(min_count=1))
    full = c.notna().groupby(day).sum() >= 20
    last = last.where(full)
    r = np.log(c).diff()
    vol7 = r.rolling(168, min_periods=120).std().groupby(day).last()
    vol30 = r.rolling(720, min_periods=480).std().groupby(day).last()
    skew7 = r.rolling(168, min_periods=120).skew().groupby(day).last()
    ema_gap = np.log(c.ewm(span=240, adjust=False).mean() / c.ewm(span=960, adjust=False).mean()).groupby(day).last()
    s24 = r.rolling(24).sum()
    vr = (s24.rolling(720, min_periods=480).var() / (24 * r.rolling(720, min_periods=480).var())).groupby(day).last()
    hh = h.rolling(336, min_periods=240).max().groupby(day).last()
    ll = l.rolling(336, min_periods=240).min().groupby(day).last()
    lc = np.log(last)
    ret = {n: lc - lc.shift(n) for n in (1, 3, 7, 14, 21, 30)}
    btc = ret[7]["BTC/USD"]
    park = np.log(dh / dl) ** 2 / (4 * np.log(2))
    daily_r = lc.diff()
    F = {
        **{"ret%d" % n: v for n, v in ret.items()},
        "rel7": ret[7].sub(ret[7]["BTC/USD"], axis=0), "rel14": ret[14].sub(ret[14]["BTC/USD"], axis=0),
        "vol7": vol7, "vol30": vol30, "volratio": vol7 / vol30,
        "range7": np.sqrt(park.rolling(7, min_periods=5).mean()),
        "dvol_ratio": dollar.rolling(7).mean() / dollar.rolling(30).mean(), "dvol": np.log(dollar.rolling(30).mean()),
        "taker7": taker.rolling(7).mean(), "wr14": -100 * (hh - last) / (hh - ll),
        "ema_gap": ema_gap, "tstat14": ret[14] / (vol30 * np.sqrt(336)), "vr30": vr, "skew7": skew7,
        "max30": daily_r.rolling(30, min_periods=20).max(),
    }
    ranked = {k: v.where(last.notna()).rank(axis=1, pct=True) for k, v in F.items()}
    market = {"btc7": btc, "btc30": ret[30]["BTC/USD"], "btcvol": vol30["BTC/USD"]}
    fwd = lc.shift(-HORIZON) - lc
    target = fwd.rank(axis=1, pct=True)
    # the BTC filter at each day's last hour
    btc_on = (c["BTC/USD"].ewm(span=168, adjust=False).mean() > c["BTC/USD"].ewm(span=672, adjust=False).mean())
    btc_on = btc_on.groupby(day).last()
    rows = []
    for k, v in ranked.items():
        rows.append(v.stack(future_stack=True).rename(k))
    X = pd.concat(rows, axis=1)
    for k, v in market.items():
        X[k] = v.reindex(X.index.get_level_values(0)).values
    X["y"] = target.stack(future_stack=True).reindex(X.index).values
    X["fwd"] = fwd.stack(future_stack=True).reindex(X.index).values
    X = X[X["ret14"].notna()]
    return X, ret, last, btc_on


def k2_score(ret):
    out = 0
    for n, w in ((7, 2.0), (14, 2.0), (21, 1.0)):
        rk = ret[n].rank(axis=1)
        cnt = ret[n].notna().sum(axis=1)
        out = out + w * pd.DataFrame(norm.ppf(((rk.sub(0.5)).div(cnt, axis=0)).values), index=rk.index, columns=rk.columns)
    return out


def models():
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "M1 ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "M2 LightGBM": lambda: LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=200,
                                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                             random_state=0),
        "M3 neural net": lambda: [make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                                early_stopping=True, max_iter=200,
                                                                                random_state=s)) for s in range(3)],
    }


def rotation(score, ret, last, btc_on, cost, top=3):
    """Daily returns of a top-`top` rotation on `score` (positive 14-day return, BTC filter)."""
    lc = np.log(last)
    nxt = np.exp(lc.shift(-1) - lc) - 1
    w_prev = pd.Series(0.0, index=last.columns)
    out = []
    for d in score.index:
        s = score.loc[d].where(ret[14].loc[d] > 0).dropna()
        w = pd.Series(0.0, index=last.columns)
        if btc_on.get(d, False) and len(s):
            picks = s.sort_values(ascending=False).index[:top]
            w[picks] = 1.0 / top
        turn = (w - w_prev).abs()
        fee = float((turn * cost.reindex(w.index).fillna(0.002)).sum())
        g = float((w * nxt.loc[d].fillna(0)).sum())
        out.append(g - fee)
        w_prev = w
    return pd.Series(out, index=score.index)


def judge_series(r):
    eq = (1 + r).cumprod()
    dd = (1 - eq / eq.cummax()).max()
    win = [(1 + r.iloc[i:i + 14]).prod() - 1 for i in range(0, len(r) - 13, 1)]
    return eq.iloc[-1] - 1, dd, float(np.median(win)) if win else np.nan


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_1h()
    cost = costs(list(D.c.columns))
    X, ret, last, btc_on = build(D)
    feats = [k for k in X.columns if k not in ("y", "fwd")]
    days = X.index.get_level_values(0)
    print("H92 ML ranking: %d coin-days, %d features" % (len(X), len(feats)), flush=True)
    scores = {"K2 ranking": k2_score(ret), "14-day return (R54b)": ret[14]}
    preds = {name: pd.Series(np.nan, index=X.index) for name in models()}
    for s, e in FOLDS:
        s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
        tr = (days < s - pd.Timedelta(days=HORIZON)) & X["y"].notna()
        te = (days >= s) & (days < e)
        Xtr, ytr = X.loc[tr, feats].fillna(0.5), X.loc[tr, "y"]
        Xte = X.loc[te, feats].fillna(0.5)
        for name, make in models().items():
            m = make()
            if isinstance(m, list):
                p = np.mean([mm.fit(Xtr, ytr).predict(Xte) for mm in m], axis=0)
            else:
                p = m.fit(Xtr, ytr).predict(Xte)
            preds[name].loc[te] = p
        print("  trained fold %d on %d rows" % (s.year, tr.sum()), flush=True)
    for name, p in preds.items():
        scores[name] = p.unstack()
    scores["M4 LightGBM + neural net"] = (scores["M2 LightGBM"].rank(axis=1, pct=True)
                                          + scores["M3 neural net"].rank(axis=1, pct=True))
    fwd = X["fwd"].unstack()
    print("\nInformation coefficient (mean daily rank correlation with the next 7 days), by fold:")
    years = [s[:4] for s, _ in FOLDS]
    print("  %-26s %s" % ("", " ".join("%7s" % y for y in years)))
    for name, sc in scores.items():
        cells = []
        for s, e in FOLDS:
            idx = [d for d in sc.index if pd.Timestamp(s, tz="UTC") <= d < pd.Timestamp(e, tz="UTC")]
            ics = []
            for d in idx[::1]:
                a, b = sc.loc[d], fwd.loc[d] if d in fwd.index else None
                if b is None:
                    continue
                m = a.notna() & b.notna()
                if m.sum() >= 10:
                    ics.append(spearmanr(a[m], b[m]).correlation)
            cells.append("%+7.3f" % np.nanmean(ics) if ics else "     --")
        print("  %-26s %s" % (name, " ".join(cells)), flush=True)
    print("\nRotation on each ranking (top 3, BTC filter, costs): yearly return / worst drawdown / median 14-day return")
    res = {}
    for name, sc in scores.items():
        row = {}
        for s, e in FOLDS:
            sub = sc.loc[(sc.index >= pd.Timestamp(s, tz="UTC")) & (sc.index < pd.Timestamp(e, tz="UTC"))]
            row[s[:4]] = judge_series(rotation(sub, ret, last, btc_on, cost))
        res[name] = row
        print("  %-26s %s" % (name, " ".join("%+6.0f%%/%2.0f%%/%+4.1f%%" % (v[0] * 100, v[1] * 100, v[2] * 100)
                                             for v in row.values())), flush=True)
    base = res["K2 ranking"]
    print("\nAgainst K2 (folds better on yearly return and on median 14-day return):")
    for name, row in res.items():
        if name == "K2 ranking":
            continue
        a = sum(row[y][0] > base[y][0] for y in years)
        b = sum(row[y][2] > base[y][2] for y in years)
        print("  %-26s return %d/6, 14-day %d/6 -> %s" % (name, a, b, "PASS" if a >= 5 and b >= 5 else "fail"))


if __name__ == "__main__":
    main()
