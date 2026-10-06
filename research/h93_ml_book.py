"""H93: H92's models where their edge is (written after H92, before running these).

H92's models ranked the 49 coins far better than K2 on average (daily rank correlation with the
next 7 days +0.05 to +0.12 in every fold, K2 about -0.02), yet their top 3 made less than K2's
in every fold: the rotation earns from the few coins that run away, and the models learned the
average ordering. An edge across the whole cross-section is what a long-short book uses, and a
model can be trained on the top instead. Walk-forward as H92 (same features, folds and gap):

  B0  the book as live, in this daily simulation: every coin long while its 240h EMA is above
      its 960h, short while below, weights 1 / 30-day volatility, gross 1, re-weighted daily
  B1  an ML long-short book: long the top fifth by LightGBM's score, short the bottom fifth,
      weights 1 / volatility, half the gross each side, re-picked every 7 days
  B2  the same with the ridge model
  B3  the book with an ML veto: B0, but no long below the day's median ML score (LightGBM) and
      no short above it; vetoed coins in cash
  R1  the rotation on a classifier (LightGBM) of whether a coin will be in the top tenth over the
      next 7 days, against K2

The book designs pass if they beat B0's yearly return in at least 5 of 6 folds with a worst
drawdown at most 5 points deeper; R1 if its yearly return beats K2's in 5 of 6. Costs: the fee
and half the spread on each change of weight.

    python -m research.h93_ml_book
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import costs, load_1h
from research.h92_ml_ranking import HORIZON, build, judge_series, k2_score, models, rotation


def walk_forward(X, feats, make, target="y", classify=False):
    days = X.index.get_level_values(0)
    out = pd.Series(np.nan, index=X.index)
    for s, e in FOLDS:
        s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
        tr = (days < s - pd.Timedelta(days=HORIZON)) & X[target].notna()
        te = (days >= s) & (days < e)
        m = make()
        m.fit(X.loc[tr, feats].fillna(0.5), X.loc[tr, target])
        xt = X.loc[te, feats].fillna(0.5)
        out.loc[te] = m.predict_proba(xt)[:, 1] if classify else m.predict(xt)
    return out.unstack()


def book(weights_fn, days, last, cost, every=1):
    lc = np.log(last)
    nxt = (np.exp(lc.shift(-1) - lc) - 1).fillna(0)
    w_prev = pd.Series(0.0, index=last.columns)
    out = []
    for i, d in enumerate(days):
        w = weights_fn(d) if i % every == 0 else w_prev            # held at its weights between re-picks
        w = w.reindex(last.columns).fillna(0.0)
        fee = float(((w - w_prev).abs() * cost.reindex(w.index).fillna(0.002)).sum())
        out.append(float((w * nxt.loc[d]).sum()) - fee)
        w_prev = w
    return pd.Series(out, index=days)


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_1h()
    cost = costs(list(D.c.columns))
    X, ret, last, btc_on = build(D)
    feats = [k for k in X.columns if k not in ("y", "fwd")]
    X["top"] = (X["y"] >= 0.9).astype(float).where(X["y"].notna())
    M = models()
    from lightgbm import LGBMClassifier
    lgb = walk_forward(X, feats, M["M2 LightGBM"])
    ridge = walk_forward(X, feats, M["M1 ridge"])
    top = walk_forward(X, feats, lambda: LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.03,
                                                         min_child_samples=200, subsample=0.8, subsample_freq=1,
                                                         colsample_bytree=0.8, verbose=-1, random_state=0),
                       target="top", classify=True)
    c = D.c
    day = c.index.normalize()
    gap = np.log(c.ewm(span=240, adjust=False).mean() / c.ewm(span=960, adjust=False).mean()).groupby(day).last()
    vol = np.log(c).diff().rolling(720, min_periods=480).std().groupby(day).last()
    alive = last.notna() & vol.notna()

    def trend(d):
        s = np.sign(gap.loc[d]).where(alive.loc[d]) / vol.loc[d]
        s = s.dropna()
        return s / s.abs().sum() if s.abs().sum() > 0 else s

    def ml_ls(score):
        def f(d):
            if d not in score.index:
                return pd.Series(dtype=float)
            s = score.loc[d].where(alive.loc[d]).dropna()
            if len(s) < 10:
                return pd.Series(dtype=float)
            q = s.rank(pct=True)
            iv = 1 / vol.loc[d]
            lo, sh = iv[q[q >= 0.8].index], iv[q[q <= 0.2].index]
            return pd.concat([0.5 * lo / lo.sum(), -0.5 * sh / sh.sum()])
        return f

    def veto(d):
        w = trend(d)
        if d not in lgb.index:
            return w
        s = lgb.loc[d].reindex(w.index)
        med = s.median()
        keep = ((w > 0) & (s >= med)) | ((w < 0) & (s <= med))
        return w.where(keep, 0.0)

    designs = {"B0 the trend book (as live)": (trend, 1), "B1 ML long-short (LightGBM)": (ml_ls(lgb), 7),
               "B2 ML long-short (ridge)": (ml_ls(ridge), 7), "B3 trend book with ML veto": (veto, 1)}
    years = [s[:4] for s, _ in FOLDS]
    print("H93: yearly return / worst drawdown / median 14-day return, by fold %s" % " ".join(years), flush=True)
    res = {}
    for name, (fn, every) in designs.items():
        row = {}
        for s, e in FOLDS:
            idx = [d for d in last.index if pd.Timestamp(s, tz="UTC") <= d < pd.Timestamp(e, tz="UTC")]
            row[s[:4]] = judge_series(book(fn, idx, last, cost, every))
        res[name] = row
        print("  %-30s %s" % (name, " ".join("%+6.0f%%/%2.0f%%/%+5.1f%%" % (v[0] * 100, v[1] * 100, v[2] * 100)
                                             for v in row.values())), flush=True)
    base = res["B0 the trend book (as live)"]
    bdd = max(v[1] for v in base.values())
    for name, row in res.items():
        if name.startswith("B0"):
            continue
        a = sum(row[y][0] > base[y][0] for y in years)
        dd = max(v[1] for v in row.values())
        print("  %-30s return better %d/6, worst DD %.0f%% vs %.0f%% -> %s" % (
            name, a, dd * 100, bdd * 100, "PASS" if a >= 5 and dd <= bdd + 0.05 else "fail"))
    print("\nR1: the rotation on a top-tenth classifier against K2")
    k2 = k2_score(ret)
    for name, sc in (("K2 ranking", k2), ("R1 top-tenth classifier", top)):
        row = []
        for s, e in FOLDS:
            sub = sc.loc[(sc.index >= pd.Timestamp(s, tz="UTC")) & (sc.index < pd.Timestamp(e, tz="UTC"))]
            row.append(judge_series(rotation(sub, ret, last, btc_on, cost)))
        res[name] = row
        print("  %-30s %s" % (name, " ".join("%+6.0f%%/%2.0f%%" % (v[0] * 100, v[1] * 100) for v in row)), flush=True)
    a = sum(x[0] > y[0] for x, y in zip(res["R1 top-tenth classifier"], res["K2 ranking"]))
    print("  R1 return better %d/6 -> %s" % (a, "PASS" if a >= 5 else "fail"))


if __name__ == "__main__":
    main()
