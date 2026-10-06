"""H94: machine learning for the long-short book alone (the user: ML, deep learning, RL "only for
the book").

The book decides, for each coin on its own, long or short (the 240h/960h EMA trend), and sizes it
by 1 / volatility. H93 tried cross-sectional ML books (long the best-ranked, short the worst);
these instead learn the book's own decisions. Daily at 00:00 UTC, 49 coins, walk-forward exactly
as H92 (each fold predicted by models trained only on earlier days, with a 7-day gap). Written
down before running:

  T1  direction by ML: a LightGBM classifier of whether the coin's next 7-day return is positive,
      from H92's 24 features plus the coin's own raw trend gap, returns and volatility; long when
      p > 0.5, short below, 1 / volatility as now
  T2  the same with a neural network (MLP 64-32, three seeds)
  T3  meta-labelling (Lopez de Prado): keep the EMA direction, and learn whether that position
      will make money over the next 7 days (LightGBM); hold it only when p > 0.5, else cash
  T4  meta-sizing: T3's probability sizes the position, (2p - 1) clipped at 0, instead of
      dropping it, re-scaled to the book's gross
  Q1  a learned controller (one-step fitted Q, a contextual bandit): each day choose the whole
      book's mode, all positions, longs only, shorts only, or flat, by a LightGBM model of each
      mode's next 7-day return from the market's state (BTC's 7- and 30-day returns and
      volatility, the share of coins in uptrends, the cross-section's dispersion, the book's own
      last 30 days); trained on earlier days only

Judged in H93's daily simulation against B0, the book as live: better yearly return in at least 5
of 6 folds, worst drawdown at most 5 points deeper. A pass goes into the full backtester.

    python -m research.h94_ml_book
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import costs, load_1h
from research.h92_ml_ranking import HORIZON, build, judge_series
from research.h93_ml_book import book


def fit_predict(X, feats, target, make, classify=True):
    days = X.index.get_level_values(0)
    out = pd.Series(np.nan, index=X.index)
    for s, e in FOLDS:
        s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
        tr = (days < s - pd.Timedelta(days=HORIZON)) & X[target].notna()
        te = (days >= s) & (days < e)
        ms = make()
        ms = ms if isinstance(ms, list) else [ms]
        xt, xtr = X.loc[te, feats].fillna(0), X.loc[tr, feats].fillna(0)
        ps = []
        for m in ms:
            m.fit(xtr, X.loc[tr, target])
            ps.append(m.predict_proba(xt)[:, 1] if classify else m.predict(xt))
        out.loc[te] = np.mean(ps, axis=0)
    return out.unstack()


def main() -> None:
    warnings.filterwarnings("ignore")
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    D = load_1h()
    cost = costs(list(D.c.columns))
    X, ret, last, btc_on = build(D)
    c = D.c
    day = c.index.normalize()
    gap = np.log(c.ewm(span=240, adjust=False).mean() / c.ewm(span=960, adjust=False).mean()).groupby(day).last()
    vol = np.log(c).diff().rolling(720, min_periods=480).std().groupby(day).last()
    alive = last.notna() & vol.notna()
    raw = {"raw_gap": gap, "raw_ret7": ret[7], "raw_ret30": ret[30], "raw_vol": vol,
           "raw_gap_z": gap / (vol * np.sqrt(24 * 30))}
    for k, v in raw.items():
        X[k] = v.stack(future_stack=True).reindex(X.index).values
    side = np.sign(X["raw_gap"])
    X["up"] = (X["fwd"] > 0).astype(float).where(X["fwd"].notna())
    X["meta"] = ((side * X["fwd"]) > 0).astype(float).where(X["fwd"].notna() & side.ne(0))
    feats = [k for k in X.columns if k not in ("y", "fwd", "up", "meta")]
    lgb = lambda: LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=200,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    mlp = lambda: [make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                                 early_stopping=True, max_iter=200, random_state=s))
                   for s in range(3)]
    p_up = fit_predict(X, feats, "up", lgb)
    print("  T1 trained", flush=True)
    p_up_nn = fit_predict(X, feats, "up", mlp)
    print("  T2 trained", flush=True)
    p_meta = fit_predict(X, feats, "meta", lgb)
    print("  T3 trained", flush=True)

    def base(d):
        s = (np.sign(gap.loc[d]) / vol.loc[d]).where(alive.loc[d]).dropna()
        return s / s.abs().sum() if s.abs().sum() > 0 else s

    def by_prob(p):
        def f(d):
            if d not in p.index:
                return base(d)
            q = p.loc[d].where(alive.loc[d]).dropna()
            s = np.sign(q - 0.5) / vol.loc[d].reindex(q.index)
            return s / s.abs().sum() if s.abs().sum() > 0 else s
        return f

    def meta(keep_only):
        def f(d):
            w = base(d)
            if d not in p_meta.index:
                return w
            q = p_meta.loc[d].reindex(w.index)
            if keep_only:
                return w.where(q > 0.5, 0.0)
            scale = (2 * q - 1).clip(lower=0).fillna(0)
            v = w * scale
            return v / v.abs().sum() * w.abs().sum() if v.abs().sum() > 0 else v
        return f

    # Q1: per-mode daily returns of the book (no costs), the market state, and a walk-forward
    # model of each mode's next 7-day return
    lc = np.log(last)
    nxt = (np.exp(lc.shift(-1) - lc) - 1).fillna(0)
    days = list(last.index)
    modes = {"all": lambda w: w, "longs": lambda w: w.clip(lower=0), "shorts": lambda w: w.clip(upper=0),
             "flat": lambda w: w * 0}
    mode_ret = pd.DataFrame({m: [float((fn(base(d)) * nxt.loc[d].reindex(base(d).index).fillna(0)).sum())
                                 for d in days] for m, fn in modes.items()}, index=days)
    fwd7 = mode_ret[::-1].rolling(HORIZON).sum()[::-1]                 # the next 7 days (mode_ret[d]: d to d+1)
    breadth = (gap > 0).astype(float).where(alive).mean(axis=1)
    disp = ret[7].where(alive).std(axis=1)
    state = pd.DataFrame({"btc7": ret[7]["BTC/USD"], "btc30": ret[30]["BTC/USD"], "btcvol": vol["BTC/USD"],
                          "breadth": breadth, "disp": disp, "book30": mode_ret["all"].shift(1).rolling(30).sum()})   # known at d
    choice = pd.Series("all", index=days)
    for s, e in FOLDS:
        s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
        tr = (state.index < s - pd.Timedelta(days=HORIZON)) & fwd7["all"].notna() & state.notna().all(axis=1)
        te = (state.index >= s) & (state.index < e)
        preds = {}
        for m in modes:
            if m == "flat":
                preds[m] = np.zeros(te.sum())
                continue
            model = LGBMRegressor(n_estimators=200, num_leaves=7, learning_rate=0.03, min_child_samples=50,
                                  verbose=-1, random_state=0)
            model.fit(state[tr], fwd7.loc[tr, m])
            preds[m] = model.predict(state[te].fillna(0))
        choice[te] = pd.DataFrame(preds, index=state.index[te]).idxmax(axis=1)
    print("  Q1 trained; modes chosen: %s" % choice[choice.index >= pd.Timestamp(FOLDS[0][0], tz="UTC")]
          .value_counts().to_dict(), flush=True)

    def controller(d):
        return modes[choice.get(d, "all")](base(d))

    designs = {"B0 the book as live": base, "T1 direction by LightGBM": by_prob(p_up),
               "T2 direction by neural net": by_prob(p_up_nn), "T3 meta-labels, keep if p > 0.5": meta(True),
               "T4 meta-labels, size by 2p - 1": meta(False), "Q1 learned controller": controller}
    years = [s[:4] for s, _ in FOLDS]
    print("\nH94: yearly return / worst drawdown / median 14-day return, by fold %s" % " ".join(years), flush=True)
    res = {}
    for name, fn in designs.items():
        row = {}
        for s, e in FOLDS:
            idx = [d for d in last.index if pd.Timestamp(s, tz="UTC") <= d < pd.Timestamp(e, tz="UTC")]
            row[s[:4]] = judge_series(book(fn, idx, last, cost))
        res[name] = row
        print("  %-34s %s" % (name, " ".join("%+6.0f%%/%2.0f%%/%+5.1f%%" % (v[0] * 100, v[1] * 100, v[2] * 100)
                                             for v in row.values())), flush=True)
    b = res["B0 the book as live"]
    bdd = max(v[1] for v in b.values())
    for name, row in res.items():
        if name.startswith("B0"):
            continue
        a = sum(row[y][0] > b[y][0] for y in years)
        m14 = sum(row[y][2] > b[y][2] for y in years)
        dd = max(v[1] for v in row.values())
        print("  %-34s return better %d/6, 14-day better %d/6, worst DD %.0f%% vs %.0f%% -> %s" % (
            name, a, m14, dd * 100, bdd * 100, "PASS" if a >= 5 and dd <= bdd + 0.05 else "fail"))


if __name__ == "__main__":
    main()
