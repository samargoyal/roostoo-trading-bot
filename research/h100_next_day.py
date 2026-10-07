"""H100: can the next day's market direction be predicted? (the user: "this is a day-to-day
competition, we have to play whether the next day is negative or positive").

The competition ranks on return and then on Sortino, Sharpe and Calmar, which count every red
day. Target: the sign of the next UTC day's return of the market the bot trades (an equal-weight
index of its coins, and BTC). Features known at 00:00 UTC: the index's and BTC's returns over 1,
3, 7 and 14 days, their 7-day/28-day EMA gaps, 7-day volatility and its ratio to 30-day, the
share of coins up yesterday and over 3 days, BTC's weight in the moves (BTC minus the index),
the day of the week, the mean perpetual funding rate, and the Fear & Greed index. Walk-forward
as H92 (each fold predicted by models trained only on earlier days).

  N1  logistic regression
  N2  LightGBM (200 trees, 7 leaves)
  N0  the simplest rules: tomorrow like today (momentum), and tomorrow against today (reversal)

Judged on: accuracy and its confidence (a coin flip is 50% +- about 2.6 points on 365 days), and
a daily strategy on the index: long when the model says up, short when it says down, or only
when it is confident (p beyond 0.55), paying 0.1% plus half a typical spread (0.15% in all) on
every change of position, against simply holding the index.

    python -m research.h100_next_day
"""
import os
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS, funding_table
from research.h60_swing_mft import load_1h


def features(D):
    c = D.c
    day = c.index.normalize()
    last = c.groupby(day).last()
    ok = c.notna().groupby(day).sum() >= 20
    last = last.where(ok)
    r = np.log(last).diff()
    idx_r = r.drop(columns=["BTC/USD"], errors="ignore").mean(axis=1)
    btc_r = r["BTC/USD"]
    idx = idx_r.cumsum()
    btc = np.log(last["BTC/USD"])
    X = pd.DataFrame(index=last.index)
    for name, s, lev in (("idx", idx_r, idx), ("btc", btc_r, btc)):
        for n in (1, 3, 7, 14):
            X["%s_r%d" % (name, n)] = s.rolling(n).sum()
        X[name + "_gap"] = lev.ewm(span=7, adjust=False).mean() - lev.ewm(span=28, adjust=False).mean()
        X[name + "_vol7"] = s.rolling(7).std()
        X[name + "_volratio"] = s.rolling(7).std() / s.rolling(30).std()
    X["breadth1"] = (r > 0).astype(float).where(r.notna()).mean(axis=1)
    X["breadth3"] = (r.rolling(3).sum() > 0).astype(float).where(r.notna()).mean(axis=1)
    X["btc_minus_idx"] = btc_r - idx_r
    X["dow"] = X.index.dayofweek
    fund = funding_table(72)
    if fund:
        f = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC").normalize(): np.mean(list(v.values()))
                       for k, v in fund.items() if v})
        f = f[~f.index.duplicated(keep="last")].sort_index()
        X["funding"] = f.reindex(X.index).ffill()
    path = os.path.join("data", "fear_greed.csv")
    if os.path.exists(path):
        fg = pd.read_csv(path)
        s = pd.Series(fg["value"].values, index=pd.to_datetime(fg["ts"], unit="s", utc=True)).sort_index()
        s.index = s.index.normalize()
        X["fear_greed"] = s[~s.index.duplicated()].reindex(X.index).ffill()
    return X, idx_r, btc_r


def run(X, y_ret, name):
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    nxt = y_ret.shift(-1)                                    # tomorrow's return, known after the fact
    y = (nxt > 0).astype(float).where(nxt.notna())
    models = {"N1 logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
              "N2 LightGBM": lambda: LGBMClassifier(n_estimators=200, num_leaves=7, learning_rate=0.03,
                                                     min_child_samples=40, verbose=-1, random_state=0)}
    probs = {m: pd.Series(np.nan, index=X.index) for m in models}
    feats = list(X.columns)
    for s, e in FOLDS:
        s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
        tr = (X.index < s - pd.Timedelta(days=1)) & y.notna() & X.notna().all(axis=1)
        te = (X.index >= s) & (X.index < e) & X.notna().all(axis=1)
        for m, make in models.items():
            model = make().fit(X.loc[tr, feats], y[tr])
            probs[m].loc[te] = model.predict_proba(X.loc[te, feats])[:, 1]
    today = y_ret
    probs["N0 tomorrow like today"] = (today > 0).astype(float)
    probs["N0 tomorrow against today"] = (today <= 0).astype(float)
    cost = 0.0015
    print("\n%s: accuracy by fold | daily strategy (yearly return by fold) | holding it" % name)
    years = [s[:4] for s, _ in FOLDS]
    hold = {}
    for s, e in FOLDS:
        m = (X.index >= pd.Timestamp(s, tz="UTC")) & (X.index < pd.Timestamp(e, tz="UTC")) & nxt.notna()
        hold[s[:4]] = np.exp(nxt[m].sum()) - 1
    print("  %-28s %s" % ("hold the market", " ".join("%+6.0f%%" % (hold[y_] * 100) for y_ in years)))
    for m, p in probs.items():
        for band in (0.0, 0.05):
            accs, rets = [], []
            for s, e in FOLDS:
                sel = (X.index >= pd.Timestamp(s, tz="UTC")) & (X.index < pd.Timestamp(e, tz="UTC")) & p.notna() & nxt.notna()
                pp, rr = p[sel], nxt[sel]
                pos = np.where(pp > 0.5 + band, 1.0, np.where(pp < 0.5 - band, -1.0, 0.0))
                acc = ((pos > 0) == (rr > 0))[pos != 0].mean() if (pos != 0).any() else np.nan
                simple = np.exp(rr.values) - 1
                pnl = pos * simple - cost * np.abs(np.diff(np.concatenate([[0.0], pos])))
                accs.append(acc)
                rets.append(np.prod(1 + pnl) - 1)
            n = int(sum(((X.index >= pd.Timestamp(FOLDS[0][0], tz="UTC")) & p.notna()).astype(int)))
            print("  %-28s band %.2f | accuracy %s (mean %.1f%%) | strategy %s" % (
                m, band, " ".join("%4.1f%%" % (a * 100) for a in accs), np.nanmean(accs) * 100,
                " ".join("%+6.0f%%" % (x * 100) for x in rets)), flush=True)
            if m.startswith("N0"):
                break
    return n


def main() -> None:
    warnings.filterwarnings("ignore")
    D = load_1h()
    X, idx_r, btc_r = features(D)
    X = X.dropna(axis=1, how="all")
    print("features:", ", ".join(X.columns))
    run(X, idx_r, "Altcoin index (equal weight)")
    run(X, btc_r, "BTC")


if __name__ == "__main__":
    main()
