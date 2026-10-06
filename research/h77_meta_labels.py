"""H77 (E6): meta-labelled sizing of the rotation's picks (RESEARCH_QUEUE.md part E), last
because it is the item most exposed to overfitting (rounds 14 and H2 found no value in machine
learning for ranking).

Every UTC day at 00:00, the 5 coins with the best positive 14-day return are the candidates the
rotation picks from. Each gets features known at that moment, and a label: 1 if its next 72
hours beat the round trip's cost (0.3%), else 0. Features: its funding against its own 30-day
norm, its relative volume (the last 24 hours against the same hours of the 4 weeks before),
permutation entropy (order 4, 168 hours), BTC's volatility ratio (24-hour realised variance over
its 30-day median), BTC's distance from its 28-day EMA, breadth (the share of coins above their
240-hour EMA), the gap between its 14-day return and the next candidate's, and its rank.

A model is trained for each fold on the earlier ones only (2019's candidates for the first),
leaving out the 14 days before the fold (the labels look 3 days ahead), with features
standardised on the training rows: a logistic regression, and gradient-boosted trees of depth 3
as the neighbour. Its probability p sizes the pick, w = w_base x clip(2p, 0.5, 1), so it only
ever shrinks a pick and never changes which are taken. Written to
runs/research/h77/meta_<model>.json for the backtester (research flag "meta").

    python -m research.h77_meta_labels
"""
import json
import math
import os
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS, funding_z_table
from research.fullbars import load as load_full
from research.h60_swing_mft import ALL_1H, Data, ema

OUT = os.path.join("runs", "research", "h77")
FEATS = ["funding_z", "rvol", "entropy", "vol_ratio", "btc_gap", "breadth", "gap", "rank"]


def entropy(window, order=4):
    counts = {}
    for i in range(len(window) - order + 1):
        w = window[i:i + order]
        key = tuple(np.argsort(w, kind="stable"))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return -sum(n / total * math.log(n / total) for n in counts.values()) / math.log(math.factorial(order))


def dataset():
    f = load_full(ALL_1H)
    D = Data({k: v.loc["2019-01-01":"2026-10-08"] for k, v in f.items()}, 1)
    c, qv, ok = D.c, D.qv, D.ok.astype(bool)
    btc = c["BTC/USD"]
    r = np.log(btc).diff()
    rv24 = (r ** 2).rolling(24).sum()
    vol_ratio = rv24 / rv24.rolling(720).median()
    btc_gap = btc / ema(btc, 672) - 1
    breadth = ((c > ema(c, 240)) & ok).sum(axis=1) / ok.sum(axis=1).replace(0, np.nan)
    r336 = c / c.shift(336) - 1
    fwd = c.shift(-72) / c - 1
    v24 = qv.rolling(24).sum()
    rvol = v24 / pd.concat([v24.shift(168 * k) for k in (1, 2, 3, 4)]).groupby(level=0).median().reindex(v24.index)
    fz = funding_z_table()
    rows = []
    for t in D.idx[(D.idx + pd.Timedelta(hours=1)).hour == 0]:
        if t > D.idx[-73]:
            break
        cand = r336.loc[t].where(ok.loc[t])
        cand = cand[cand > 0].sort_values(ascending=False)
        day = int((t + pd.Timedelta(hours=1)).timestamp() * 1000)
        loc = c.index.get_loc(t)
        for k, p in enumerate(cand.index[:5]):
            nxt = cand.iloc[k + 1] if k + 1 < len(cand) else 0.0
            rows.append({"t": t, "day": day, "pair": p, "label": int(fwd.at[t, p] > 0.003),
                         "funding_z": fz.get(day, {}).get("FZ:" + p, 0.0), "rvol": rvol.at[t, p],
                         "entropy": entropy(c[p].values[loc - 167:loc + 1]), "vol_ratio": vol_ratio.at[t],
                         "btc_gap": btc_gap.at[t], "breadth": breadth.at[t], "gap": cand.iloc[k] - nxt, "rank": k})
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS)
    df["fold"] = df["t"].apply(lambda t: next((s[:4] for s, e in FOLDS if pd.Timestamp(s, tz="UTC") <= t
                                               < pd.Timestamp(e, tz="UTC")), "2019" if t.year < 2021 else None))
    return df[df["fold"].notna()]


def main() -> None:
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    os.makedirs(OUT, exist_ok=True)
    df = dataset()
    print("candidate-days: %d, labelled 1 (beat 0.3%% in 72 hours): %.0f%%" % (len(df), df.label.mean() * 100))
    models = {"logistic": lambda: LogisticRegression(max_iter=1000),
              "trees": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                                              min_samples_leaf=50, random_state=0)}
    for name, make in models.items():
        table, aucs = {}, []
        for start, _ in FOLDS:
            y = start[:4]
            embargo = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=14)
            train = df[(df.t < embargo)]
            test = df[df.fold == y]
            mu, sd = train[FEATS].mean(), train[FEATS].std().replace(0, 1)
            m = make().fit((train[FEATS] - mu) / sd, train.label)
            p = m.predict_proba((test[FEATS] - mu) / sd)[:, 1]
            aucs.append("%s %.3f" % (y, roc_auc_score(test.label, p)))
            for (day, pair), prob in zip(zip(test.day, test.pair), p):
                table.setdefault(str(day), {})["META:" + pair] = float(prob)
        json.dump(table, open(os.path.join(OUT, "meta_%s.json" % name), "w"))
        print("  %-9s walk-forward AUC by fold: %s" % (name, "  ".join(aucs)))


if __name__ == "__main__":
    main()
