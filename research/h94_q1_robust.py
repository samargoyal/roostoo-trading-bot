"""H94's Q1 (the learned controller for the book) under neighbouring settings, written down after
seeing Q1 (4/6 on yearly return, 5/6 on the median 14-day return, worst drawdown 40% against 47%)
and before running these: the reward horizon 5 and 10 days instead of 7, trees with 3 and 15
leaves instead of 7, and the mode set without "flat". A robust controller passes most of them.

    python -m research.h94_q1_robust
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h60_swing_mft import costs, load_1h
from research.h92_ml_ranking import build, judge_series
from research.h93_ml_book import book

MODES = {"all": lambda w: w, "longs": lambda w: w.clip(lower=0), "shorts": lambda w: w.clip(upper=0),
         "flat": lambda w: w * 0}


def main() -> None:
    warnings.filterwarnings("ignore")
    from lightgbm import LGBMRegressor
    D = load_1h()
    cost = costs(list(D.c.columns))
    X, ret, last, btc_on = build(D)
    c = D.c
    day = c.index.normalize()
    gap = np.log(c.ewm(span=240, adjust=False).mean() / c.ewm(span=960, adjust=False).mean()).groupby(day).last()
    vol = np.log(c).diff().rolling(720, min_periods=480).std().groupby(day).last()
    alive = last.notna() & vol.notna()
    cache = {}

    def base(d):
        if d not in cache:
            s = (np.sign(gap.loc[d]) / vol.loc[d]).where(alive.loc[d]).dropna()
            cache[d] = s / s.abs().sum() if s.abs().sum() > 0 else s
        return cache[d]

    lc = np.log(last)
    nxt = (np.exp(lc.shift(-1) - lc) - 1).fillna(0)
    days = list(last.index)
    mode_ret = pd.DataFrame({m: [float((fn(base(d)) * nxt.loc[d].reindex(base(d).index).fillna(0)).sum())
                                 for d in days] for m, fn in MODES.items()}, index=days)
    state = pd.DataFrame({"btc7": ret[7]["BTC/USD"], "btc30": ret[30]["BTC/USD"], "btcvol": vol["BTC/USD"],
                          "breadth": (gap > 0).astype(float).where(alive).mean(axis=1),
                          "disp": ret[7].where(alive).std(axis=1),
                          "book30": mode_ret["all"].shift(1).rolling(30).sum()})

    def choose(horizon=7, leaves=7, modes=("all", "longs", "shorts", "flat")):
        fwd = mode_ret[::-1].rolling(horizon).sum()[::-1]
        choice = pd.Series("all", index=days)
        for s, e in FOLDS:
            s, e = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
            tr = (state.index < s - pd.Timedelta(days=horizon)) & fwd["all"].notna() & state.notna().all(axis=1)
            te = (state.index >= s) & (state.index < e)
            preds = {}
            for m in modes:
                if m == "flat":
                    preds[m] = np.zeros(te.sum())
                    continue
                model = LGBMRegressor(n_estimators=200, num_leaves=leaves, learning_rate=0.03, min_child_samples=50,
                                      verbose=-1, random_state=0)
                model.fit(state[tr], fwd.loc[tr, m])
                preds[m] = model.predict(state[te].fillna(0))
            choice[te] = pd.DataFrame(preds, index=state.index[te]).idxmax(axis=1)
        return choice

    designs = {"B0 the book as live": None, "Q1 as run (7 days, 7 leaves)": choose(),
               "horizon 5 days": choose(horizon=5), "horizon 10 days": choose(horizon=10),
               "3 leaves": choose(leaves=3), "15 leaves": choose(leaves=15),
               "no flat mode": choose(modes=("all", "longs", "shorts"))}
    years = [s[:4] for s, _ in FOLDS]
    print("Q1 neighbours: yearly return / worst drawdown / median 14-day return, by fold %s" % " ".join(years))
    res = {}
    for name, choice in designs.items():
        fn = base if choice is None else (lambda ch: (lambda d: MODES[ch.get(d, "all")](base(d))))(choice)
        row = {}
        for s, e in FOLDS:
            idx = [d for d in days if pd.Timestamp(s, tz="UTC") <= d < pd.Timestamp(e, tz="UTC")]
            row[s[:4]] = judge_series(book(fn, idx, last, cost))
        res[name] = row
        print("  %-30s %s" % (name, " ".join("%+6.0f%%/%2.0f%%/%+5.1f%%" % (v[0] * 100, v[1] * 100, v[2] * 100)
                                             for v in row.values())), flush=True)
    b = res["B0 the book as live"]
    bdd = max(v[1] for v in b.values())
    for name, row in res.items():
        if name.startswith("B0"):
            continue
        a = sum(row[y][0] > b[y][0] for y in years)
        m14 = sum(row[y][2] > b[y][2] for y in years)
        print("  %-30s return better %d/6, 14-day better %d/6, worst DD %.0f%% vs %.0f%%" % (
            name, a, m14, max(v[1] for v in row.values()) * 100, bdd * 100))


if __name__ == "__main__":
    main()
