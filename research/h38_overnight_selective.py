"""H38: hold a share overnight only when its predicted overnight return beats the cost.

H37 held the top fifth by one signal every night. The user's refinement: put money only in the
shares predicted to earn more overnight than the round trip costs, and otherwise stay out.
Written down before running (daily data 2016-2026, Yahoo):

  Features  (known at the close) mean overnight return over the last 5, 21 and 63 nights; mean
            session return over 5 and 21 days; today's session return and opening gap; volume
            over its 21-day mean; day of the week; the market's (equal-weight) mean overnight
            return over 21 nights
  Target    the next overnight return (close to next open)
  Models    ridge regression and gradient-boosted trees, retrained each January on all earlier
            years, predicting 2018 to 2026 (walk-forward)
  Policy    hold, equally weighted, every share whose predicted return exceeds the round trip
            (0.1% with limit orders, 0.2% at market); cash otherwise
  Pass      after costs, a higher yearly return than buying and holding the same shares in most
            years
Run on 2011's 40 largest US companies (no hindsight) and on the tokenized shares.

    python -m research.h38_overnight_selective
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from research.folds import candidate_table
from research.h37_overnight import BIG_2011


def panel(tickers):
    df = yf.download(tickers, start="2015-06-01", progress=False, auto_adjust=True, group_by="ticker", threads=True)
    names = [t for t in tickers if t in df.columns.get_level_values(0)]
    get = lambda f: pd.DataFrame({t: df[t][f] for t in names})
    return get("Open"), get("Close"), get("Volume")


def features(o, c, v):
    night_past = o / c.shift(1) - 1                  # last night's return, known at today's open
    session = c / o - 1
    f = {"night5": night_past.rolling(5).mean(), "night21": night_past.rolling(21).mean(),
         "night63": night_past.rolling(63).mean(), "sess5": session.rolling(5).mean(),
         "sess21": session.rolling(21).mean(), "session": session, "gap": night_past,
         "relvol": v / v.rolling(21).mean(),
         "mkt_night21": pd.DataFrame(np.repeat(night_past.mean(axis=1).rolling(21).mean().values[:, None],
                                               c.shape[1], axis=1), index=c.index, columns=c.columns)}
    for d in range(5):
        f["dow%d" % d] = pd.DataFrame((c.index.dayofweek == d).astype(float)[:, None].repeat(c.shape[1], axis=1),
                                      index=c.index, columns=c.columns)
    target = o.shift(-1) / c - 1
    long = pd.concat({k: x.stack() for k, x in f.items()}, axis=1)
    long["y"] = target.stack()
    return long.dropna()


def run(name, tickers):
    warnings.filterwarnings("ignore")
    o, c, v = panel(tickers)
    data = features(o, c, v)
    X, y = data.drop(columns="y"), data["y"]
    years = sorted(set(X.index.get_level_values(0).year))
    hold = (c / c.shift(1) - 1).mean(axis=1)
    print("\n%s (%d shares)" % (name, c.shape[1]))
    print("  %-8s %-6s %12s %12s %12s %16s" % ("model", "cost", "share-nights", "bp/trade", "net/yr", "years beat hold"))
    for model_name, make in (("ridge", lambda: Ridge(alpha=10.0)),
                             ("trees", lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                                              max_leaf_nodes=15, min_samples_leaf=200,
                                                                              random_state=0))):
        preds = pd.Series(np.nan, index=X.index)
        for yr in years:
            if yr < 2018:
                continue
            train = X.index.get_level_values(0).year < yr
            test = X.index.get_level_values(0).year == yr
            if train.sum() < 2000 or test.sum() == 0:
                continue
            m = make().fit(X[train].values, y[train].values)
            preds[test] = m.predict(X[test].values)
        for cost in (0.001, 0.002):
            chosen = preds > cost
            realised = y[chosen]
            nightly = (realised - cost).groupby(level=0).mean()             # equal weight across chosen
            days = pd.Index(sorted(set(X.index.get_level_values(0)[X.index.get_level_values(0).year >= 2018])))
            daily = nightly.reindex(days).fillna(0.0)                       # cash when nothing is chosen
            span = len(days) / 252
            net = (1 + daily).prod() ** (1 / span) - 1 if span > 0 else 0.0
            beat = 0
            for yr in range(2018, 2026):
                d = daily[daily.index.year == yr]
                h = hold[hold.index.year == yr]
                beat += (1 + d).prod() > (1 + h).prod()
            print("  %-8s %-6s %12d %12.1f %+11.1f%% %13d/8" % (
                model_name, "%.1f%%" % (cost * 100), int(chosen.sum()),
                realised.mean() * 1e4 if len(realised) else float("nan"), net * 100, beat))
    h = hold[hold.index.year >= 2018]
    print("  hold the same shares: %+.1f%% a year" % (((1 + h).prod() ** (252 / len(h)) - 1) * 100))


def main() -> None:
    tok = [r["pair"].split("/")[0][:-1] for r in candidate_table() if r["asset_type"] == "stock"]
    run("2011's 40 largest US companies (no hindsight)", BIG_2011)
    run("the tokenized shares (chosen with hindsight)", tok)


if __name__ == "__main__":
    main()
