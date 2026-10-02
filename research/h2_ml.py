"""H2: does a machine-learning model rank coins better than simple signals?

Walk-forward: on the first day of every month from October 2024, each model is trained on
all earlier data (minus an embargo as long as the target horizon, so no training target
overlaps the test month) and then scores every hour of that month. Only out-of-sample
scores are kept. They are compared by daily rank IC with the forward return, within the
point-in-time universe.

Inputs are cross-sectionally ranked features plus market-wide context; the target is the
cross-sectional rank of the forward return. Models: ridge regression, gradient-boosted
trees, and two fixed blends that need no fitting.

    python -m research.h2_ml [--horizon 24]
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from bot.config import UniverseConfig
from research.h1_signals import HOLD_OUT_START, daily_ic
from research.panel import HOUR, cached_pairs, features, load_panel, monthly_universe, targets

OUT_DIR = os.path.join("runs", "research")
MARKET = ["btc_mom_24", "btc_mom_168", "btc_trend", "breadth"]
TEST_START = "2024-10-01"


def ranked(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank scaled to [-0.5, 0.5] within the universe at each hour."""
    f = frame.where(mask)
    return f.rank(axis=1, pct=True) - 0.5


def build(horizon: int):
    pairs = cached_pairs()
    panel = load_panel(pairs)
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    feats = features(panel)
    fwd = targets(panel, horizon)["fwd"]
    cross = [n for n in feats if n not in MARKET]
    columns = {n: ranked(feats[n], mask) for n in cross}
    columns.update({n: feats[n].where(mask) for n in MARKET})
    target = ranked(fwd, mask)
    long = pd.DataFrame({n: c.stack() for n, c in columns.items()})
    long["target"] = target.stack()
    long["fwd"] = fwd.where(mask).stack()
    long.index.names = ["time", "pair"]
    return long, list(columns), fwd, mask


def walk_forward(long: pd.DataFrame, cols, horizon: int) -> pd.DataFrame:
    months = pd.date_range(TEST_START, HOLD_OUT_START, freq="MS", tz="UTC")[:-1]
    times = long.index.get_level_values("time")
    preds = []
    for start in months:
        end = start + pd.offsets.MonthBegin(1)
        train = long[(times < start - horizon * HOUR) & (times.hour % 4 == 0)].dropna()
        test = long[(times >= start) & (times < end)].dropna(subset=cols)
        X, y = train[cols].values, train["target"].values
        out = pd.DataFrame(index=test.index)
        ridge = Ridge(alpha=100.0).fit(X, y)
        out["ridge"] = ridge.predict(test[cols].values)
        gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                            min_samples_leaf=500, l2_regularization=1.0,
                                            random_state=0).fit(X, y)
        out["gbm"] = gbm.predict(test[cols].values)
        out["current_score"] = 0.5 * test["mom_72"] + 0.5 * test["mom_168"]
        out["quality_blend"] = (-test["vol_168"] + test["liquidity"] + test["drawdown_168"]
                                + test["mom_336"]) / 4
        out["low_vol"] = -test["vol_168"]
        preds.append(out)
        print("  %s trained on %d rows" % (start.date(), len(train)), flush=True)
    return pd.concat(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    long, cols, fwd, mask = build(args.horizon)
    preds = walk_forward(long, cols, args.horizon)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "predictions_%dh.pkl" % args.horizon), "wb") as f:
        pickle.dump(preds, f)

    periods = {"Y0 Oct24-Sep25": ("2024-10-01", "2025-10-01"),
               "Y1 Oct25-Jun26": ("2025-10-01", HOLD_OUT_START)}
    print("\nOut-of-sample daily rank IC with the next %dh return" % args.horizon)
    print("%-15s" % "model" + "".join("%22s" % p for p in periods))
    for model in preds.columns:
        score = preds[model].unstack("pair").reindex(index=fwd.index, columns=fwd.columns)
        ic = daily_ic(score, fwd, mask)
        cells = []
        for start, end in periods.values():
            x = ic.loc[start:end]
            x = x[x.index < end]
            cells.append("%8.3f (t %5.2f)" % (x.mean(), x.mean() / x.std() * np.sqrt(len(x))))
        print("%-15s" % model + "".join("%22s" % c for c in cells))


if __name__ == "__main__":
    main()
