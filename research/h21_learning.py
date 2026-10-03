"""H21: neural networks and deep learning for ranking coins, against ridge, gradient-boosted
trees and the defensive book's low-volatility ranking.

Every day at 00:00 UTC each model scores every coin in the month's universe (the bot's rule:
top 45 crypto by 30-day USD volume, PAXG out) on how it will do over the next 24 hours
relative to the others. github.com/derinteke/crypto-cross-sectional-forecasting found that a
GRU and a Transformer fed 120 hours of hourly bars added nothing over LightGBM; this checks
that on Roostoo's coins, with the bot's own features.

Written down before running:
  Features  24, known at the decision: the H1 set (momentum over 1h to 720h scaled by
            volatility, EMA trend, RSI, volatility level and ratio, relative volume, range
            position, drawdown, liquidity) and the H20 set (taker flow over 24h and 168h,
            trade size, Amihud illiquidity, Corwin-Schultz spread, abnormal volume against
            the hour-of-week volume surface, Kalman trend slope). Each is turned into a
            normal score of its rank across the coins that day.
  Target    the normal score of the next 24 hours' log return's rank across the coins.
  Models    ridge; histogram gradient boosting; an MLP (64-32, early stopping); a GRU that
            reads the last 72 hourly bars of each coin (return, volume surprise, taker flow)
            and joins the 24 features before its output layers.
  Walk-forward  retrained at the start of every quarter on all earlier days (from June 2020),
            leaving a day between training and testing so no 24-hour target overlaps.
  Pass      the H20 rule on the daily IC: the same sign in at least 5 of 6 folds, mean at
            least 0.02, and with low volatility partialled out the same sign in at least 4.
  Portfolio (information) hold the top 8 by score, equal weight, rebalanced daily at 00:00
            with 0.1% per unit of turnover, against the top 8 by low volatility.

    python -m research.h21_learning
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from bot.config import UniverseConfig, load_config
from bot.metrics import summarize
from research import fullbars
from research.folds import FOLDS, candidates
from research.h20_screen import build, in_fold, rank_ic
from research.panel import DEFENSIVE, features, monthly_universe

SCORES = "runs/research/h21_scores.csv.gz"
SEQ = 72
FEE = 0.001
TOP = 8


def gauss_rank(frame: pd.DataFrame) -> pd.DataFrame:
    r = frame.rank(axis=1)
    n = r.notna().sum(axis=1)
    return pd.DataFrame(norm.ppf((r.sub(0.5)).div(n, axis=0).clip(1e-6, 1 - 1e-6)),
                        index=frame.index, columns=frame.columns).where(r.notna())


def channels(full: dict) -> np.ndarray:
    """Hourly inputs for the GRU, hour x pair x 3: return over its 168h volatility, volume
    over its 168h mean (log), taker-buy share minus 0.5. Missing values are 0."""
    logp = np.log(full["close"])
    ret = logp.diff()
    vol = ret.rolling(168, min_periods=48).std()
    qv = full["quote_volume"]
    c1 = (ret / vol).clip(-5, 5)
    c2 = np.log(qv / qv.rolling(168, min_periods=48).mean()).clip(-5, 5)
    c3 = (full["taker_buy_quote"] / qv - 0.5).clip(-0.5, 0.5)
    stack = np.stack([c.values for c in (c1, c2, c3)], axis=-1).astype(np.float32)
    return np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)


def fit_gru(seq_idx, static, y, hourly, epochs=4, seed=0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    class Net(nn.Module):
        def __init__(self, n_static):
            super().__init__()
            self.gru = nn.GRU(3, 32, batch_first=True)
            self.head = nn.Sequential(nn.Linear(32 + n_static, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, seq, stat):
            _, h = self.gru(seq)
            return self.head(torch.cat([h[-1], stat], dim=1)).squeeze(1)

    net = Net(static.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    hourly_t = torch.from_numpy(hourly)
    stat_t = torch.from_numpy(static.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    offsets = torch.arange(-SEQ + 1, 1)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for b in range(0, len(order), 1024):
            idx = order[b:b + 1024]
            rows = torch.from_numpy(seq_idx[idx, 0])[:, None] + offsets[None, :]
            cols = torch.from_numpy(seq_idx[idx, 1])[:, None].expand(-1, SEQ)
            seq = hourly_t[rows, cols]
            loss = ((net(seq, stat_t[idx]) - y_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(seq_idx_p, static_p):
        out = []
        with torch.no_grad():
            st = torch.from_numpy(static_p.astype(np.float32))
            for b in range(0, len(seq_idx_p), 4096):
                rows = torch.from_numpy(seq_idx_p[b:b + 4096, 0])[:, None] + offsets[None, :]
                cols = torch.from_numpy(seq_idx_p[b:b + 4096, 1])[:, None].expand(-1, SEQ)
                out.append(net(hourly_t[rows, cols], st[b:b + 4096]).numpy())
        return np.concatenate(out) if out else np.zeros(0)
    return predict


def top_curve(score: pd.DataFrame, simple: pd.DataFrame, start: str, end: str):
    """Top TOP coins by score at each 00:00, equal weight, held for the next 24 hours."""
    days = score.index[(score.index >= pd.Timestamp(start, tz="UTC")) & (score.index < pd.Timestamp(end, tz="UTC"))]
    equity, held, curve = 100000.0, pd.Series(dtype=float), []
    for d in days:
        row = score.loc[d].dropna()
        picks = row.nlargest(TOP).index
        want = pd.Series(1.0 / len(picks), index=picks) if len(picks) else pd.Series(dtype=float)
        turnover = want.sub(held, fill_value=0.0).abs().sum()
        equity *= 1 - FEE * turnover
        r = simple.loc[d].reindex(want.index).fillna(0.0) if d in simple.index else 0.0
        equity *= 1 + float((want * r).sum())
        held = want
        curve.append((int(d.value // 10 ** 6) + 86_400_000, equity))
    return curve


def main(quarters_to_run: int = 0) -> None:
    """quarters_to_run > 0 trains only that many quarters (a quick check of the code)."""
    warnings.filterwarnings("ignore")
    cfg = load_config()
    pairs = sorted(candidates(cfg))
    full = fullbars.load(pairs)
    pairs = list(full["close"].columns)
    mask = monthly_universe(full, UniverseConfig(), pairs)
    mask[DEFENSIVE] = False
    micro = build(full, mask)
    days = micro["mom_336"].index
    classic = features({k: full[k] for k in ("close", "high", "low", "volume")})
    names = [n for n in classic if not n.startswith("btc_") and n != "breadth"]
    feats = {n: classic[n].loc[days].where(mask.loc[days]) for n in names}
    feats.update({n: micro[n] for n in ("taker_flow_24", "taker_flow_168", "trade_size", "amihud_168",
                                         "cs_spread_168", "abnormal_volume", "kalman_slope")})
    logp = np.log(full["close"])
    fwd = (logp.shift(-24) - logp).loc[days].where(mask.loc[days])
    z = {n: gauss_rank(f) for n, f in feats.items()}
    y = gauss_rank(fwd)

    # Long format: one row per (day, coin) inside the universe with a known target or a test day.
    long = pd.concat({n: f.stack(dropna=False) for n, f in z.items()}, axis=1)
    long["y"] = y.stack(dropna=False)
    long["in"] = mask.loc[days].stack()
    long = long[long["in"]].drop(columns="in")
    long[list(z)] = long[list(z)].fillna(0.0)
    hour_pos = pd.Series(np.arange(len(full["close"].index)), index=full["close"].index)
    col_pos = {p: i for i, p in enumerate(pairs)}
    seq_idx = np.stack([hour_pos.loc[long.index.get_level_values(0)].values,
                        np.array([col_pos[p] for p in long.index.get_level_values(1)])], axis=1)
    hourly = channels(full)
    X = long[list(z)].values
    target = long["y"].values
    day_of = long.index.get_level_values(0)

    quarters = pd.date_range("2020-10-01", "2026-10-01", freq="QS-OCT", tz="UTC")
    if quarters_to_run:
        quarters = quarters[:quarters_to_run + 1]
    preds = {m: np.full(len(long), np.nan) for m in ("ridge", "boosting", "mlp", "gru")}
    for q0, q1 in zip(quarters[:-1], quarters[1:]):
        train = (day_of < q0 - pd.Timedelta(days=1)) & np.isfinite(target)
        test = (day_of >= q0) & (day_of < q1)
        if train.sum() < 1000 or test.sum() == 0:
            continue
        Xtr, ytr = X[train], target[train]
        preds["ridge"][test] = Ridge(alpha=10.0).fit(Xtr, ytr).predict(X[test])
        preds["boosting"][test] = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
            random_state=0).fit(Xtr, ytr).predict(X[test])
        preds["mlp"][test] = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                          max_iter=200, random_state=0).fit(Xtr, ytr).predict(X[test])
        predict = fit_gru(seq_idx[train], Xtr, ytr, hourly)
        preds["gru"][test] = predict(seq_idx[test], X[test])
        print("trained through %s: %d rows, tested %d" % (q0.date(), train.sum(), test.sum()), flush=True)

    scores = {m: pd.Series(p, index=long.index).unstack() for m, p in preds.items()}
    # Saved for research/h23_ml_strategy.py, which puts the rankings into the bot's backtester.
    saved = pd.DataFrame({m: pd.Series(p, index=long.index) for m, p in preds.items()}).dropna(how="all")
    saved.index.names = ["day", "pair"]
    saved.to_csv(SCORES, compression="gzip")
    scores["low_vol_168 (book)"] = feats["vol_168"].mul(-1)
    simple = full["close"].pct_change(24).shift(-24).loc[days]   # next 24h simple return
    rows = []
    for name, sc in scores.items():
        sc = sc.reindex(index=days, columns=pairs)
        ic = rank_ic(sc, fwd)
        partial = rank_ic(sc, fwd, micro["low_vol_168"])
        means = [in_fold(ic, s, e).mean() for s, e in FOLDS]
        pmeans = [in_fold(partial, s, e).mean() for s, e in FOLDS]
        overall = float(np.nanmean(means))
        same = sum(np.sign(x) == np.sign(overall) for x in means)
        psame = sum(np.sign(x) == np.sign(overall) for x in pmeans)
        comps = [summarize(top_curve(sc, simple, s, e), 100000.0)["composite"] for s, e in FOLDS]
        rows.append({"model": name, "IC by fold": " ".join("%+.3f" % x for x in means),
                     "mean IC": round(overall, 3), "same sign": "%d/6" % same,
                     "partial same": "%d/6" % psame,
                     "passes": "YES" if same >= 5 and abs(overall) >= 0.02 and psame >= 4 and "book" not in name else "",
                     "top-8 composite by fold": " ".join("%.2f" % c for c in comps),
                     "top-8 median": round(float(np.median(comps)), 2)})
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 20)
    print("Daily rank IC with the next 24h return, and a daily top-8 portfolio (folds from October 2020)")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
