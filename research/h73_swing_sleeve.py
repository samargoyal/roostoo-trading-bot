"""H73: a swing sleeve beside the live bot (the user: "experiment on this more").

h61's nine swing strategies that passed Stage A (each positive alone in at least 5 of 6 years),
equally weighted, as 10% of the account taken from both books alike, cut the worst drawdown from
52% to 48% and the mean 14-day return from +5.2% to +4.9% (h72). Written down before running,
the variants, each against the live bot (R54b):

  E1  10%, taken from both books alike (the h72 test, for reference)
  E2  10% taken from the long-short book: 70% rotation, 20% book, 10% swing
  E3  10% taken from the rotation: 60% rotation, 30% book, 10% swing
  E4  idle capital: while BTC's 7-day EMA is below its 28-day EMA the rotation sits in PAXG or
      cash; there its 70% runs the swing sleeve instead (0.15% a side at each switch)
  E4h the same with half the idle share (35%), the rest in PAXG or cash as now
  E5  E1 with the nine weighted by inverse volatility (their last 30 days) instead of equally

The bot is modelled as its sleeves (the rotation alone, the long-short book alone, the swing
sleeve), rebalanced daily to their shares; that model of the live bot is the baseline the
variants are judged against, and its own results are shown beside the bot's. Judged as before:
the yearly composite better in at least 5 of the 6 folds with the worst drawdown at most 2
points deeper, then both holdout years; with the median 14-day score and Sharpe ratio and the
mean 14-day return, the competition's horizon.

    python -m research.h73_swing_sleeve
"""
import copy
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from bot.metrics import calmar, composite, max_drawdown, sharpe, sortino
from research.folds import FOLDS
from research.holdout2018 import HOLDOUT

OUT = os.path.join("runs", "research", "h73")
BOOK = {"strategy": {"rotation_weight": 0.0, "book_mode": "long_short", "ls_trend": [240, 960],
                     "short_stop_atr": 10.0, "short_exclude_external": True}}
COST = 0.0015


def book_curve(fold):
    """The long-short book alone (R54b's, with the funding filter), hourly."""
    from bot.backtest import run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import HOUR_MS, BinanceClient, load_history
    from research.folds import candidates, funding_table, ms
    start, end = fold
    path = os.path.join(OUT, "book_%s.csv" % start)
    if not os.path.exists(path):
        cfg = load_config()
        apply_overrides(cfg, copy.deepcopy(BOOK))
        s, e = ms(start), ms(end)
        slip = candidates(cfg)
        client = BinanceClient()
        bars = {p: load_history(client, p, s - cfg.backtest.warmup_bars * HOUR_MS, e, cfg.backtest.data_dir) for p in slip}
        r = run_backtest(cfg, {p: b for p, b in bars.items() if b}, s, e, cfg.backtest.taker_fee,
                         cfg.backtest.taker_slippage, "taker", monthly_universe=True, slippage_by_pair=slip,
                         external_scores=funding_table(72))
        pd.DataFrame(r.curve, columns=["ts", "equity"]).to_csv(path, index=False)
    df = pd.read_csv(path)
    return pd.Series(df["equity"].values / 100000.0, index=pd.to_datetime(df["ts"], unit="ms", utc=True))


def mix(curves, shares):
    """Sleeves rebalanced to `shares` at each UTC day's close; curves indexed by bar close."""
    idx = curves[0].index
    for c in curves[1:]:
        idx = idx.intersection(c.index)
    day = (idx - pd.Timedelta(hours=1)).normalize()
    growth = np.zeros(len(idx))
    for c, w in zip(curves, shares):
        c = c.loc[idx]
        ref = c.groupby(day).last().shift(1).fillna(c.iloc[0]).reindex(day).values
        growth += w * c.values / ref
    g = pd.Series(growth, index=idx)
    start = g.groupby(day).last().cumprod().shift(1).fillna(1.0).reindex(day).values
    return pd.Series(start * growth, index=idx)


def scores(curve, start):
    """(yearly return, worst drawdown, yearly composite, median 14-day composite, median 14-day
    Sharpe, mean 14-day return) of one fold's curve."""
    vals = curve.values
    days = curve.resample("1D").last().dropna().values
    d = list(days[1:] / days[:-1] - 1)
    total, mdd = vals[-1] / vals[0] - 1, max_drawdown(list(vals))
    year = composite(sortino(d), sharpe(d), calmar(total, len(vals) / 24, mdd))
    ts = (curve.index.astype("int64") // 10 ** 6).values
    first, day = int(pd.Timestamp(start, tz="UTC").value // 10 ** 6), 24 * 3600 * 1000
    comps, sharpes, rets = [], [], []
    while first + 14 * day <= ts[-1]:
        at = np.maximum(np.searchsorted(ts, [first + i * day for i in range(15)], side="right") - 1, 0)
        daily = vals[at]
        w = list(daily[1:] / daily[:-1] - 1)
        t14, m14 = daily[-1] / daily[0] - 1, max_drawdown(list(vals[at[0]:at[-1] + 1] / daily[0]))
        comps.append(composite(sortino(w), sharpe(w), calmar(t14, 14, m14)))
        sharpes.append(sharpe(w))
        rets.append(t14)
        first += 14 * day
    return total, mdd, year, float(np.median(comps)), float(np.median(sharpes)), float(np.mean(rets))


def swing_returns(D, weighting="equal"):
    """Hourly net returns of the sleeve: h61's Stage A survivors, equally or inverse-volatility weighted."""
    from research.h60_swing_mft import costs, simulate
    from research.h61_swing import OUT as H61, REGISTRY, Daily
    results = json.load(open(os.path.join(H61, "stage_a.json")))
    Y = Daily(D)
    cost = costs(list(D.c.columns))
    nets = {}
    for name, fn, params, _ in REGISTRY:
        if results.get(name, {}).get("stage_a"):
            g, f, _, _ = simulate(fn(D, Y, **params), D, cost)
            nets[name.split(" ")[0]] = g - f
    R = pd.DataFrame(nets).fillna(0.0)
    if weighting == "equal":
        return R.mean(axis=1)
    vol = R.rolling(720, min_periods=168).std().shift(1)
    w = (1.0 / vol.replace(0.0, np.nan))
    w = w.div(w.sum(axis=1), axis=0).fillna(1.0 / R.shape[1])
    return (w * R).sum(axis=1)


def sleeve_curve(r, start, end, index):
    sl = slice(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=1))
    c = (1 + r.loc[sl]).cumprod()
    c.index = c.index + pd.Timedelta(hours=1)                 # returns are indexed by bar open
    return c.reindex(index).ffill().fillna(1.0)


def idle(rot, swing, off, share, start):
    """The rotation's curve with `share` of it running the swing sleeve while BTC's filter is off."""
    r_rot = rot.pct_change().fillna(0.0)
    r_sw = swing.pct_change().fillna(0.0)
    o = off.reindex(rot.index).ffill().fillna(False).astype(bool)
    r = np.where(o, share * r_sw + (1 - share) * r_rot, r_rot)
    switch = o.astype(int).diff().abs().fillna(0.0).values
    r = r - switch * share * 2 * COST                          # sell one, buy the other
    return pd.Series(np.cumprod(1 + r), index=rot.index)


def main() -> None:
    warnings.filterwarnings("ignore")
    from research.h60_swing_mft import bot_curve, ema
    from research.h61_swing import load, rotation_curve
    os.makedirs(OUT, exist_ok=True)
    folds = list(FOLDS) + list(HOLDOUT)
    with ProcessPoolExecutor(max_workers=8) as pool:
        bots = dict(zip([f[0] for f in folds], pool.map(bot_curve, folds)))
        rots = dict(zip([f[0] for f in folds], pool.map(rotation_curve, folds)))
        books = dict(zip([f[0] for f in folds], pool.map(book_curve, folds)))
    D = load()
    eq_r, iv_r = swing_returns(D, "equal"), swing_returns(D, "inverse_vol")
    btc = D.c["BTC/USD"]
    off = ema(btc, 168) < ema(btc, 672)
    off.index = off.index + pd.Timedelta(hours=1)
    designs = ["live bot (R54b)", "model: 70% rotation + 30% book", "E1 10% from both", "E2 10% from the book",
               "E3 10% from the rotation", "E4 idle rotation share", "E4h half the idle share", "E5 E1, inverse vol",
               "E1 at 5%", "E1 at 20%", "E4 idle, at a quarter"]
    res = {n: {} for n in designs}
    for start, end in folds:
        bot, rot, book = bots[start], rots[start], books[start]
        idx = bot.index.intersection(rot.index).intersection(book.index)
        bot, rot, book = bot.loc[idx], rot.loc[idx], book.loc[idx]
        sw, swi = sleeve_curve(eq_r, start, end, idx), sleeve_curve(iv_r, start, end, idx)
        curves = {
            "live bot (R54b)": bot,
            "model: 70% rotation + 30% book": mix([rot, book], [0.7, 0.3]),
            "E1 10% from both": mix([rot, book, sw], [0.63, 0.27, 0.1]),
            "E2 10% from the book": mix([rot, book, sw], [0.7, 0.2, 0.1]),
            "E3 10% from the rotation": mix([rot, book, sw], [0.6, 0.3, 0.1]),
            "E4 idle rotation share": mix([idle(rot, sw, off, 1.0, start), book], [0.7, 0.3]),
            "E4h half the idle share": mix([idle(rot, sw, off, 0.5, start), book], [0.7, 0.3]),
            "E5 E1, inverse vol": mix([rot, book, swi], [0.63, 0.27, 0.1]),
            "E1 at 5%": mix([rot, book, sw], [0.665, 0.285, 0.05]),
            "E1 at 20%": mix([rot, book, sw], [0.56, 0.24, 0.2]),
            "E4 idle, at a quarter": mix([idle(rot, sw, off, 0.25, start), book], [0.7, 0.3]),
        }
        for n, c in curves.items():
            res[n][start[:4]] = scores(c, start)
    for label, fs in (("six folds", FOLDS), ("holdout", HOLDOUT)):
        ys = [f[0][:4] for f in fs]
        base = res["model: 70% rotation + 30% book"]
        print("\n%s (judged against the model of the live bot)" % label)
        for n in designs:
            by = res[n]
            better = sum(by[y][2] > base[y][2] for y in ys)
            w14 = sum(by[y][3] > base[y][3] for y in ys)
            s14 = sum(by[y][4] > base[y][4] for y in ys)
            worst, bworst = max(by[y][1] for y in ys), max(base[y][1] for y in ys)
            ok = better >= (5 if len(ys) == 6 else len(ys)) and worst <= bworst + 0.02
            print("  %-32s %s | DD %.0f%% | year better %d/%d | 14d score %d/%d, Sharpe %d/%d, mean %+.1f%%%s" % (
                n, " ".join("%+6.0f%%" % (by[y][0] * 100) for y in ys), worst * 100, better, len(ys), w14, len(ys),
                s14, len(ys), np.mean([by[y][5] for y in ys]) * 100,
                "" if n.startswith(("live", "model")) else ("  -> passes" if ok else "")))


if __name__ == "__main__":
    main()
