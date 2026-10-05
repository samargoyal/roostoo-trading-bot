"""H69: can indicators tell when a held trend is over or turning choppy? (step 1: evidence)

The user's idea for securing profits: not a fixed profit target, but leaving a coin when most
signals say its trend has gone or is about to turn choppy. Round 68 showed that cutting the
rotation's winners on fixed rules costs dearly, since its returns come from a few huge winners;
an exit is only worth having if it predicts poor days ahead, not merely a pause.

Written down before running. Every UTC day at 00:00, while BTC's 168h EMA is above its 672h
EMA, the rotation's candidates are the 5 coins with the best positive 336-hour return. For each,
12 warnings (1 = the trend may be fading or choppy), from hourly bars, and the outcome: its
return over the next 72 hours (and 168).

   1 momentum turned       3-day return below zero
   2 trend line broken     close below its 72-hour EMA
   3 noisy path            Kaufman efficiency ratio over 72 hours below 0.25
   4 choppy                Choppiness Index (56 hours) above 61.8
   5 trend strength gone   ADX (56 hours) below 20, or -DI above +DI
   6 weak highs            within 3% of the 7-day high with RSI(14) below 60
   7 volume divergence     up over 3 days on less than 70% of the previous 3 days' volume
   8 sellers in charge     taker-buy share over 24 hours below 50%
   9 overextended          more than 2.5 standard deviations above the 20-day mean
  10 crowded longs         72-hour perpetual funding above 0.05% per print
  11 lagging BTC           3-day return below BTC's
  12 volatility vs trend   24-hour ATR 1.5x its level 3 days before, on a falling day

Then: each warning's forward return when on and off; the vote (how many are on) against the
forward return; and gradient-boosted trees on the continuous indicators predicting a falling
next 72 hours, trained walk-forward (each fold's model only on earlier years). An exit rule goes
to step 2 (the bot's own backtester) only if its warning predicts a lower forward return in at
least 5 of the 6 folds.

Added after the first results, since warned coins still rose on average (so selling them for
cash would lose): the rotation re-picks daily, so a warning is worth acting on only if the next
healthy candidate beats the warned one over the next 24 hours. Each rule (and the trees, with a
threshold from the training years) replaces warned picks; the best went to round 69. Then the
same question for the market (round 70): nine warnings measured on BTC and on all coins, against
the rotation's own next 24 hours, and gates on them with switching costs.

    python -m research.h69_trend_exit
"""
import warnings

import numpy as np
import pandas as pd

from research.folds import FOLDS, funding_table
from research.h60_swing_mft import ALL_1H, Data, atr, ema, rsi
from research.fullbars import load as load_full

warnings.filterwarnings("ignore")
NAMES = ["momentum turned", "trend line broken", "noisy path", "choppy", "trend strength gone", "weak highs",
         "volume divergence", "sellers in charge", "overextended", "crowded longs", "lagging BTC", "volatility vs trend"]


def adx_di(D, n):
    up, down = D.h.diff(), -D.l.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    a = atr(D, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def indicators(D):
    c = D.c
    ret3 = c / c.shift(72) - 1
    path = c.diff().abs().rolling(72).sum()
    er = (c - c.shift(72)).abs() / path.replace(0, np.nan)
    prev = c.shift(1)
    tr = pd.concat([(D.h - D.l), (D.h - prev).abs(), (D.l - prev).abs()]).groupby(level=0).max().reindex(D.idx)
    chop = 100 * np.log10(tr.rolling(56).sum() / (D.h.rolling(56).max() - D.l.rolling(56).min())) / np.log10(56)
    adx, pdi, mdi = adx_di(D, 56)
    r14 = rsi(c, 14)
    near_high = c >= 0.97 * c.rolling(168).max()
    vol3 = D.qv.rolling(72).sum()
    share = (D.taker * c).rolling(24).sum() / D.qv.rolling(24).sum()
    z20 = (c - c.rolling(480).mean()) / c.rolling(480).std()
    table = funding_table(72)
    f = pd.DataFrame.from_dict(table, orient="index")
    f.index = pd.to_datetime(f.index, unit="ms", utc=True)
    fund = f.sort_index().reindex(D.idx, method="ffill").reindex(columns=c.columns)
    rel = ret3.sub(ret3["BTC/USD"], axis=0)
    a24 = atr(D, 24)
    cont = {"ret3": ret3, "ema_gap": c / ema(c, 72) - 1, "er": er, "chop": chop, "adx": adx, "di": pdi - mdi,
            "rsi": r14, "vol_ratio": vol3 / vol3.shift(72), "taker": share, "z20": z20, "funding": fund,
            "rel_btc": rel, "atr_jump": a24 / a24.shift(72), "ret1": c / c.shift(24) - 1}
    flags = {
        "momentum turned": ret3 < 0,
        "trend line broken": c < ema(c, 72),
        "noisy path": er < 0.25,
        "choppy": chop > 61.8,
        "trend strength gone": (adx < 20) | (mdi > pdi),
        "weak highs": near_high & (r14 < 60),
        "volume divergence": (ret3 > 0) & (vol3 < 0.7 * vol3.shift(72)),
        "sellers in charge": share < 0.5,
        "overextended": z20 > 2.5,
        "crowded longs": fund > 0.0005,
        "lagging BTC": rel < 0,
        "volatility vs trend": (cont["atr_jump"] > 1.5) & (cont["ret1"] < 0),
    }
    return cont, flags


def dataset(D):
    cont, flags = indicators(D)
    c = D.c
    btc = c["BTC/USD"]
    on = ema(btc, 168) > ema(btc, 672)
    r336 = c / c.shift(336) - 1
    fwd24, fwd72, fwd168 = c.shift(-24) / c - 1, c.shift(-72) / c - 1, c.shift(-168) / c - 1
    mid = D.idx[(D.idx + pd.Timedelta(hours=1)).hour == 0]            # bars that close at 00:00 UTC
    rows = []
    for t in mid:
        if not on.loc[t]:
            continue
        r = r336.loc[t].where(D.ok.loc[t])
        cands = r[r > 0].sort_values(ascending=False).index[:5]
        for rank, p in enumerate(cands):
            row = {"t": t, "pair": p, "rank": rank, "fwd24": fwd24.at[t, p], "fwd72": fwd72.at[t, p],
                   "fwd168": fwd168.at[t, p]}
            for k, v in cont.items():
                row[k] = v.at[t, p]
            for k, v in flags.items():
                row["F:" + k] = bool(v.at[t, p]) if v.at[t, p] == v.at[t, p] else False
            rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=["fwd72"])
    df["votes"] = df[["F:" + n for n in NAMES]].sum(axis=1)
    df["fold"] = df["t"].apply(lambda t: next((s[:4] for s, e in FOLDS if pd.Timestamp(s, tz="UTC") <= t < pd.Timestamp(e, tz="UTC")),
                                              "2019" if t < pd.Timestamp(FOLDS[0][0], tz="UTC") else None))
    return df


FEATS = ["ret3", "ema_gap", "er", "chop", "adx", "di", "rsi", "vol_ratio", "taker", "z20", "funding", "rel_btc",
         "atr_jump", "ret1", "rank"]


def trees():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
                                          random_state=0)


def picks_return(df, warned, years):
    """Each day's 2 picks once warned candidates are skipped (an empty slot earns 0): the mean
    next-24h return per fold, and the swaps a day against the top 2."""
    out, swaps, days = {}, 0, 0
    for y in years:
        rets = []
        for _, g in df[df.fold == y].groupby("t"):
            g = g.sort_values("rank")
            keep = g[~warned.loc[g.index]].iloc[:2]
            rets.append(keep["fwd24"].sum() / 2)
            swaps += len(set(g["pair"].iloc[:2]) - set(keep["pair"]))
            days += 1
        out[y] = np.mean(rets)
    return out, swaps / days


def replace_check(df, full, years):
    four = ["momentum turned", "weak highs", "volume divergence", "lagging BTC"]
    seven = four[:1] + ["trend line broken", "noisy path", "trend strength gone"] + four[1:]
    F = {n: df["F:" + n].astype(bool) for n in NAMES}
    rules = {
        "live (no rule)": pd.Series(False, index=df.index),
        "weak highs or volume divergence": F["weak highs"] | F["volume divergence"],
        "weak highs only": F["weak highs"],
        "volume divergence only": F["volume divergence"],
        "2+ of the 4 that held 6/6": sum(F[n].astype(int) for n in four) >= 2,
        "4+ of the 7 that held 5/6": sum(F[n].astype(int) for n in seven) >= 4,
        "6+ of all 12": df["votes"] >= 6,
    }
    for share in (0.2, 0.1):                       # the trees, with a threshold from the training years
        warned = pd.Series(False, index=df.index)
        for y in years:
            train = full[(full.fold < y) & full["fwd72"].notna()]
            m = trees().fit(train[FEATS], (train["fwd72"] < 0).astype(int))
            cut = np.quantile(m.predict_proba(train[FEATS])[:, 1], 1 - share)
            test = df[df.fold == y]
            warned.loc[test.index] = m.predict_proba(test[FEATS])[:, 1] >= cut
        rules["trees: top %d%% predicted to fall" % (share * 100)] = warned
    print("\nreplacing warned picks with the next healthy candidate: the 2 picks' next 24 hours, before costs")
    base = None
    for name, warned in rules.items():
        per, swaps = picks_return(df, warned.astype(bool), years)
        base = base or per
        print("  %-36s %s | mean %+.3f%% | better %d/6 | swaps a day %.2f" % (
            name, " ".join("%+.2f%%" % (per[y] * 100) for y in years), np.mean(list(per.values())) * 100,
            sum(per[y] > base[y] for y in years), swaps))


def stats(r):
    eq = np.cumprod(1 + r)
    dd = (1 - eq / np.maximum.accumulate(eq)).max()
    sharpe = r.mean() / (r.std() or 1e-9) * np.sqrt(365)
    sortino = r.mean() / (r[r < 0].std() or 1e-9) * np.sqrt(365)
    calmar = (eq[-1] ** (365 / len(r)) - 1) / max(dd, 1e-9)
    return eq[-1] - 1, sharpe, 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar, dd


def market_check(D, df, flags, years):
    c = D.c
    ok = D.ok.astype(bool)
    n = ok.sum(axis=1).astype(float).replace(0.0, np.nan)
    r336 = (c / c.shift(336) - 1).where(ok)
    disp = r336.std(axis=1)
    b = "BTC/USD"
    market = {
        "BTC choppy (CHOP56 > 61.8)": flags["choppy"][b],
        "BTC noisy path (ER72 < 0.25)": flags["noisy path"][b],
        "BTC trend strength gone": flags["trend strength gone"][b],
        "BTC momentum turned (3d < 0)": flags["momentum turned"][b],
        "BTC below its 72h EMA": flags["trend line broken"][b],
        "BTC weak highs": flags["weak highs"][b],
        "under half above their 240h EMA": ((c > ema(c, 240)) & ok).sum(axis=1).astype(float) / n < 0.5,
        "under 40% rising over 14 days": ((r336 > 0) & ok).sum(axis=1).astype(float) / n < 0.4,
        "low dispersion (Stivers-Sun)": disp < disp.rolling(24 * 180, min_periods=24 * 60).median(),
    }
    days = df.groupby("t")
    rot = pd.DataFrame({"fwd24": days.apply(lambda g: g.sort_values("rank").iloc[:2]["fwd24"].sum() / 2),
                        "fold": days["fold"].first()})
    print("\nmarket warnings against the rotation's next 24 hours (%d invested days, mean %+.2f%%):" % (
        len(rot), rot.fwd24.mean() * 100))
    votes = pd.Series(0, index=rot.index)
    for name, s in market.items():
        on = s.reindex(rot.index).fillna(False).astype(bool)
        votes += on.astype(int)
        lower = sum(rot[(rot.fold == y) & on].fwd24.mean() < rot[(rot.fold == y) & ~on].fwd24.mean() for y in years)
        print("  %-32s on %3.0f%% | on %+.2f%% off %+.2f%% | lower when on %d/6" % (
            name, on.mean() * 100, rot[on].fwd24.mean() * 100, rot[~on].fwd24.mean() * 100, lower))
    for lo, hi in ((0, 1), (2, 3), (4, 5), (6, 9)):
        s = rot[(votes >= lo) & (votes <= hi)]
        print("  %d-%d warnings: %4d days, next 24h %+.2f%%" % (lo, hi, len(s), s.fwd24.mean() * 100))
    dip = market["BTC momentum turned (3d < 0)"].reindex(rot.index).fillna(False).astype(bool)
    gates = {"always in (live)": pd.Series(1.0, index=rot.index)}
    for k in (5, 6, 7):
        gates["cash at %d+ warnings" % k] = (votes < k).astype(float)
    gates["half while BTC's 3 days are negative"] = pd.Series(np.where(dip, 0.5, 1.0), index=rot.index)
    print("  gates, with 0.15% a side to switch:")
    base = None
    for name, g in gates.items():
        out = {}
        for y in years:
            w = g[rot.fold == y].values
            cost = np.abs(np.diff(np.concatenate([[1.0], w]))) * 0.0015
            out[y] = stats(w * rot.fwd24[rot.fold == y].values - cost)
        base = base or out
        print("    %-38s %s | composite better %d/6, Sharpe better %d/6" % (
            name, " ".join("%+5.0f%%" % (out[y][0] * 100) for y in years),
            sum(out[y][2] > base[y][2] for y in years), sum(out[y][1] > base[y][1] for y in years)))


def main() -> None:
    f = load_full(ALL_1H)
    D = Data({k: v.loc["2019-06-01":"2026-10-08"] for k, v in f.items()}, 1)
    full = dataset(D)
    full = full[full["fold"].notna()]
    df = full[full.fold != "2019"]
    years = [s[:4] for s, _ in FOLDS]
    print("candidate-days: %d (top 5 by 336h return while BTC's filter is on); mean next-72h return %+.2f%%" % (
        len(df), df["fwd72"].mean() * 100))
    print("\n%-22s %10s %10s %8s %16s" % ("warning", "on: fwd72", "off: fwd72", "on %", "lower when on"))
    for n in NAMES:
        k = "F:" + n
        lower = sum(df[(df.fold == y) & df[k]]["fwd72"].mean() < df[(df.fold == y) & ~df[k]]["fwd72"].mean() for y in years)
        print("%-22s %+9.2f%% %+9.2f%% %7.0f%% %13d/6" % (n, df[df[k]]["fwd72"].mean() * 100, df[~df[k]]["fwd72"].mean() * 100,
                                                       df[k].mean() * 100, lower))
    print("\nvotes (how many warnings are on) against the next 72 hours:")
    for lo, hi in ((0, 3), (4, 5), (6, 7), (8, 12)):
        s = df[(df.votes >= lo) & (df.votes <= hi)]
        per = " ".join("%s %+5.1f%%" % (y, s[s.fold == y]["fwd72"].mean() * 100) for y in years)
        print("  %2d-%-2d warnings: n=%5d, mean %+5.2f%%, falling %3.0f%% | %s" % (
            lo, hi, len(s), s["fwd72"].mean() * 100, (s["fwd72"] < 0).mean() * 100, per))
    for k in (6, 7, 8):
        lower = sum(df[(df.fold == y) & (df.votes >= k)]["fwd72"].mean() < df[(df.fold == y) & (df.votes < k)]["fwd72"].mean()
                    for y in years)
        print("  exit at %d+ warnings: lower forward return in %d/6 folds" % (k, lower))
    # Gradient-boosted trees, walk-forward by fold.
    from sklearn.metrics import roc_auc_score
    feats = FEATS
    print("\ntrees predicting a falling next 72 hours (trained on earlier folds only):")
    lower_n = 0
    for i, y in enumerate(years):
        train, test = full[full.fold < y], full[full.fold == y]
        if len(train) < 300:
            print("  %s: too little history to train" % y)
            continue
        m = trees().fit(train[feats], (train["fwd72"] < 0).astype(int))
        p = m.predict_proba(test[feats])[:, 1]
        auc = roc_auc_score((test["fwd72"] < 0).astype(int), p)
        hi = p >= np.quantile(p, 0.8)
        lower_n += test[hi]["fwd72"].mean() < test[~hi]["fwd72"].mean()
        print("  %s: AUC %.3f | top fifth by predicted fall: fwd72 %+.2f%% vs others %+.2f%%" % (
            y, auc, test[hi]["fwd72"].mean() * 100, test[~hi]["fwd72"].mean() * 100))
    print("  top fifth lower in %d folds" % lower_n)
    replace_check(df, full, years)
    market_check(D, df, indicators(D)[1], years)


if __name__ == "__main__":
    main()
