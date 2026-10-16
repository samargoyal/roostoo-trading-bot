"""H151: modelling ideas from another forecasting competition, applied to the long/short rotation.

The user shared the experiment log of a sequence-forecasting competition and asked for strategies
built from its ideas. Each is turned into a way of choosing the long and the short pair for the
70/30 design (70% long the top pair, 30% short the bottom one, re-picked daily), fixed before running:

  E1 ensemble (its teacher ensembles and seed averages): rank on the mean of the cross-sectional
     ranks of the 14, 21, 25, 28 and 35-day returns instead of one lookback
  E2 regime specialists (its per-class specialists picked by a detector): four regimes from BTC
     (30-day volatility above or below its one-year median, close above or below its 50-day average);
     in each, the lookback (7, 14, 21, 28 or 35 days) with the best mean daily return in that regime
     over the fit period
  E3 distributional ranking (its distribution head, summarised by E[y|y|]/E[|y|]): each pair-day
     falls in a cell of momentum decile (25-day return across pairs) x volatility tercile (20-day);
     the cell's score is that summary of the next day's returns over the fit period
  E4 E3 within today's regime (its weighting of training rows towards what is scored): the cell
     summaries built only from fit-period days in the same BTC regime as today
  E5 acceleration (its row-to-row change features): the 25-day return's normal score plus that of
     its change over the last 7 days
  E6 chosen on the competition's score (its metric-aware loss): among long shares 50-80%, lookbacks
     14-35 days and 1-2 pairs a side, the design with the best median 14-day composite over the fit
     period (and, for contrast, the one with the best return)
  E7 class-specific lookbacks (its classes): pairs in three volatility classes (90-day volatility
     tercile across pairs), each ranked on the lookback that did best for that class over the fit
     period, as normal scores within the class
The baseline is the 70/30 design on the 25-day return (H149, H150).

Fit period October 2020 to October 2023; every choice above is made there. Judged on October 2023 to
October 2026, untouched: total return and by year, the mean daily return above zero (7-day block
bootstrap) and above 1,000 runs with random pairs on each leg (the noise floor), and the median
14-day competition score; then the competition, 4 October 2026 to the latest price. Daily
simulation as research/h149_neutral_tilts.py (costs, spreads, shorts' fees).

    python -m research.h151_from_experiments
"""
import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research import h144_bomber_search as h
from research.h149_neutral_tilts import simulate

FIT = ("2020-10-01", "2023-10-01")
TEST = ("2023-10-01", "2026-10-01")
COMP = "2026-10-04"
LOOKS = (7, 14, 21, 25, 28, 35)
_D = {}


def load():
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["asset_type"] == "crypto" and r["pair"] not in ("PAXG/USD", "MUB/USD")]
    closes, half = {}, {}
    for r in rows:
        path = os.path.join("data", "binance", r["pair"].replace("/USD", "USDT") + "_1h.csv")
        if not os.path.exists(path):
            continue
        x = pd.read_csv(path).drop_duplicates("ts").sort_values("ts")
        c = pd.Series(x["close"].values, index=pd.to_datetime(x["ts"], unit="ms", utc=True))
        n = c.resample("1D").count()
        closes[r["pair"]] = c.resample("1D").last()[(n >= 24) | (n.index == n.index[-1])]
        half[r["pair"]] = max(float(r["spread"]) / 2, 0.00005)
    C = pd.DataFrame(closes).sort_index()
    C = C[C.index >= pd.Timestamp("2020-01-01", tz="UTC")]
    return C, pd.Series(half)


def mask(idx, a, b):
    return (idx >= pd.Timestamp(a, tz="UTC")) & (idx < pd.Timestamp(b, tz="UTC"))


def zrow(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def regimes(C):
    btc = C["BTC/USD"]
    vol = np.log(btc).diff().rolling(30).std()
    high = vol > vol.rolling(365, min_periods=180).median()
    up = btc > btc.rolling(50).mean()
    return (high.astype(int) * 2 + up.astype(int)).where(vol.notna())


def setup():
    warnings.filterwarnings("ignore")
    C, half = load()
    lc = np.log(C)
    eligible = C.notna() & C.shift(36).notna()
    mom = {L: (lc - lc.shift(L)).where(eligible) for L in LOOKS}
    sd20 = lc.diff().rolling(20, min_periods=15).std().where(eligible)
    fit = mask(C.index, *FIT)
    nxt = (C.shift(-1) / C - 1)
    reg = regimes(C)
    scores = {"baseline 25-day": mom[25]}
    # E1 ensemble of lookback ranks
    scores["E1 ensemble of lookbacks"] = sum(m.rank(axis=1, pct=True) for L, m in mom.items() if L >= 14) / 5
    # E2 regime specialists: per regime, the lookback with the best fit-period mean daily return
    best = {}
    for g in range(4):
        days = fit & (reg == g).values
        perf = {}
        for L in (7, 14, 21, 28, 35):
            r = simulate(C[fit], half, mom[L][fit], 1, 1, 0.7, None, None, None)
            perf[L] = r[days[fit]].mean()
        best[g] = max(perf, key=perf.get)
    e2 = mom[25].copy() * np.nan
    for g, L in best.items():
        rows = (reg == g).values
        e2[rows] = mom[L][rows]
    scores["E2 regime specialists %s" % "/".join(str(best[g]) for g in range(4))] = e2
    # E3 / E4 distributional cells: momentum decile x volatility tercile
    mdec = np.floor((mom[25].rank(axis=1, pct=True) * 10).clip(upper=9.999))
    vter = np.floor((sd20.rank(axis=1, pct=True) * 3).clip(upper=2.999))
    cell = (mdec * 3 + vter)

    def summary(y):
        y = y[np.isfinite(y)]
        return (np.sum(y * np.abs(y)) / np.sum(np.abs(y))) if len(y) > 50 else np.nan

    def table(days):
        c, y = cell[days].values.ravel(), nxt[days].values.ravel()
        ok = np.isfinite(c) & np.isfinite(y)
        return {int(k): summary(y[ok & (c == k)]) for k in range(30)}

    t3 = table(fit)
    tie = mom[25].rank(axis=1, pct=True) * 1e-6
    scores["E3 distributional cells"] = cell.apply(lambda col: col.map(t3)) + tie
    t4 = {g: table(fit & (reg == g).values) for g in range(4)}
    e4 = cell.copy() * np.nan
    for g in range(4):
        rows = (reg == g).values
        e4[rows] = cell[rows].apply(lambda col: col.map(t4[g]))
    scores["E4 cells within today's regime"] = e4 + tie
    # E5 acceleration
    scores["E5 momentum + acceleration"] = zrow(mom[25]) + zrow(mom[25] - mom[25].shift(7))
    # E7 class-specific lookbacks: volatility class by 90-day volatility tercile
    sd90 = lc.diff().rolling(90, min_periods=60).std().where(eligible)
    vcls = np.floor((sd90.rank(axis=1, pct=True) * 3).clip(upper=2.999))
    best7 = {}
    for k in range(3):
        perf = {}
        for L in (7, 14, 21, 28, 35):
            s = mom[L].where(vcls == k)
            perf[L] = simulate(C[fit], half, s[fit], 1, 1, 0.7, None, None, None).mean()
        best7[k] = max(perf, key=perf.get)
    e7 = mom[25] * np.nan
    for k, L in best7.items():
        part = mom[L].where(vcls == k)
        e7 = e7.fillna(part.sub(part.mean(axis=1), axis=0).div(part.std(axis=1), axis=0))
    scores["E7 class lookbacks %s" % "/".join(str(best7[k]) for k in range(3))] = e7
    _D.update(C=C, half=half, eligible=eligible, scores=scores, mom=mom, fit=fit)


def composite14(daily):
    from bot.metrics import window_stats
    eq = 100_000 * (1 + daily).cumprod()
    curve = [(int(t.value // 10 ** 6), v) for t, v in eq.items()]
    return window_stats(curve).get("window_composite_median", float("nan"))


def evaluate(job):
    if not _D:
        setup()
    name, k, long_share, score_key = job
    C, half, el = _D["C"], _D["half"], _D["eligible"]
    score = _D["scores"][score_key] if score_key in _D["scores"] else _D["mom"][int(score_key)]
    test, comp = mask(C.index, *TEST), C.index >= pd.Timestamp(COMP, tz="UTC")
    r = simulate(C[test], half, score[test], k, 1, long_share, None, None, None)
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    rand = np.array([np.log1p(simulate(C[test], half, score[test], k, 1, long_share, rng, el[test], None)).sum()
                     for _ in range(1000)])
    h.RNG = rng
    tot = np.log1p(r).sum()
    years = [np.expm1(np.log1p(r[mask(r.index, a, b)]).sum()) for a, b in (("2023-10-01", "2024-10-01"),
                                                                             ("2024-10-01", "2025-10-01"),
                                                                             ("2025-10-01", "2026-10-01"))]
    rc = simulate(C[comp], half, score[comp], k, 1, long_share, None, None, None)
    return (name, np.expm1(tot), years, h.block_p(r.values), (np.sum(rand >= tot) + 1) / 1001, composite14(r),
            np.prod(1 + rc.values[:-1]) - 1)


def choose_e6():
    """E6: on the fit period, the design with the best median 14-day composite (and the best return)."""
    C, half, fit = _D["C"], _D["half"], _D["fit"]
    rows = []
    for ls in (0.5, 0.6, 0.7, 0.8):
        for L in (14, 21, 25, 28, 35):
            for k in (1, 2):
                r = simulate(C[fit], half, _D["mom"][L][fit], k, 1, ls, None, None, None)
                rows.append((composite14(r), np.log1p(r).sum(), ls, L, k))
    by_score = max(rows, key=lambda x: x[0])
    by_return = max(rows, key=lambda x: x[1])
    return by_score, by_return


def main() -> None:
    warnings.filterwarnings("ignore")
    setup()
    s6, r6 = choose_e6()
    jobs = [(name, 1, 0.7, name) for name in _D["scores"]]
    jobs.append(("E6 chosen on the score: long %d%%, %dd, %d a side" % (100 * s6[2], s6[3], s6[4]), s6[4], s6[2], str(s6[3])))
    jobs.append(("E6' chosen on return: long %d%%, %dd, %d a side" % (100 * r6[2], r6[3], r6[4]), r6[4], r6[2], str(r6[3])))
    with ProcessPoolExecutor(max_workers=min(15, len(jobs))) as pool:
        rows = list(pool.map(evaluate, jobs))
    print("H151: fit %s to %s, judged %s to %s; competition from %s to %s" % (FIT + TEST + (COMP, _D["C"].index[-1].date())))
    print("%-44s | %9s %-22s %7s %7s %7s | %8s" % ("design", "test 3y", "by year", "p mean", "p rand", "14d", "comp"))
    for name, tot, yrs, pm, pr, c14, cr in rows:
        print("%-44s | %+8.0f%% %-22s %7.4f %7.4f %7.2f | %+7.1f%%" % (
            name, 100 * tot, "/".join("%+.0f%%" % (100 * y) for y in yrs), pm, pr, c14, 100 * cr))


if __name__ == "__main__":
    main()
