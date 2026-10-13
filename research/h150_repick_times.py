"""H150: how often (and when) to re-pick H149's best long/short design.

H149's best: 70% long the pair with the best 25-day return, 30% short the one with the worst, re-picked
daily. Here on hourly closes of every pair (data/binance), re-picked every 1, 2, 4, 6, 8, 12, 24,
48, 72 or 120 hours (the daily one at 00:00 UTC), and daily at 06:00, 12:00 and 18:00 UTC. The
score is the log return over the last 600 hours; a pair needs 30 days beyond that of history. The
two legs are held at 70% and 30% between re-picks; each change of pair pays 0.1% (0.1% for the short
leg) on both the sale and the purchase plus half each pair's spread. Tests as H144-H149, over October
2023 to October 2026: the mean daily return above zero (7-day block bootstrap) and above 1,000 runs
with random eligible pairs on each leg on the same schedule, both at p < 0.05; the competition from
4 October 2026 and its last three days reported, with the competition's score (0.4 Sortino + 0.3
Sharpe + 0.3 Calmar, bot.metrics) over the competition so far and its median over every 14-day window
of the three years.

    python -m research.h150_repick_times
"""
import csv
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research import h144_bomber_search as h

LONG, SHORT_W, LOOKBACK = 0.7, 0.3, 600
FEE, SHORT_FEE = 0.001, 0.001
_D = {}


def _setup():
    warnings.filterwarnings("ignore")
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["asset_type"] == "crypto" and r["pair"] not in ("PAXG/USD", "MUB/USD")]
    closes, half = {}, {}
    for r in rows:
        path = os.path.join("data", "binance", r["pair"].replace("/USD", "USDT") + "_1h.csv")
        if os.path.exists(path):
            x = pd.read_csv(path).drop_duplicates("ts")
            closes[r["pair"]] = pd.Series(x["close"].values, index=pd.to_datetime(x["ts"], unit="ms", utc=True))
            half[r["pair"]] = max(float(r["spread"]) / 2, 0.00005)
    C = pd.DataFrame(closes).sort_index()
    C = C[C.index >= pd.Timestamp("2023-06-01", tz="UTC")]
    lc = np.log(C)
    score = lc - lc.shift(LOOKBACK)
    eligible = C.notna() & C.shift(LOOKBACK + 720).notna()
    _D.update(C=C, R=(C.shift(-1) / C - 1).values, S=score.where(eligible).values, E=eligible.values,
              half=pd.Series(half).reindex(C.columns).fillna(0.001).values, idx=C.index)


def run(mask, every, offset, rng=None):
    """Hourly returns of the account on the hours in mask."""
    R, S, E, hs, idx = (_D[k] for k in ("R", "S", "E", "half", "idx"))
    pos = np.flatnonzero(mask)
    hours = (idx[pos].asi8 // 3_600_000_000_000).astype(np.int64)
    out = np.zeros(len(pos))
    L = Sh = -1
    for i, t in enumerate(pos):
        cost = 0.0
        if i == 0 or (hours[i] - offset) % every == 0:
            ok = np.isfinite(S[t]) & np.isfinite(R[t]) & E[t]
            cand = np.flatnonzero(ok)
            if len(cand) >= 2:
                if rng is not None:
                    a, b = rng.choice(cand, 2, replace=False)
                else:
                    order = cand[np.argsort(S[t, cand])]
                    a, b = order[-1], order[0]
                if a != L:
                    cost += LONG * ((FEE + hs[L]) if L >= 0 else 0) + LONG * (FEE + hs[a])
                    L = a
                if b != Sh:
                    cost += SHORT_W * ((SHORT_FEE + hs[Sh]) if Sh >= 0 else 0) + SHORT_W * (SHORT_FEE + hs[b])
                    Sh = b
        r = 0.0
        if L >= 0 and np.isfinite(R[t, L]):
            r += LONG * R[t, L]
        if Sh >= 0 and np.isfinite(R[t, Sh]):
            r -= SHORT_W * R[t, Sh]
        out[i] = r - cost
    return pd.Series(out, index=idx[pos])


def _job(job):
    if not _D:
        _setup()
    every, offset = job
    idx = _D["idx"]
    dev = (idx >= pd.Timestamp(h.DEV[0], tz="UTC")) & (idx < pd.Timestamp(h.DEV[1], tz="UTC"))
    comp = idx >= pd.Timestamp("2026-10-04 13:00", tz="UTC")
    r = run(dev, every, offset)
    daily = (1 + r).resample("1D").prod() - 1
    tot = np.log1p(r).sum()
    rng = np.random.default_rng(every * 100 + offset)
    rand = np.array([np.log1p(run(dev, every, offset, rng)).sum() for _ in range(1000)])
    h.RNG = rng
    c = run(comp, every, offset)
    last = c[c.index >= c.index[-1] - pd.Timedelta(days=3)]
    from bot.metrics import summarize, window_stats
    curve = lambda x: [(int(t.value // 10 ** 6), v) for t, v in (100_000 * (1 + x).cumprod()).items()]
    comp_score = summarize(curve(c.iloc[:-1]), 100_000).get("composite", float("nan"))
    w14 = window_stats(curve(r)).get("window_composite_median", float("nan"))
    return (every, offset, np.expm1(tot), h.block_p(daily.values), (np.sum(rand >= tot) + 1) / 1001,
            np.prod(1 + c.values[:-1]) - 1, np.prod(1 + last.values[:-1]) - 1, comp_score, w14)


def main() -> None:
    warnings.filterwarnings("ignore")
    _setup()
    jobs = [(e, 0) for e in (1, 2, 4, 6, 8, 12, 24, 48, 72, 120)] + [(24, o) for o in (6, 12, 18)]
    with ProcessPoolExecutor(max_workers=13) as pool:
        rows = list(pool.map(_job, jobs))
    print("H150: 70%% long the best 25-day pair, 30%% short the worst; hourly data to %s" % _D["idx"][-1])
    print("%-22s | %10s %7s %7s %9s | %8s %8s %9s" % ("re-pick", "3 years", "p mean", "p rand", "14d score",
                                                     "comp", "last 3d", "comp score"))
    for every, offset, tot, pm, pr, cr, lr, cs, w14 in rows:
        label = ("every %dh" % every) + (" at %02d:00 UTC" % offset if every == 24 else "")
        ok = pm < 0.05 and pr < 0.05 and cr >= 0.10
        print("%-22s | %+9.0f%% %7.4f %7.4f %9.2f | %+7.1f%% %+7.1f%% %9.2f %s" % (
            label, 100 * tot, pm, pr, w14, 100 * cr, 100 * lr, cs, "PASS" if ok else ""))


if __name__ == "__main__":
    main()
