"""H149: around H147's near miss (dollar neutral, 21-day momentum, two pairs a side, held two days).

Fixed before running: long the top k pairs and short the bottom k (k = 1, 2, 3) on the return over
14, 18, 21, 25, 28 or 35 days, held 1, 2, 3 or 5 days, with the long side taking 50% (neutral), 60%
or 70% of the account and the short side the rest; and half the account in H144's C2 (the single
pair with the best 28-day return, re-picked with the book) beside half in the neutral pair book.
Costs, the random-pick test (random pairs on each side), the mean's bootstrap and the pass marks as
H146-H148 (p < 0.05 for both, +10% in the competition); the last three days reported.

    python -m research.h149_neutral_tilts
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research import h144_bomber_search as h
from research.h146_long_short_bomber import SHORT_FEE

_D = {}


def _setup():
    warnings.filterwarnings("ignore")
    C, V, half = h.load()
    lc = np.log(C)
    eligible = C.notna() & C.shift(36).notna()
    scores = {d: (lc - lc.shift(d)).where(eligible) for d in (14, 18, 21, 25, 28, 35)}
    dev = (C.index >= pd.Timestamp(h.DEV[0], tz="UTC")) & (C.index < pd.Timestamp(h.DEV[1], tz="UTC"))
    comp = C.index >= pd.Timestamp(h.COMP_START, tz="UTC")
    _D.update(C=C, half=half, scores=scores, eligible=eligible, dev=dev, comp=comp, last3=C.index >= C.index[-4],
              c2=(lc - lc.shift(28)).where(eligible))


def simulate(C, half, score, k, hold, long_share, rng=None, eligible=None, bomber=None):
    R = (C.shift(-1) / C - 1).values
    S = score.values
    E = eligible.values if eligible is not None else np.isfinite(S)
    B = bomber.values if bomber is not None else None
    hs = half.reindex(C.columns).fillna(0.001).values
    n, m = S.shape
    out, w = np.zeros(n), np.zeros(m)
    for t in range(n - 1):
        cost = 0.0
        new = w
        if t % hold == 0:
            ok = np.isfinite(S[t]) & np.isfinite(R[t]) & E[t]
            idx = np.flatnonzero(ok)
            new = np.zeros(m)
            if len(idx) >= 2 * k:
                if rng is not None:
                    pick = rng.permutation(idx)
                    longs, shorts = pick[:k], pick[k:2 * k]
                else:
                    order = idx[np.argsort(S[t, idx])]
                    longs, shorts = order[-k:], order[:k]
                book = 0.5 if B is not None else 1.0
                new[longs] += book * long_share / k
                new[shorts] -= book * (1 - long_share) / k
            if B is not None:                      # C2's half re-picked with the book
                row = np.where(np.isfinite(B[t]) & np.isfinite(R[t]) & E[t], B[t], -np.inf)
                j = int(np.argmax(row)) if np.isfinite(row).any() and row.max() > 0 else -1
                if j >= 0:
                    new[j] += 0.5
        if new is not w:
            d = new - w
            cost = np.sum(np.where(d > 0, np.where(w < 0, SHORT_FEE, h.FEE), np.where(w > 0, h.FEE, SHORT_FEE))
                          * np.abs(d) + np.abs(d) * hs)
            w = new
        out[t] = np.nansum(w * np.nan_to_num(R[t])) - cost
        grown = w * (1 + np.nan_to_num(R[t]))
        w = grown
    return pd.Series(out, index=C.index)


def _job(job):
    if not _D:
        _setup()
    d, k, hold, ls, with_c2 = job
    C, half, el = _D["C"], _D["half"], _D["eligible"]
    score, bomber = _D["scores"][d], (_D["c2"] if with_c2 else None)
    rng = np.random.default_rng(hash(job) % (2 ** 32))
    run = lambda msk, r=None, hd=hold: simulate(C[msk], half, score[msk], k, hd, ls, r, el[msk] if r is not None else None,
                                               bomber[msk] if bomber is not None else None)
    r_dev = run(_D["dev"])
    tot = np.log1p(r_dev).sum()
    rand = np.array([np.log1p(run(_D["dev"], rng)).sum() for _ in range(1000)])
    h.RNG = rng
    p_mean, p_rand = h.block_p(r_dev.values), (np.sum(rand >= tot) + 1) / 1001
    comp_ret = np.prod(1 + run(_D["comp"]).values[:-1]) - 1
    last_ret = np.prod(1 + run(_D["last3"], None, 1).values[:-1]) - 1
    return (p_mean < 0.05 and p_rand < 0.05 and comp_ret >= 0.10, d, k, hold, ls, with_c2, np.expm1(tot), p_mean,
            p_rand, comp_ret, last_ret)


def main() -> None:
    warnings.filterwarnings("ignore")
    _setup()
    jobs = [(d, k, hold, ls, False) for d in (14, 18, 21, 25, 28, 35) for k in (1, 2, 3) for hold in (1, 2, 3, 5)
            for ls in (0.5, 0.6, 0.7)]
    jobs += [(d, k, hold, 0.5, True) for d in (14, 21, 28) for k in (1, 2) for hold in (1, 2, 3)]
    with ProcessPoolExecutor(max_workers=15) as pool:
        rows = list(pool.map(_job, jobs, chunksize=2))
    rows.sort(key=lambda x: (not x[0], -x[9]))
    print("H149: data to %s; %d designs" % (_D["C"].index[-1], len(rows)))
    print("%-4s %-4s %2s %4s %5s %-6s | %10s %7s %7s | %8s %8s" % ("", "days", "k", "hold", "long", "+C2", "3 years",
                                                                 "p mean", "p rand", "comp", "last 3d"))
    for ok, d, k, hold, ls, c2, tot, pm, pr, cr, lr in rows:
        print("%s %4d %2d %4d %4.0f%% %-6s | %+9.0f%% %7.4f %7.4f | %+7.1f%% %+7.1f%%" % (
            "PASS" if ok else "    ", d, k, hold, 100 * ls, "yes" if c2 else "", 100 * tot, pm, pr, 100 * cr, 100 * lr))
    print("\n%d of %d designs pass all three" % (sum(r[0] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
