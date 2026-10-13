"""H147: a dollar-neutral bomber, long the strongest pairs and short the weakest in equal amounts.

The user asked for a dollar-neutral version beside H146's long-or-short bomber: half the account
long the k pairs with the highest score, half short the k with the lowest (k = 1, 2, 3), so a
market-wide fall moves both sides alike. Scores, holds, costs, tests and pass marks as H146 (p < 0.05
for the mean daily return and against 1,000 runs with random pairs on each side, +10% in the
competition; the last three days reported), fixed before running.

    python -m research.h147_dollar_neutral
"""
import warnings

import numpy as np
import pandas as pd

from research import h144_bomber_search as h
from research.h146_long_short_bomber import SHORT_FEE


def simulate(C, half, score, k, hold, random_rng=None, eligible=None):
    R = (C.shift(-1) / C - 1).values
    S = score.values
    E = eligible.values if eligible is not None else np.isfinite(S)
    hs = half.reindex(C.columns).fillna(0.001).values
    n, m = S.shape
    out = np.zeros(n)
    w = np.zeros(m)
    for t in range(n - 1):
        cost = 0.0
        if t % hold == 0:
            ok = np.isfinite(S[t]) & np.isfinite(R[t]) & E[t]
            new = np.zeros(m)
            idx = np.flatnonzero(ok)
            if len(idx) >= 2 * k:
                if random_rng is not None:
                    pick = random_rng.permutation(idx)
                    longs, shorts = pick[:k], pick[k:2 * k]
                else:
                    order = idx[np.argsort(S[t, idx])]
                    longs, shorts = order[-k:], order[:k]
                new[longs] = 0.5 / k
                new[shorts] = -0.5 / k
            d = new - w
            cost = np.sum(np.where(d > 0, np.where(w < 0, SHORT_FEE, h.FEE), np.where(w > 0, h.FEE, SHORT_FEE))
                          * np.abs(d) + np.abs(d) * hs)
            w = new
        out[t] = np.nansum(w * np.nan_to_num(R[t])) - cost
    return pd.Series(out, index=C.index)


def main() -> None:
    warnings.filterwarnings("ignore")
    C, V, half = h.load()
    lc = np.log(C)
    sdv = lc.diff().rolling(20, min_periods=15).std()
    eligible = C.notna() & C.shift(30).notna()
    dev = (C.index >= pd.Timestamp(h.DEV[0], tz="UTC")) & (C.index < pd.Timestamp(h.DEV[1], tz="UTC"))
    comp = C.index >= pd.Timestamp(h.COMP_START, tz="UTC")
    last3 = C.index >= C.index[-4]
    rows = []
    for kd in (7, 14, 21, 28):
        r = (lc - lc.shift(kd)).where(eligible)
        for sname, score in (("%dd" % kd, r), ("%dd/vol" % kd, r / (sdv * np.sqrt(kd)))):
            for k in (1, 2, 3):
                for hold in (1, 2, 3):
                    r_dev = simulate(C[dev], half, score[dev], k, hold)
                    tot = np.log1p(r_dev).sum()
                    rand = [np.log1p(simulate(C[dev], half, score[dev], k, hold, random_rng=h.RNG,
                                              eligible=eligible[dev])).sum() for _ in range(1000)]
                    p_rand = (np.sum(np.array(rand) >= tot) + 1) / 1001
                    p_mean = h.block_p(r_dev.values)
                    comp_ret = np.prod(1 + simulate(C[comp], half, score[comp], k, hold).values[:-1]) - 1
                    last_ret = np.prod(1 + simulate(C[last3], half, score[last3], k, 1).values[:-1]) - 1
                    ok = p_mean < 0.05 and p_rand < 0.05 and comp_ret >= 0.10
                    rows.append((ok, sname, k, hold, np.expm1(tot), p_mean, p_rand, comp_ret, last_ret))
            print("  done", sname, flush=True)
    rows.sort(key=lambda x: (not x[0], -x[7]))
    print("\nH147: data to %s; %d designs" % (C.index[-1], len(rows)))
    print("%-8s %2s %4s | %10s %7s %7s | %8s %8s" % ("score", "k", "hold", "3 years", "p mean", "p rand", "comp", "last 3d"))
    for ok, sname, k, hold, tot, pm, pr, cr, lr in rows:
        print("%s %-8s %2d %4d | %+9.0f%% %7.4f %7.4f | %+7.1f%% %+7.1f%%" % (
            "PASS" if ok else "    ", sname, k, hold, 100 * tot, pm, pr, 100 * cr, 100 * lr))
    print("\n%d of %d designs pass all three" % (sum(r[0] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
