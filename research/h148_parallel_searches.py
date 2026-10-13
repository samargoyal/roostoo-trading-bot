"""H146 and H147 run in parallel across CPU cores (same designs, tests and pass marks).

    python -m research.h148_parallel_searches
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research import h144_bomber_search as h
from research import h146_long_short_bomber as ls
from research import h147_dollar_neutral as dn

_D = {}


def _setup():
    warnings.filterwarnings("ignore")
    C, V, half = h.load()
    lc = np.log(C)
    sdv = lc.diff().rolling(20, min_periods=15).std()
    eligible = C.notna() & C.shift(30).notna()
    scores = {}
    for k in (7, 14, 21, 28):
        r = (lc - lc.shift(k)).where(eligible)
        scores["%dd" % k] = r
        scores["%dd/vol" % k] = r / (sdv * np.sqrt(k))
    sides = ls.sides(C, eligible, {k: v for k, v in scores.items() if k.endswith("/vol")})
    dev = (C.index >= pd.Timestamp(h.DEV[0], tz="UTC")) & (C.index < pd.Timestamp(h.DEV[1], tz="UTC"))
    comp = C.index >= pd.Timestamp(h.COMP_START, tz="UTC")
    last3 = C.index >= C.index[-4]
    _D.update(C=C, half=half, scores=scores, sides=sides, eligible=eligible, dev=dev, comp=comp, last3=last3)


def _job(job):
    if not _D:
        _setup()
    kind, a, sname, hold = job
    C, half, dev, comp, last3, el = _D["C"], _D["half"], _D["dev"], _D["comp"], _D["last3"], _D["eligible"]
    score = _D["scores"][sname]
    rng = np.random.default_rng(hash(job) % (2 ** 32))
    if kind == "ls":
        side = _D["sides"][a]
        run = lambda m, rr=None, hd=hold: ls.simulate(C[m], half, score[m], side[m], hd, random_rng=rr,
                                                       eligible=el[m] if rr is not None else None)
    else:
        run = lambda m, rr=None, hd=hold: dn.simulate(C[m], half, score[m], a, hd, random_rng=rr,
                                                       eligible=el[m] if rr is not None else None)
    r_dev = run(dev)
    tot = np.log1p(r_dev).sum()
    rand = np.array([np.log1p(run(dev, rng)).sum() for _ in range(1000)])
    p_rand = (np.sum(rand >= tot) + 1) / 1001
    h.RNG = rng
    p_mean = h.block_p(r_dev.values)
    comp_ret = np.prod(1 + run(comp).values[:-1]) - 1
    last_ret = np.prod(1 + run(last3, None, 1).values[:-1]) - 1
    ok = p_mean < 0.05 and p_rand < 0.05 and comp_ret >= 0.10
    return (ok, kind, a, sname, hold, np.expm1(tot), p_mean, p_rand, comp_ret, last_ret)


def main() -> None:
    warnings.filterwarnings("ignore")
    _setup()
    jobs = []
    for rule in _D["sides"]:
        for sname in _D["scores"]:
            if rule.startswith("symmetric") and not rule.endswith("(%s)" % sname):
                continue
            for hold in (1, 2, 3):
                jobs.append(("ls", rule, sname, hold))
    for sname in _D["scores"]:
        for k in (1, 2, 3):
            for hold in (1, 2, 3):
                jobs.append(("dn", k, sname, hold))
    with ProcessPoolExecutor(max_workers=15) as pool:
        rows = list(pool.map(_job, jobs, chunksize=2))
    rows.sort(key=lambda x: (not x[0], -x[8]))
    print("H146/H147: data to %s; %d designs (long-or-short %d, dollar neutral %d)" % (
        _D["C"].index[-1], len(rows), sum(r[1] == "ls" for r in rows), sum(r[1] == "dn" for r in rows)))
    print("%-4s %-34s %-7s %4s | %10s %7s %7s | %8s %8s" % ("", "design", "score", "hold", "3 years", "p mean", "p rand",
                                                            "comp", "last 3d"))
    for ok, kind, a, sname, hold, tot, pm, pr, cr, lr in rows:
        name = ("long or short: %s" % a) if kind == "ls" else ("dollar neutral, %d each side" % a)
        print("%s %-34s %-7s %4d | %+9.0f%% %7.4f %7.4f | %+7.1f%% %+7.1f%%" % (
            "PASS" if ok else "    ", name, sname, hold, 100 * tot, pm, pr, 100 * cr, 100 * lr))
    print("\n%d of %d designs pass all three" % (sum(r[0] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
