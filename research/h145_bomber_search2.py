"""H145: a second round around H144's survivors (21- and 28-day momentum, raw or over volatility).

Same simulation, tests and pass marks as research/h144_bomber_search.py (p < 0.05 for the mean
daily return and against 1,000 random-pick runs over October 2023 to October 2026, and +10% or
more in the competition to the latest price), fixed before running:
  rankings: return over 35, 42, 56 and 90 days, raw and over volatility; 28-day return over
      volatility only for pairs within 10% of their 30-day high; only when its 7-day volume is above
      its 30-day average; only while BTC is above its 50-day average (cash otherwise); and the mean
      of the 21- and 28-day normal scores
  holds of 1, 2, 3, 5 and 7 days; the top 1, 2 or 3 pairs; picks from any pair or only those up
  over the last 14 days.
The designs tried in both rounds are counted in the summary.

    python -m research.h145_bomber_search2
"""
import warnings

import numpy as np
import pandas as pd

from research import h144_bomber_search as h

HOLDS, TOPS = (1, 2, 3, 5, 7), (1, 2, 3)


def rankings(C, V):
    lr = np.log(C).diff()
    sd = lr.rolling(20, min_periods=15).std()
    out = {}
    for k in (35, 42, 56, 90):
        r = np.log(C / C.shift(k))
        out["return %dd" % k] = r
        out["return %dd / vol" % k] = r / (sd * np.sqrt(k))
    r28 = np.log(C / C.shift(28)) / (sd * np.sqrt(28))
    out["28d/vol, within 10% of 30d high"] = r28.where(C >= 0.9 * C.rolling(30).max())
    out["28d/vol, volume rising"] = r28.where(V.rolling(7).mean() > V.rolling(30).mean())
    btc = C["BTC/USD"]
    btc_up = btc > btc.rolling(50).mean()
    out["28d/vol, BTC above 50-day"] = r28.where(pd.DataFrame({c: btc_up for c in C.columns}))
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    out["21d+28d mix"] = z(np.log(C / C.shift(21))) + z(np.log(C / C.shift(28)))
    eligible = C.notna() & C.shift(90).notna()
    return {k: v.where(eligible) for k, v in out.items()}, eligible


def main() -> None:
    warnings.filterwarnings("ignore")
    C, V, half = h.load()
    ranks, eligible = rankings(C, V)
    dev = (C.index >= pd.Timestamp(h.DEV[0], tz="UTC")) & (C.index < pd.Timestamp(h.DEV[1], tz="UTC"))
    comp = C.index >= pd.Timestamp(h.COMP_START, tz="UTC")
    up14 = (C / C.shift(14) > 1).fillna(False)
    random_total = {}
    for gate in (False, True):
        for hold in HOLDS:
            for top in TOPS:
                totals = []
                for _ in range(1000):
                    r = h.simulate(C[dev], half, ranks["return 35d"][dev], hold, top, random_rng=h.RNG,
                                   eligible=eligible[dev], rising=up14[dev] if gate else None)
                    totals.append(np.log1p(r).sum())
                random_total[(gate, hold, top)] = np.array(totals)
    rows = []
    for name, score in ranks.items():
        for gate in (False, True):
            for hold in HOLDS:
                for top in TOPS:
                    r_dev = h.simulate(C[dev], half, score[dev], hold, top, rising=up14[dev] if gate else None)
                    r_comp = h.simulate(C[comp], half, score[comp], hold, top, rising=up14[comp] if gate else None)
                    tot = np.log1p(r_dev).sum()
                    p_mean = h.block_p(r_dev.values)
                    p_rand = (np.sum(random_total[(gate, hold, top)] >= tot) + 1) / 1001
                    comp_ret = np.prod(1 + r_comp.values[:-1]) - 1
                    ok = p_mean < 0.05 and p_rand < 0.05 and comp_ret >= 0.10
                    rows.append((ok, name + (", rising 14d" if gate else ""), hold, top, np.expm1(tot), p_mean, p_rand,
                                 comp_ret))
    rows.sort(key=lambda x: (not x[0], -x[7]))
    print("H145: data to %s; %d designs (H144 tried 324)" % (C.index[-1].date(), len(rows)))
    print("%-40s %4s %3s | %10s %8s %8s | %s" % ("ranking", "hold", "top", "3 years", "p mean", "p random", "competition"))
    for ok, name, hold, top, tot, pm, pr, cr in rows[:60]:
        print("%s %-38s %4d %3d | %+9.0f%% %8.4f %8.4f | %+6.1f%%" % ("PASS" if ok else "    ", name, hold, top, 100 * tot, pm, pr, 100 * cr))
    print("\n%d of %d designs pass all three" % (sum(r[0] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
