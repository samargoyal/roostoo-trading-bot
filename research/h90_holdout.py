"""H90's survivors on the untouched holdout years, 2018-19 and 2019-20, with the same hourly
bars, costs and three holds (12, 24, 48 hours).

    python -m research.h90_holdout
"""
import warnings

import numpy as np
import pandas as pd

from research.fullbars import load as load_full
from research.h60_swing_mft import ALL_1H, Data, costs
from research.h86_scalping import engine
from research.h90_order_flow import signals
from research.holdout2018 import HOLDOUT

SURVIVORS = [("OF05 exhaustion", "OF09 initiative", 1), ("OF09 initiative", "OF16 trapped traders", -1),
             ("OF09 initiative", "OF17 structure shift", 1), ("OF05 exhaustion", None, 1)]


def main() -> None:
    warnings.filterwarnings("ignore")
    f = load_full(ALL_1H)
    D = Data({k: v.loc["2018-01-01":"2020-10-01"] for k, v in f.items()}, 1)
    cost = costs(list(D.c.columns))
    S, stop = signals(D)
    starts = [pd.Timestamp(s, tz="UTC").value for s, _ in HOLDOUT] + [pd.Timestamp(HOLDOUT[-1][1], tz="UTC").value]
    nan = np.full(len(D.idx), np.nan)
    print("H90 survivors on the holdout: net bp per trade (trades) for 2018-19, 2019-20, by hold")
    for a, b, side in SURVIVORS:
        ea = (S[a] == side)
        if b is None:
            e = ea.astype(int) * side
        else:
            eb = S[b] == side
            fa = ea.astype(int).rolling(6, min_periods=1).max() > 0
            fb = eb.astype(int).rolling(6, min_periods=1).max() > 0
            e = (fa & fb & (ea | eb)).astype(int) * side
        cells = []
        for hold in (24, 12, 48):
            years = {0: [], 1: []}
            for pair in D.c.columns:
                c = D.c[pair].values
                if np.isfinite(c).sum() < 24 * 60:
                    continue
                for i, s, g in engine(c, D.h[pair].values, D.l[pair].values, e[pair].fillna(0).values.astype(int),
                                      3 * stop[pair].values, nan, hold):
                    k = np.searchsorted(starts, D.idx.asi8[i], side="right") - 1
                    if 0 <= k < 2:
                        years[k].append(g - 2 * cost[pair])
            cells.append("%dh: %s" % (hold, " ".join("%+6.0f (%d)" % (np.mean(v) * 1e4 if v else np.nan, len(v))
                                                    for v in years.values())))
        print("  %-48s %-5s %s" % (a + (" + " + b if b else ""), "long" if side > 0 else "short", " | ".join(cells)))


if __name__ == "__main__":
    main()
