"""H128: scalping only the few coins in play, second by second.

H127 scalped all 20 pairs all the time and every design lost about 20 basis points a trade. A coin in
play (news, a breakout, a volume surge) may move enough inside a minute to pay the fees when a quiet
one cannot, so this scalps only the 3 hottest pairs at a time, chosen from what was known before:

  D1 volume surge: the 3 pairs whose dollar volume over the last 24 hours is highest against
     their own average day of the 7 before (re-picked hourly)
  D2 biggest movers: the 3 with the largest absolute return over the last 24 hours
  D3 most volatile: the 3 with the highest standard deviation of 1-second returns over 24 hours
  D4 hot hour: the 3 whose dollar volume over the last hour is highest against their own
     average hour of the 24 before

On the chosen pairs it runs H127's simulation (in one second after the signal, out at the target,
the stop or 55 seconds; the same costs) with these signals, longs and shorts, fixed before running:
S1 follow a 15 or 30 bp move over 20 seconds, S2 fade it, S3 volume burst, S6 spike fade, with the
25/20 bp exits. Data: the 20 pairs of research/klines1s.py, July to September 2026.

Pass, as H127: the mean net return per trade positive in all 3 months (20 trades or more each) and
over all of them, and again with twice the spread; and better per trade than the same signal on
every pair (H127).

    python -m research.h128_hot_coins
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd

from research.h127_second_scalping import DATA, MONTHS, entries, features, load, signals, simulate, spreads

HOUR = 3600
TOP = 3
KEEP = ("S1 follow 15bp/20s", "S1 follow 30bp/20s", "S2 fade 15bp/20s", "S2 fade 30bp/20s",
        "S3 volume burst", "S6 spike fade")


def hourly(d):
    """Per hour: dollar volume, last close, sum of squared 1-second log returns."""
    n = len(d["c"]) // HOUR * HOUR
    dv = (d["v"][:n] * d["c"][:n]).reshape(-1, HOUR).sum(1)
    close = d["c"][:n].reshape(-1, HOUR)[:, -1]
    r = np.r_[0.0, np.diff(np.log(d["c"][:n]))]
    ss = (r ** 2).reshape(-1, HOUR).sum(1)
    return dv, close, ss


def picks(stats):
    """{rule: (hours x pairs) bool}: the pairs chosen for each hour from hours before it."""
    pairs = list(stats)
    dv = pd.DataFrame({p: stats[p][0] for p in pairs})
    close = pd.DataFrame({p: stats[p][1] for p in pairs})
    ss = pd.DataFrame({p: stats[p][2] for p in pairs})
    day_dv = dv.rolling(24).sum()
    score = {
        "D1 volume surge": day_dv / (dv.rolling(24 * 8).sum() - day_dv) * 7,
        "D2 biggest movers": np.log(close / close.shift(24)).abs(),
        "D3 most volatile": np.sqrt(ss.rolling(24).sum()),
        "D4 hot hour": dv / dv.shift(1).rolling(24).mean(),
    }
    out = {}
    for rule, s in score.items():
        s = s.shift(1)                                     # known at the start of the hour
        rank = s.rank(axis=1, ascending=False)
        out[rule] = (rank <= TOP) & s.notna()
    return out, pairs


def main() -> None:
    warnings.filterwarnings("ignore")
    spread = spreads()
    symbols = sorted({os.path.basename(p).rsplit("-", 2)[0] for p in glob.glob(os.path.join(DATA, "*.parquet"))})
    # Hourly stats run across months, so the 24-hour and 8-day windows are filled at each month's start.
    stats = {s: [np.array([]), np.array([]), np.array([])] for s in symbols}
    month_hours = []
    for month in MONTHS:
        hours = None
        for sym in symbols:
            d = load(sym, month)
            h = hourly(d)
            hours = len(h[0])
            for k in range(3):
                stats[sym][k] = np.r_[stats[sym][k], h[k]]
        month_hours.append(hours)
    chosen, pairs = picks(stats)
    starts = np.r_[0, np.cumsum(month_hours)]
    for rule, m in chosen.items():
        share = m.mean().sort_values(ascending=False)
        print("%s: most often chosen %s" % (rule, ", ".join("%s %.0f%%" % (p[:-4], 100 * v) for p, v in share.head(6).items())))

    res = {}
    for mi, month in enumerate(MONTHS):
        b = load("BTCUSDT", month)
        fb = features(b)
        btc_z = fb["r10"] / (fb["sd1"] * np.sqrt(10))
        for sym in symbols:
            d = load(sym, month)
            f = features(d)
            sigs, _ = signals(f, None if sym == "BTCUSDT" else btc_z)
            n = len(d["c"])
            ok = np.isfinite(f["sd1"]) & (f["sd1"] > 0)
            tgt, stp = np.full(n, 0.0025), np.full(n, 0.002)
            hour_of = np.minimum(np.arange(n) // HOUR, month_hours[mi] - 1) + starts[mi]
            masks = {"all pairs": np.ones(n, dtype=bool)}
            for rule, m in chosen.items():
                masks[rule] = m[sym].values[hour_of]
            for name in KEEP:
                for side in (1, -1):
                    for rule, mask in masks.items():
                        i = entries(sigs[name], side, ok & mask)
                        if not len(i):
                            continue
                        for mult in (1, 2):
                            half = max(spread.get(sym, 0.002) / 2, 0.00005) * mult
                            g, net, _ = simulate(d, i, side, tgt, stp, half)
                            res.setdefault((name, side, rule, mult), []).append((month, g, net))
            print("  done", month, sym, flush=True)

    print("\nH128 scalping the hottest %d pairs. Net bp per trade by month %s" % (TOP, " ".join(MONTHS)))
    passed = []
    for name in KEEP:
        for side in (1, -1):
            base = res.get((name, side, "all pairs", 1))
            base_net = np.concatenate([r[2] for r in base]).mean() if base else np.nan
            for rule in ["all pairs"] + list(chosen):
                verdicts = []
                for mult in (1, 2):
                    rows = res.get((name, side, rule, mult), [])
                    if not rows:
                        verdicts.append(False)
                        continue
                    net = np.concatenate([r[2] for r in rows])
                    cells, good, have = [], 0, 0
                    for month in MONTHS:
                        x = np.concatenate([r[2] for r in rows if r[0] == month] or [np.array([])])
                        if len(x) < 20:
                            cells.append("     --")
                            continue
                        have += 1
                        good += x.mean() > 0
                        cells.append("%+7.1f" % (1e4 * x.mean()))
                    verdicts.append(have == len(MONTHS) and good == have and net.mean() > 0)
                    if mult == 1:
                        gross = np.concatenate([r[1] for r in rows])
                        print("  %-19s %-5s %-18s %s | %6d trades (%5.0f/day) | bp gross %+6.1f net %+6.1f | "
                              "win %3.0f%% | months %d/%d" % (name, "long" if side > 0 else "short", rule,
                                                              " ".join(cells), len(net), len(net) / 92.0,
                                                              1e4 * gross.mean(), 1e4 * net.mean(),
                                                              100 * (net > 0).mean(), good, have), flush=True)
                        better = net.mean() > base_net
                if rule != "all pairs" and all(verdicts) and better:
                    passed.append((name, "long" if side > 0 else "short", rule))
    print("Passed:", passed or "none")


if __name__ == "__main__":
    main()
