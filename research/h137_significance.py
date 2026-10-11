"""H137: are the results real? Randomness checks at the 5% level (the user's request).

Fixed before running:
  B1 (the live setup) and the other kept setups: is the mean daily return above zero beyond chance?
      A moving-block bootstrap (blocks of 7 days, 10,000 resamples) of the October 2023
      to October 2026 backtest's daily returns, centred on zero; one-sided p. And the deflated Sharpe
      ratio (Bailey and Lopez de Prado): the probability that the Sharpe ratio beats the best of N
      unskilled designs by luck, for N = 100 and 1,000, with the spread of Sharpe ratios taken from
      the kept setups and variants backtested over the same years.
  H135's meta-labeled dip buys (cutoff 0.67): in validation and the holdout, (a) do the chosen trades
      beat the same number of candidates chosen at random? 1,000 shuffles of the model's
      probabilities among that period's candidates, the same cutoff and one-trade-per-pair rule; p
      for the mean and for the win rate; (b) is their mean above zero? a bootstrap that resamples
      whole days (trades on one day are not independent).
  H131's capitulation buy and H134's 4-hour residual dip: the mean above zero by the same day
      bootstrap, in each period; and, for the capitulation buy, the Bonferroni level for having
      tried 1,008 designs (0.05 / 1,008).

    python -m research.h137_significance
"""
import math
import os
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

from research import h130_high_win_rate as h
from research import h131_capitulation as cap
from research import h135_meta_labels as meta
from research.h132_frequent_bounces import HOLDOUT, run
from research.h134_strong_coin_dips import setup as dips_setup

RNG = np.random.default_rng(0)
B = 10_000
SETUPS = ["comp_b1_60", "comp_b1", "comp_book_cap3", "comp_k2_3", "comp_r54b", "b1_60_stop10", "b1_60_stop15",
          "b1_60_stop8atr", "rot_entry0", "rot_entry2", "rot_entry5", "book_entry0", "both_entry0", "rot_atr4",
          "rot_atr6", "book_long_atr8", "book_short_atr5", "b1_60_rotshorts", "b1_rot1h", "b1_rot4h", "b1_rot12h",
          "b1_intrabar"]


def daily(name):
    path = os.path.join("runs", "research", "sixday", name, "taker", "equity.csv")
    if not os.path.exists(path):
        return None
    e = pd.read_csv(path)
    e["time"] = pd.to_datetime(e["time"])
    return e.set_index("time")["equity"].resample("1D").last().pct_change().dropna().values


def block_bootstrap_p(r, block=7):
    """One-sided p for mean > 0: a moving-block bootstrap (blocks of 7 days) of the centred series."""
    n, x = len(r), r - r.mean()
    k = -(-n // block)
    starts = RNG.integers(0, n - block + 1, size=(B, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return (np.sum(x[idx].mean(axis=1) >= r.mean()) + 1) / (B + 1)


def deflated_sharpe(r, n_trials, var_sr):
    z, g = NormalDist().inv_cdf, 0.5772156649
    sr0 = math.sqrt(var_sr) * ((1 - g) * z(1 - 1 / n_trials) + g * z(1 - 1 / (n_trials * math.e)))
    sr = r.mean() / r.std(ddof=1)
    skew = ((r - r.mean()) ** 3).mean() / r.std() ** 3
    kurt = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    return NormalDist().cdf((sr - sr0) * math.sqrt(len(r) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def day_bootstrap_p(ts, net):
    """One-sided p for mean > 0, resampling whole days (centred)."""
    days = ts // 86_400_000
    uniq, inv = np.unique(days, return_inverse=True)
    sums = np.bincount(inv, weights=net - net.mean())
    counts = np.bincount(inv)
    k = len(uniq)
    hits = 0
    for _ in range(B):
        pick = RNG.integers(k, size=k)
        hits += sums[pick].sum() / counts[pick].sum() >= net.mean()
    return (hits + 1) / (B + 1), k


def main() -> None:
    warnings.filterwarnings("ignore")
    print("== Kept setups: mean daily return > 0 (block bootstrap) and deflated Sharpe, Oct 2023 - Oct 2026")
    rets = {n: r for n in SETUPS if (r := daily(n)) is not None}
    srs = np.array([r.mean() / r.std(ddof=1) for r in rets.values()])
    var_sr = srs.var(ddof=1)
    for name in ("comp_b1_60", "comp_b1", "comp_book_cap3", "comp_k2_3", "comp_r54b"):
        r = rets.get(name)
        if r is None:
            continue
        print("  %-15s mean %+.3f%%/day, Sharpe %.2f, p(mean > 0) = %.4f, deflated Sharpe: N=100 %.2f, N=1000 %.2f" % (
            name, 100 * r.mean(), r.mean() / r.std(ddof=1) * math.sqrt(365), block_bootstrap_p(r),
            deflated_sharpe(r, 100, var_sr), deflated_sharpe(r, 1000, var_sr)))
        if name == "comp_b1_60":
            for y0, y1 in (("2023-10", "2024-10"), ("2024-10", "2025-10"), ("2025-10", "2026-10")):
                e = pd.read_csv(os.path.join("runs", "research", "sixday", name, "taker", "equity.csv"))
                e["time"] = pd.to_datetime(e["time"])
                s = e.set_index("time")["equity"].resample("1D").last().pct_change().dropna()
                ry = s[(s.index >= y0) & (s.index < y1)].values
                print("     %s to %s: mean %+.3f%%/day, p = %.4f" % (y0, y1, 100 * ry.mean(), block_bootstrap_p(ry)))

    print("\n== H135 meta-labeled dip buys (cutoff from development)")
    sc = meta.score(meta.dataset())
    cut = sc["cut"]
    print("  cutoff %.2f" % cut)
    for key in ("val", "holdout", "comp"):
        d = sc[key]
        chosen = meta.one_per_pair(d[d["p"] >= cut])
        if len(chosen) < 2:
            print("  %-8s %d trades: too few to test" % (key, len(chosen)))
            continue
        real_mean, real_wr = chosen["net"].mean(), (chosen["net"] > 0).mean()
        shuffled_means, shuffled_wrs = [], []
        p = d["p"].values
        for _ in range(1000):
            d2 = d.assign(p=RNG.permutation(p))
            c2 = meta.one_per_pair(d2[d2["p"] >= cut])
            shuffled_means.append(c2["net"].mean())
            shuffled_wrs.append((c2["net"] > 0).mean())
        pm = (np.sum(np.array(shuffled_means) >= real_mean) + 1) / 1001
        pw = (np.sum(np.array(shuffled_wrs) >= real_wr) + 1) / 1001
        pb, k = day_bootstrap_p(chosen["ts"].values, chosen["net"].values)
        print("  %-8s %4d trades on %d days: WR %.1f%% mean %+.2f%% | vs random picks: p(mean) = %.3f, p(WR) = %.3f "
              "(random picks: WR %.1f%%, mean %+.2f%%) | p(mean > 0) = %.4f" % (
                  key, len(chosen), k, 100 * real_wr, 100 * real_mean, pm, pw, 100 * np.mean(shuffled_wrs),
                  100 * np.mean(shuffled_means), pb))

    print("\n== H131 capitulation buy (BTC -6%, closed upper half, +1.5 ATR / -3 ATR / 3 days)")
    h.MAX_H = 3 * 24                                   # the meta-label step set it to 12 hours
    coins = cap.build()
    a = cap.evaluate(coins, "BTC -6%, closed upper half")[(1.5, 3.0, 3)]
    for label, (s, e) in (("dev", h.DEV), ("val", h.VAL), ("2018-20", cap.HOLDOUT), ("all", ("2018-01-01", "2026-10-13"))):
        sel = (a[:, 0] >= h.ms(s)) & (a[:, 0] < h.ms(e))
        if sel.sum() < 2:
            continue
        pb, k = day_bootstrap_p(a[sel, 0], a[sel, 1])
        print("  %-8s %3d trades on %2d days: mean %+.2f%%, p(mean > 0) = %.4f" % (label, sel.sum(), k, 100 * a[sel, 1].mean(), pb))
    print("  Bonferroni level for 1,008 designs: %.6f" % (0.05 / 1008))

    print("\n== H134 4-hour residual dip (2 sd, uptrend; +1.5 / -2 daily moves / 12h)")
    rows = run(dips_setup(), (1.5,), (2.0,), (12,), 1)
    a = next(x[3] for x in rows if x[0] == "P2 residual 4h 2 sd, uptrend")
    for label, (s, e) in (("dev", h.DEV), ("val", h.VAL), ("2018-20", HOLDOUT), ("comp", ("2026-10-04 13:00", "2026-10-13"))):
        sel = (a[:, 0] >= h.ms(s)) & (a[:, 0] < h.ms(e))
        if sel.sum() < 2:
            continue
        pb, k = day_bootstrap_p(a[sel, 0], a[sel, 1])
        print("  %-8s %4d trades on %3d days: mean %+.2f%%, p(mean > 0) = %.4f" % (label, sel.sum(), k, 100 * a[sel, 1].mean(), pb))


if __name__ == "__main__":
    main()
