"""H146: the all-in coin, long or short (the user: it must not ride an altcoin dump down).

The long-only bombers (H142-H145) held their coin through 11-13 October 2026, when the coins that
had run hardest fell 6-15% while BTC hardly moved, so BTC's trend filter never turned. Here each
re-pick chooses a side for the whole account, fixed before running:
  breadth N: short when fewer than half the eligible pairs are up over the last N days, else long
  symmetric: long the strongest pair if its score is larger in size than the weakest pair's, else
      short the weakest (scores over volatility)
  alt index N: short while an equal-weight index of every pair is down over the last N days
  BTC N: short while BTC is down over the last N days
with N = 3, 7, 14 or 28 days. Long the pair with the highest score, short the one with the lowest,
scores the return over 7, 14, 21 or 28 days (raw or over volatility); held 1, 2 or 3 days. Costs on
every change: 0.1% to sell and to buy, 0.1% to open and to close a short (/v6), and half each pair's
spread. A short's daily return is minus the pair's (rebalanced daily, as the simulator holds it).

Pass, all three: over October 2023 to October 2026 the mean daily return above zero at p < 0.05 (7-day
block bootstrap) and above 1,000 runs holding a random eligible pair on the same side each day at
p < 0.05; and +10% or more in the competition to the latest price. The last three days (the altcoin
fall of 11-13 October) are reported, not required (the user's choice).

    python -m research.h146_long_short_bomber
"""
import warnings

import numpy as np
import pandas as pd

from research import h144_bomber_search as h

SHORT_FEE = 0.001


def simulate(C, half, score, side, hold, random_rng=None, eligible=None):
    """Daily returns holding one pair: side[t] = +1 long the top score, -1 short the bottom."""
    R = (C.shift(-1) / C - 1).values
    S = score.values
    E = eligible.values if eligible is not None else np.isfinite(S)
    sd = side.values
    hs = half.reindex(C.columns).fillna(0.001).values
    n, m = S.shape
    out = np.zeros(n)
    cur_j, cur_s = -1, 0
    for t in range(n - 1):
        cost = 0.0
        if t % hold == 0:
            s = int(sd[t]) if np.isfinite(sd[t]) else 0
            ok = np.isfinite(S[t]) & np.isfinite(R[t]) & E[t]
            if s == 0 or not ok.any():
                j = -1
            elif random_rng is not None:
                j = int(random_rng.choice(np.flatnonzero(ok)))
            else:
                row = np.where(ok, S[t], np.nan)
                j = int(np.nanargmax(row)) if s > 0 else int(np.nanargmin(row))
            if (j, s) != (cur_j, cur_s):
                if cur_j >= 0:
                    cost += (h.FEE if cur_s > 0 else SHORT_FEE) + hs[cur_j]
                if j >= 0:
                    cost += (h.FEE if s > 0 else SHORT_FEE) + hs[j]
                cur_j, cur_s = j, (s if j >= 0 else 0)
        r = cur_s * R[t, cur_j] if cur_j >= 0 and np.isfinite(R[t, cur_j]) else 0.0
        out[t] = r - cost
    return pd.Series(out, index=C.index)


def sides(C, eligible, scores_vol):
    lc = np.log(C)
    out = {}
    for n in (3, 7, 14, 28):
        up = (lc - lc.shift(n) > 0).where(eligible)
        out["breadth %dd" % n] = pd.Series(np.where(up.mean(axis=1) >= 0.5, 1, -1), index=C.index)
        idx = (C / C.shift(n)).where(eligible).mean(axis=1)
        out["alt index %dd" % n] = pd.Series(np.where(idx >= 1, 1, -1), index=C.index)
        out["BTC %dd" % n] = pd.Series(np.where(C["BTC/USD"] >= C["BTC/USD"].shift(n), 1, -1), index=C.index)
    for name, sv in scores_vol.items():
        top, bot = sv.max(axis=1), sv.min(axis=1)
        out["symmetric (%s)" % name] = pd.Series(np.where(top >= -bot, 1, -1), index=C.index)
    return out


def main() -> None:
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
    vol_scores = {k: v for k, v in scores.items() if k.endswith("/vol")}
    side_rules = sides(C, eligible, vol_scores)
    dev = (C.index >= pd.Timestamp(h.DEV[0], tz="UTC")) & (C.index < pd.Timestamp(h.DEV[1], tz="UTC"))
    comp = C.index >= pd.Timestamp(h.COMP_START, tz="UTC")
    last3 = C.index >= C.index[-4]
    rows = []
    for rule, side in side_rules.items():
        for sname, score in scores.items():
            if rule.startswith("symmetric") and not rule.endswith("(%s)" % sname):
                continue                                         # symmetric decides with its own score
            for hold in (1, 2, 3):
                r_dev = simulate(C[dev], half, score[dev], side[dev], hold)
                tot = np.log1p(r_dev).sum()
                rand = []
                for _ in range(1000):
                    rr = simulate(C[dev], half, score[dev], side[dev], hold, random_rng=h.RNG, eligible=eligible[dev])
                    rand.append(np.log1p(rr).sum())
                p_rand = (np.sum(np.array(rand) >= tot) + 1) / 1001
                p_mean = h.block_p(r_dev.values)
                r_c = simulate(C[comp], half, score[comp], side[comp], hold)
                comp_ret = np.prod(1 + r_c.values[:-1]) - 1
                r_3 = simulate(C[last3], half, score[last3], side[last3], 1)
                last_ret = np.prod(1 + r_3.values[:-1]) - 1
                short_share = float((side[dev] < 0).mean())
                ok = p_mean < 0.05 and p_rand < 0.05 and comp_ret >= 0.10
                rows.append((ok, rule, sname, hold, np.expm1(tot), p_mean, p_rand, comp_ret, last_ret, short_share))
        print("  done", rule, flush=True)
    rows.sort(key=lambda x: (not x[0], -x[7]))
    print("\nH146: data to %s; %d designs" % (C.index[-1], len(rows)))
    print("%-24s %-7s %4s | %10s %7s %7s | %8s %8s | %s" % ("side rule", "score", "hold", "3 years", "p mean", "p rand",
                                                         "comp", "last 3d", "short share"))
    for ok, rule, sname, hold, tot, pm, pr, cr, lr, ss in rows:
        print("%s %-22s %-7s %4d | %+9.0f%% %7.4f %7.4f | %+7.1f%% %+7.1f%% | %3.0f%%" % (
            "PASS" if ok else "    ", rule, sname, hold, 100 * tot, pm, pr, 100 * cr, 100 * lr, 100 * ss))
    print("\n%d of %d designs pass all three" % (sum(r[0] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
