"""H144: a wide search for the all-in coin, with significance, random picks and the competition.

The user's brief: the whole account in one coin, re-picked as moves rotate; it must have made +10% or
more in the competition so far and be significant. Fixed before running:

  simulation: every pair Roostoo lists with hourly Binance history (data/binance), daily closes at
      00:00 UTC; at each close the strategy ranks the pairs that have 30 days of history and holds the
      top one (or two at 50%) for the next 1, 2 or 3 days (then ranks again); every change of coin
      pays 0.1% to sell and to buy plus half of each pair's spread (research/candidates.csv); each
      design also run picking only among pairs up over the last 14 days (as the bot's rotation), cash
      when none is.
  rankings (each long the highest score): past return over 1, 2, 3, 5, 7, 10, 14, 21 and 28 days;
      the same over volatility (return / 20-day sd of daily returns); 14 days skipping the last
      one; the 1/2/3-week mix 2:2:1 of normal scores; distance to the 30-day high (closest first);
      the 7-day return times its volume against the 30-day average; return relative to BTC over 7
      and 14 days; the 1- and 3-day return reversed (the biggest loser); RSI(14); and, if their
      file exists, TimesFM 3.0's expected 1- and 3-day returns and the risk-adjusted 3-day forecast
      (research/h143_timesfm_forecasts.py).
  pass (all three): over October 2023 to October 2026, the mean daily return above zero at p < 0.05
      (a 7-day moving-block bootstrap) and above random picks at p < 0.05 (1,000 runs holding a
      random eligible pair under the same rules: the share of random runs with a total return as
      high); and +10% or more from the competition's first close (4 October 2026 00:00 UTC) to the
      latest price in the data (the current day so far). Every design is printed; the competition week is short and choosing on
      it flatters the winner.

    python -m research.h144_bomber_search
"""
import csv
import os
import warnings

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
FEE = 0.001
DEV = ("2023-10-01", "2026-10-01")
COMP_START = "2026-10-04"


def load():
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["asset_type"] == "crypto" and r["pair"] not in ("PAXG/USD", "MUB/USD")]
    closes, vols, half = {}, {}, {}
    for r in rows:
        path = os.path.join("data", "binance", r["pair"].replace("/USD", "USDT") + "_1h.csv")
        if not os.path.exists(path):
            continue
        h = pd.read_csv(path).drop_duplicates("ts").sort_values("ts")
        idx = pd.to_datetime(h["ts"], unit="ms", utc=True)
        c = pd.Series(h["close"].values, index=idx)
        v = pd.Series((h["close"] * h["volume"]).values, index=idx)
        n = c.resample("1D").count()
        whole = (n >= 24) | (n.index == n.index[-1])          # the latest day so far, for the competition
        closes[r["pair"]] = c.resample("1D").last()[whole]
        vols[r["pair"]] = v.resample("1D").sum()[whole]
        half[r["pair"]] = max(float(r["spread"]) / 2, 0.00005)
    C = pd.DataFrame(closes).sort_index()
    V = pd.DataFrame(vols).reindex(C.index)
    C = C[C.index >= pd.Timestamp("2023-01-01", tz="UTC")]
    return C, V.reindex(C.index), pd.Series(half)


def rankings(C, V):
    lr = np.log(C).diff()
    sd = lr.rolling(20, min_periods=15).std()
    out = {}
    for k in (1, 2, 3, 5, 7, 10, 14, 21, 28):
        r = np.log(C / C.shift(k))
        out["return %dd" % k] = r
        out["return %dd / vol" % k] = r / (sd * np.sqrt(k))
    out["return 14d skip 1d"] = np.log(C.shift(1) / C.shift(14))
    z = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)
    out["1/2/3-week mix 2:2:1"] = (2 * z(np.log(C / C.shift(7))) + 2 * z(np.log(C / C.shift(14)))
                                   + z(np.log(C / C.shift(21))))
    out["near 30-day high"] = C / C.rolling(30).max()
    out["7d return x volume surge"] = np.log(C / C.shift(7)) * (V.rolling(7).mean() / V.rolling(30).mean())
    btc = np.log(C["BTC/USD"])
    for k in (7, 14):
        out["vs BTC %dd" % k] = np.log(C / C.shift(k)).sub(btc - btc.shift(k), axis=0)
    out["biggest 1d loser"] = -np.log(C / C.shift(1))
    out["biggest 3d loser"] = -np.log(C / C.shift(3))
    d = C.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["RSI 14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tfm = os.path.join("runs", "research", "h143", "timesfm_daily.csv")
    if os.path.exists(tfm):
        f = pd.read_csv(tfm)
        f["day"] = pd.to_datetime(f["day"], unit="ms", utc=True)
        for col, name in (("ret1", "TimesFM 1-day"), ("ret3", "TimesFM 3-day"), ("rr3", "TimesFM risk-adjusted 3-day")):
            out[name] = f.pivot(index="day", columns="pair", values=col).reindex(index=C.index, columns=C.columns)
    eligible = C.notna() & C.shift(30).notna()
    return {k: v.where(eligible) for k, v in out.items()}, eligible


def simulate(C, half, score, hold, top, random_rng=None, eligible=None, rising=None):
    """Daily returns: at each close pick the top `top` by score (or random eligible ones), hold `hold` days.
    rising: only pairs up over the last 14 days may be picked (as the bot's rotation), else cash."""
    R = (C.shift(-1) / C - 1).values                                     # the next day's return
    S = score.values
    E = eligible.values if eligible is not None else np.isfinite(S)
    if rising is not None:
        S = np.where(rising.values, S, np.nan)
        E = E & rising.values
    h = half.reindex(C.columns).fillna(0.001).values
    n, m = S.shape
    out = np.zeros(n)
    held = np.zeros(m)
    for t in range(n - 1):
        if t % hold == 0:
            if random_rng is not None:
                cand = np.flatnonzero(E[t] & np.isfinite(R[t]))
                pick = random_rng.choice(cand, size=min(top, len(cand)), replace=False) if len(cand) else []
            else:
                row = np.where(np.isfinite(S[t]) & np.isfinite(R[t]), S[t], -np.inf)
                pick = [j for j in np.argsort(-row)[:top] if np.isfinite(row[j])]
            new = np.zeros(m)
            for j in pick:
                new[j] = 1.0 / top
            cost = np.sum(np.abs(new - held) * (FEE + h))
            held = new
        else:
            cost = 0.0
        r = np.nansum(held * np.nan_to_num(R[t]))
        out[t] = r - cost
        # weights drift with the day's returns
        grown = held * (1 + np.nan_to_num(R[t]))
        held = grown / grown.sum() * held.sum() if grown.sum() > 0 else held
    return pd.Series(out, index=C.index)


def block_p(r, block=7, n=10_000):
    x, k = r - r.mean(), -(-len(r) // block)
    starts = RNG.integers(0, len(r) - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :len(r)]
    return (np.sum(x[idx].mean(axis=1) >= r.mean()) + 1) / (n + 1)


def main() -> None:
    warnings.filterwarnings("ignore")
    C, V, half = load()
    ranks, eligible = rankings(C, V)
    dev = (C.index >= pd.Timestamp(DEV[0], tz="UTC")) & (C.index < pd.Timestamp(DEV[1], tz="UTC"))
    up14 = (C / C.shift(14) > 1).fillna(False)
    comp = C.index >= pd.Timestamp(COMP_START, tz="UTC")
    print("H144: %d pairs, data to %s; %d rankings x hold 1/2/3 days x top 1/2 x any/rising" % (
        C.shape[1], C.index[-1].date(), len(ranks)))
    # random picks: the benchmark distribution, per hold/top
    random_total = {}
    for gate in (False, True):
        for hold in (1, 2, 3):
            for top in (1, 2):
                totals = []
                for _ in range(1000):
                    r = simulate(C[dev], half, ranks["return 1d"][dev], hold, top, random_rng=RNG, eligible=eligible[dev],
                                 rising=up14[dev] if gate else None)
                    totals.append(np.log1p(r).sum())
                random_total[(gate, hold, top)] = np.array(totals)
    rows = []
    for name, score in ranks.items():
        for gate in (False, True):
            for hold in (1, 2, 3):
                for top in (1, 2):
                    r_dev = simulate(C[dev], half, score[dev], hold, top, rising=up14[dev] if gate else None)
                    r_comp = simulate(C[comp], half, score[comp], hold, top, rising=up14[comp] if gate else None)
                    tot = np.log1p(r_dev).sum()
                    p_mean = block_p(r_dev.values)
                    p_rand = (np.sum(random_total[(gate, hold, top)] >= tot) + 1) / 1001
                    comp_ret = np.prod(1 + r_comp.values[:-1]) - 1
                    ok = p_mean < 0.05 and p_rand < 0.05 and comp_ret >= 0.10
                    rows.append((ok, name + (", rising 14d" if gate else ""), hold, top, np.expm1(tot), p_mean, p_rand,
                                 comp_ret))
    rows.sort(key=lambda x: (not x[0], -x[7]))
    print("\n%-34s %4s %3s | %10s %8s %8s | %s" % ("ranking", "hold", "top", "3 years", "p mean", "p random", "competition"))
    for ok, name, hold, top, tot, pm, pr, cr in rows:
        print("%s %-32s %4d %3d | %+9.0f%% %8.4f %8.4f | %+6.1f%%" % ("PASS" if ok else "    ", name, hold, top, 100 * tot, pm, pr, 100 * cr))
    print("\n%d of %d designs pass all three" % (sum(r[0] for r in rows), len(rows)))


if __name__ == "__main__":
    main()
