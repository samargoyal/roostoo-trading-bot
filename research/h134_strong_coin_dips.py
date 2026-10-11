"""H134: buying hourly weakness in coins that are strong on longer horizons, several trades a day.

H130-H133 found the brief (a win rate of 60% or more, wins at least 1.5 times losses, positive after
costs) only in rare events, and the user wants trades every day. The one pattern with a real edge
throughout was buying weakness in strength, so this round looks for it at hourly resolution, where
it happens several times a day across the market.

Fixed before running, on hourly Binance bars of every Roostoo crypto pair (data/binance):
  P1 rsi2 pullback t: the pair above its 50-day average (1,200 hours) and up over the last week, the
     2-hour RSI below t (10, 5); shorts mirror it below the average and down over the week
  P2 residual dip k, n: over the last n hours (1, 4) the pair fell k sd (2, 3) more than its beta to
     BTC (over the last week) implies, in the same uptrend; shorts mirror it in a downtrend
  P3 leader pullback: the pair among the 10 with the best 14-day return and down 2 sd in the hour
Exits in units of the pair's daily move (its hourly sd over the last week times the square root of
24): targets 0.5, 1 and 1.5, stops 1, 1.5 and 2, held 6, 12 or 24 hours. Entries at the next hour's
open at market; costs as before.

The brief must hold in development (2020-24, 500 trades or more), validation (2024-26, 200 or more)
and the 2018-20 holdout (100 or more), with 3 trades or more in the competition's first week.
A second tier is reported too: a win rate of 55% or more, wins at least 1.2 times losses, positive
in every period.

    python -m research.h134_strong_coin_dips
"""
import warnings

import numpy as np
import pandas as pd

from research import h130_high_win_rate as h
from research.h132_frequent_bounces import HOLDOUT, run

COMP = ("2026-10-04 13:00", "2026-10-13")
TIER2 = (0.55, 1.2)


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def setup():
    spreads = h.pairs_and_spreads()
    frames = {}
    for pair in spreads:
        hr = h.load_hourly(pair)
        if hr is not None and len(hr) >= 24 * 260:
            frames[pair] = hr
    btc = frames["BTC/USD"].set_index("ts")["close"]
    closes = pd.DataFrame({p: f.set_index("ts")["close"] for p, f in frames.items()})
    ret14 = closes / closes.shift(336) - 1
    top10 = ret14.rank(axis=1, ascending=False) <= 10
    coins = {}
    for pair, hr in frames.items():
        c = hr["close"]
        ts = hr["ts"]
        lr = np.log(c).diff()
        sd = lr.rolling(168).std()
        unit = c * sd * np.sqrt(24)                              # the daily move, in price
        sma = c.rolling(1200).mean()
        week = c / c.shift(168) - 1
        up, down = (c > sma) & (week > 0), (c < sma) & (week < 0)
        r2 = rsi(c, 2)
        b = np.log(btc.reindex(ts.values)).diff().values
        lrv = lr.values
        cov = pd.Series(lrv * b).rolling(168).mean() - pd.Series(lrv).rolling(168).mean() * pd.Series(b).rolling(168).mean()
        var = pd.Series(b).rolling(168).var()
        beta = (cov / var).clip(-1, 4).values
        sig = {}
        for t in (10, 5):
            sig[("P1 rsi2 < %d, uptrend" % t, 1)] = (up & (r2 < t)).fillna(False).values
            sig[("P1 rsi2 > %d, downtrend" % (100 - t), -1)] = (down & (r2 > 100 - t)).fillna(False).values
        for n in (1, 4):
            rc = np.log(c / c.shift(n)).values
            rb = pd.Series(b).rolling(n).sum().values
            resid = rc - beta * rb
            rsd = pd.Series(resid).rolling(168).std().shift(1).values
            for k in (2, 3):
                sig[("P2 residual %dh %d sd, uptrend" % (n, k), 1)] = (up.values & (resid <= -k * rsd))
                sig[("P2 residual %dh %d sd, downtrend" % (n, k), -1)] = (down.values & (resid >= k * rsd))
        lead = top10[pair].reindex(ts.values).fillna(False).values
        sig[("P3 leader pullback", 1)] = lead & (lrv <= -2 * sd.shift(1).values)
        ok = (ts.shift(-1) - ts == 3_600_000).values
        coins[pair] = dict(h=hr, atr=unit.values, entry_row=np.arange(len(hr)) + 1.0, ok=ok,
                           sig={k: np.nan_to_num(v.astype(float)).astype(bool) for k, v in sig.items()},
                           half=max(spreads[pair] / 2, 0.00005))
    return coins


def passes(s, n, wr, ratio):
    return s["n"] >= n and s["wr"] >= wr and s["ratio"] >= ratio and s["mean"] > 0


def main() -> None:
    warnings.filterwarnings("ignore")
    rows = []
    for name, side, exit_, a in run(setup(), (0.5, 1.0, 1.5), (1.0, 1.5, 2.0), (6, 12, 24), 1):
        rows.append((name, side, exit_, h.stats(a, *h.DEV), h.stats(a, *h.VAL), h.stats(a, HOLDOUT[0], HOLDOUT[1]),
                     h.stats(a, *COMP)))

    def show(r):
        name, side, (tm, sm, hd), dv, vl, ho, cp = r
        print("  %s %s | target %.1f stop %.1f daily moves, %dh | %.1f trades a day (2024-26)\n     dev     %s\n"
              "     val     %s\n     2018-20 %s\n     comp    %s" % (
                  name, "long" if side > 0 else "short", tm, sm, hd, vl["n"] / 730.0,
                  h.fmt(dv), h.fmt(vl), h.fmt(ho), h.fmt(cp)))
    for label, (wr, ratio) in (("the brief (60%, 1.5x)", (h.MIN_WR, h.MIN_RATIO)), ("tier 2 (55%, 1.2x)", TIER2)):
        ok = [r for r in rows if passes(r[3], 500, wr, ratio) and passes(r[4], 200, wr, ratio)
              and passes(r[5], 100, wr, ratio) and r[6]["n"] >= 3]
        print("\nH134: %d designs; meeting %s in development, validation and the holdout, trading in the "
              "competition: %d" % (len(rows), label, len(ok)))
        for r in sorted(ok, key=lambda r: -r[4]["mean"])[:10]:
            show(r)
    print("\nBest of each signal by validation mean (200+ validation trades):")
    best = {}
    for r in rows:
        if r[4]["n"] >= 200 and (r[0], r[1]) not in best or (r[4]["n"] >= 200 and r[4]["mean"] > best[(r[0], r[1])][4]["mean"]):
            best[(r[0], r[1])] = r
    for r in sorted(best.values(), key=lambda r: -r[4]["mean"]):
        show(r)


if __name__ == "__main__":
    main()
