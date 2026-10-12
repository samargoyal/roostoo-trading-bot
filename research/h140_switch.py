"""H140: switching between B1 and the crash-proof book (the user: combine them, the bot decides).

B1 60/40 (config/comp_b1_60.json) has the better odds of a strong week; the book with fast shorts
(H139, config/comp_book_fast_shorts.json) made money in every crash and every year with a third of
the drawdown. The account runs one of them at a time, chosen hourly by a rule, fixed before running:
  R1 BTC trend: book while BTC's 168-hour EMA is below its 672-hour EMA
  R2 bear market: book while BTC is below its 200-day average
  R3 volatility: book while BTC's 30-day volatility is above its median of the last year
  R4 breadth: book while fewer than half the pairs have their 240-hour EMA above their 960-hour one
  R5 crash alarm: book for 7 days after BTC's 7-day return reaches -10%
  R6 drawdown brake: book once the account is 10% (15%) below its peak, until within 5% of it again
  R7 strategy momentum: whichever of the two made more over the last 14 (30) days
Each switch is held at least 24 hours and costs 0.25% of equity (turnover between the two). The
switch acts on the two strategies' hourly equity curves from the production backtester (the eight
yearly folds, October 2018 to October 2026, taker fees, chained; the competition week from its
own runs), a rule's state at an hour's close deciding the next hour.

Selection: the best development Sharpe ratio (October 2020 to October 2024). Then, for the chosen
rule: the mean daily return above zero by a moving-block bootstrap in validation (October 2024 to
October 2026) and the holdout (October 2018 to October 2020); the eight crashes; the odds of a 6.6%
gain and of 5% and 10% losses in six days from October 2023; the competition's first week.

    python -m research.h140_switch
"""
import json
import math
import warnings

import numpy as np
import pandas as pd

from research import h139_book_significance as h139
from research.folds import FOLDS
from research.h95_aggressive_sleeve import bot_curve
from research.h98_crash_validation import chained, crashes
from research.h61_swing import load
from research.holdout2018 import HOLDOUT

COST, MIN_HOLD = 0.0025, 24
B1 = ("B1 60/40 (live now)", h139.REFERENCE["B1 60/40 (live now)"])
BOOK = ("H139 book 240/960, shorts 24/72", h139.DESIGNS["H139 book 240/960, shorts 24/72"])


def curves():
    folds = sorted(HOLDOUT + FOLDS)
    out = {}
    for tag, st in (B1, BOOK):
        out[tag] = chained([bot_curve((tag, st, f))[2] for f in folds])
    comp = {}
    for tag, path in ((B1[0], "comp_b1_60"), (BOOK[0], "h139_pick")):
        e = pd.read_csv("runs/research/since_oct4/%s/taker/equity.csv" % path)
        comp[tag] = pd.Series(e["equity"].values, index=pd.to_datetime(e["time"], utc=True))
    return out, comp


def signals(idx, closes):
    """{rule: Series of True (book) / False (B1) at each hour's close}, from prices only."""
    btc = closes["BTC/USD"].reindex(idx).ffill()
    out = {}
    out["R1 BTC trend"] = btc.ewm(span=168, adjust=False).mean() < btc.ewm(span=672, adjust=False).mean()
    out["R2 BTC below 200-day"] = btc < btc.rolling(4800, min_periods=2400).mean()
    vol = np.log(btc).diff().rolling(720).std()
    out["R3 BTC volatility high"] = vol > vol.rolling(8760, min_periods=4380).median()
    c = closes.reindex(idx).ffill()
    up = (c.ewm(span=240, adjust=False).mean() > c.ewm(span=960, adjust=False).mean()).where(c.notna())
    out["R4 breadth below half"] = up.mean(axis=1) < 0.5
    alarm = (btc / btc.shift(168) - 1) <= -0.10
    out["R5 crash alarm 7d"] = alarm.astype(float).rolling(168, min_periods=1).max() > 0
    return {k: v.fillna(False) for k, v in out.items()}


def simulate(rb1, rbook, rule=None, kind=None, arg=None):
    """Hourly returns of the switched account. rule: a precomputed Series (book when True); kind
    'drawdown' or 'momentum' are path-dependent and computed here."""
    n = len(rb1)
    a, b = rb1.values, rbook.values
    want = np.zeros(n, dtype=bool) if rule is None else rule.reindex(rb1.index).fillna(False).values
    eq_b1, eq_bk = np.cumprod(1 + a), np.cumprod(1 + b)
    out = np.empty(n)
    state, held, level, peak, switches = False, MIN_HOLD, 1.0, 1.0, 0
    for t in range(n):
        if t:
            if kind == "drawdown":
                dd = 1 - level / peak
                wish = (dd >= arg) if not state else (dd > 0.05)
            elif kind == "momentum":
                lb = arg
                wish = state if t < lb else (eq_bk[t - 1] / eq_bk[t - 1 - lb] > eq_b1[t - 1] / eq_b1[t - 1 - lb])
            else:
                wish = bool(want[t - 1])
            if wish != state and held >= MIN_HOLD:
                state, held = wish, 0
                level *= 1 - COST
                switches += 1
        r = b[t] if state else a[t]
        out[t] = r
        level *= 1 + r
        peak = max(peak, level)
        held += 1
    return pd.Series(out, index=rb1.index), switches


def main() -> None:
    warnings.filterwarnings("ignore")
    full, comp = curves()
    idx = full[B1[0]].index.intersection(full[BOOK[0]].index)
    rb1 = full[B1[0]].reindex(idx).pct_change().fillna(0)
    rbk = full[BOOK[0]].reindex(idx).pct_change().fillna(0)
    D = load()
    closes = D.c
    closes.index = closes.index.tz_convert("UTC") if closes.index.tz is not None else closes.index.tz_localize("UTC")
    rules = signals(idx, closes)
    designs = {"B1 always": (None, None, None), "book always": ("book", None, None)}
    for k in rules:
        designs[k] = (k, None, None)
    designs["R6 drawdown 10%"] = (None, "drawdown", 0.10)
    designs["R6 drawdown 15%"] = (None, "drawdown", 0.15)
    designs["R7 momentum 14d"] = (None, "momentum", 336)
    designs["R7 momentum 30d"] = (None, "momentum", 720)
    btc_d = D.c["BTC/USD"].resample("1D").last().dropna()
    btc_d = btc_d[(btc_d.index >= pd.Timestamp("2018-10-01", tz="UTC")) & (btc_d.index < pd.Timestamp("2026-10-01", tz="UTC"))]
    events = crashes(btc_d)
    years = max(1e-9, (idx[-1] - idx[0]).days / 365)
    rows = {}
    for name, (rule, kind, arg) in designs.items():
        if rule == "book":
            r, sw = rbk.copy(), 0
        else:
            r, sw = simulate(rb1, rbk, rules.get(rule), kind, arg)
        eq = (1 + r).cumprod()
        cr = []
        for a_, b_, _ in events:
            seg = eq[(eq.index >= a_) & (eq.index <= b_ + pd.Timedelta(days=1))]
            cr.append(seg.iloc[-1] / seg.iloc[0] - 1)
        s = eq[eq.index >= pd.Timestamp("2023-10-01", tz="UTC")]
        six = (s.shift(-144) / s - 1).dropna()
        dv, vl, ho = h139.daily(eq, *h139.DEV), h139.daily(eq, *h139.VAL), h139.daily(eq, *h139.HOLD)
        rows[name] = dict(dev=h139.sharpe(dv), val=h139.sharpe(vl), val_p=h139.p_mean(vl), val_ret=np.prod(1 + vl) - 1,
                          hold=h139.sharpe(ho), hold_p=h139.p_mean(ho), crash=np.mean(cr), lost=sum(x < 0 for x in cr),
                          g66=(six >= 0.066).mean(), l5=(six <= -0.05).mean(), l10=(six <= -0.10).mean(),
                          dd=1 - (s / s.cummax()).min(), sw=sw / years, rule=(rule, kind, arg))
    print("%-24s %6s | %6s %7s %8s | %6s %7s | %7s %4s | %6s %6s %6s %5s | %7s" % (
        "rule", "devSR", "valSR", "val p", "val ret", "hldSR", "hold p", "crash", "lost", "g6.6", "l5", "l10", "DD",
        "sw/yr"))
    for n, r in rows.items():
        print("%-24s %6.2f | %6.2f %7.4f %+7.0f%% | %6.2f %7.4f | %+6.1f%% %4d | %5.1f%% %5.1f%% %5.1f%% %4.0f%% | %7.1f" % (
            n, r["dev"], r["val"], r["val_p"], 100 * r["val_ret"], r["hold"], r["hold_p"], 100 * r["crash"], r["lost"],
            100 * r["g66"], 100 * r["l5"], 100 * r["l10"], 100 * r["dd"], r["sw"]))
    pick = max((n for n in rows if n not in ("B1 always", "book always")), key=lambda n: rows[n]["dev"])
    print("\nChosen on development Sharpe: %s" % pick)
    # the competition week: the two strategies' own runs from 4 October, the rule's state from prices
    ci = comp[B1[0]].index.intersection(comp[BOOK[0]].index)
    ci = ci[ci >= pd.Timestamp("2026-10-04 13:00", tz="UTC")]
    cb1, cbk = comp[B1[0]].reindex(ci).pct_change().fillna(0), comp[BOOK[0]].reindex(ci).pct_change().fillna(0)
    hourly = {}
    for p in ("BTC/USD",) + tuple(c for c in closes.columns if c != "BTC/USD"):
        f = "data/binance/%s_1h.csv" % p.replace("/USD", "USDT")
        try:
            x = pd.read_csv(f).drop_duplicates("ts")
            hourly[p] = pd.Series(x["close"].values, index=pd.to_datetime(x["ts"], unit="ms", utc=True))
        except OSError:
            continue
    allc = pd.DataFrame(hourly)
    rule, kind, arg = rows[pick]["rule"]
    week = {}
    for name in ("B1 always", "book always", pick):
        rl, kd, ag = rows[name]["rule"] if name == pick else designs[name]
        if rl == "book":
            r = cbk
        else:
            sig = signals(allc.index[allc.index >= pd.Timestamp("2025-06-01", tz="UTC")], allc).get(rl)
            r, _ = simulate(cb1, cbk, sig, kd, ag)
        week[name] = (1 + r).prod() - 1
    print("Competition week (4 October 13:00 to 11 October): " + ", ".join("%s %+.1f%%" % (k, 100 * v) for k, v in week.items()))
    print("Rule config:", json.dumps(rows[pick]["rule"]))
    _ = math


if __name__ == "__main__":
    main()
