"""H15: can momentum rotation beat holding BTC, rally or not?

Dual momentum. Every `rebalance` hours, rank the universe coins (point-in-time top 20, no
PAXG) by their return over the last `lookback` hours, and hold the top `k` in equal weights,
fully invested. A coin only qualifies if that return is positive (absolute momentum); empty
slots go to PAXG if its own return is positive, else to cash. Optionally the whole book
needs BTC above a slow EMA (168h over 672h). Costs 0.12% of every unit traded.

A setting only counts if it beats holding BTC in every period, and its neighbours should too.

    python -m research.h15_rotation
"""
import itertools

import numpy as np
import pandas as pd

from bot.config import UniverseConfig
from bot.metrics import summarize
from research.h1_signals import HOLD_OUT_START
from research.panel import cached_pairs, load_panel, monthly_universe

PERIODS = {"Y0": ("2024-10-01", "2025-10-01"), "Y1": ("2025-10-01", HOLD_OUT_START),
           "HO (seen)": (HOLD_OUT_START, "2026-10-01")}
COST = 0.0012
DEFENSIVE = "PAXG/USD"


def simulate(close: pd.DataFrame, mask: pd.DataFrame, start: str, end: str, lookback: int, k: int,
             rebalance: int, regime: bool, weighting: str = "equal", stop_atr: float = 0.0,
             brake: float = 0.0, fast_exit: bool = False, atr: pd.DataFrame = None,
             sig: pd.DataFrame = None, return_curve: bool = False):
    """Dual momentum with optional risk overlays:
      weighting  "equal" or "inverse_vol" (168h volatility)
      stop_atr   > 0: sell a coin that closes this many ATRs below its highest close since entry;
                 it stays out until the next rebalance
      brake      > 0: halve the book while equity is this far below its peak (back to full
                 size once the drawdown is under half of it)
      fast_exit  leave everything as soon as the BTC trend filter turns off, not at the rebalance
    """
    sel = np.where((close.index >= start) & (close.index < end))[0]
    prices = close.ffill().values
    atr_v = None if atr is None else atr.values
    sig_v = None if sig is None else sig.values
    pairs = list(close.columns)
    d = pairs.index(DEFENSIVE)
    btc = close["BTC/USD"].ffill()
    trend_on = (btc.ewm(span=168, adjust=False).mean() > btc.ewm(span=672, adjust=False).mean()).values
    m = mask.values
    w = np.zeros(len(pairs))
    plan = np.zeros(len(pairs))        # the rebalance's intended weights, before overlays
    top = np.zeros(len(pairs))
    stopped = np.zeros(len(pairs), dtype=bool)
    equity, peak, braking = 100000.0, 100000.0, False
    intended = np.zeros(len(pairs))
    curve, trades = [], []

    def trade_to(target, i):
        nonlocal w, equity
        turnover = np.abs(target - w).sum()
        if turnover > 1e-9:
            equity -= equity * turnover * COST
            trades.append((close.index[i], turnover))
        w = target

    for step, i in enumerate(sel):
        p = prices[i]
        peak = max(peak, equity)
        dd = 1 - equity / peak
        if brake > 0:
            if braking and dd <= brake / 2:
                braking = False
            elif not braking and dd >= brake:
                braking = True
        if step % rebalance == 0:
            past = p / prices[i - lookback] - 1
            eligible = m[i] & np.isfinite(past) & (past > 0)
            eligible[d] = False
            plan = np.zeros(len(pairs))
            if not regime or trend_on[i]:
                order = [j for j in np.argsort(-np.nan_to_num(past, nan=-np.inf)) if eligible[j]][:k]
                if order:
                    if weighting == "inverse_vol" and sig_v is not None:
                        inv = np.array([1 / sig_v[i, j] if sig_v[i, j] > 0 else 0 for j in order])
                        plan[order] = inv / inv.sum() if inv.sum() > 0 else 1.0 / len(order)
                    else:
                        plan[order] = 1.0 / k
            empty = 1.0 - plan.sum()
            if empty > 0 and m[i, d] and np.isfinite(past[d]) and past[d] > 0:
                plan[d] += empty
            stopped[:] = False
            new = (plan > 0) & (w <= 0)
            top = np.where(new, p, top)
        held = w > 0
        top = np.where(held, np.maximum(top, p), top)
        if stop_atr > 0 and atr_v is not None:
            hit = held & (p < top - stop_atr * np.nan_to_num(atr_v[i]))
            hit[d] = False
            stopped |= hit
        target = np.where(stopped, 0.0, plan)
        if fast_exit and regime and not trend_on[i]:
            target = np.where(np.arange(len(pairs)) == d, target, 0.0)
        if braking:
            target = target * 0.5
        if step % rebalance == 0:
            trade_to(target.copy(), i)
            intended = target.copy()
        else:
            # Between rebalances, only positions whose intended weight changed are traded.
            changed = np.abs(target - intended) > 1e-12
            if changed.any():
                new_w = w.copy()
                new_w[changed] = target[changed]
                trade_to(new_w, i)
                intended = target.copy()
        if i + 1 < len(prices):
            r = prices[i + 1] / prices[i] - 1
            growth = float(np.nansum(w * r))
            equity *= 1 + growth
            w = w * (1 + np.nan_to_num(r)) / (1 + growth)
        curve.append((int(close.index[i].value // 1_000_000) + 3_600_000, equity))
    stats = summarize(curve, 100000.0, [(int(t.value // 1_000_000), x) for t, x in trades])
    if return_curve:
        return stats, pd.Series([v for _, v in curve], index=close.index[sel])
    return stats


def main() -> None:
    import sys
    pairs = cached_pairs()
    panel = load_panel(pairs)
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    close = panel["close"]
    btc = close["BTC/USD"]
    hold = {p: btc[(btc.index >= s) & (btc.index < e)].iloc[-1] / btc[(btc.index >= s) & (btc.index < e)].iloc[0] - 1
            for p, (s, e) in PERIODS.items()}
    print("Hold BTC: " + "  ".join("%s %+.1f%%" % (p, v * 100) for p, v in hold.items()))
    logret = np.log(close.ffill()).diff()
    sig = logret.rolling(168, min_periods=100).std()
    high, low = panel["high"].ffill(), panel["low"].ffill()
    prev = close.ffill().shift()
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    if "--overlays" in sys.argv:
        variants = []
        for lookback, k, rebalance in itertools.product((336, 720), (2, 3), (24, 168)):
            base = dict(lookback=lookback, k=k, rebalance=rebalance, regime=True)
            variants.append(("plain", base))
            variants.append(("fast exit", dict(base, fast_exit=True)))
            variants.append(("inv-vol + fast exit", dict(base, weighting="inverse_vol", fast_exit=True)))
            variants.append(("stops 10 + fast exit", dict(base, stop_atr=10.0, fast_exit=True)))
            variants.append(("stops 10 + exit + brake 15%", dict(base, stop_atr=10.0, fast_exit=True, brake=0.15)))
            variants.append(("all + inv-vol", dict(base, stop_atr=10.0, fast_exit=True, brake=0.15, weighting="inverse_vol")))
        rows = []
        for label, kw in variants:
            row = {"L": kw["lookback"], "k": kw["k"], "R": kw["rebalance"], "overlays": label}
            for p, (s, e) in PERIODS.items():
                st = simulate(close, mask, s, e, atr=atr, sig=sig, **kw)
                row[p] = st["total_return"]
                row[p + " mdd"] = st["max_drawdown"]
                row[p + " comp"] = st["composite"]
            row["beats BTC"] = sum(row[p] > hold[p] for p in PERIODS)
            rows.append(row)
        df = pd.DataFrame(rows)
        pct = lambda v: "%+.1f%%" % (v * 100)
        fmt = {c: pct for c in df.columns if c in PERIODS or c.endswith("mdd")}
        fmt.update({c: (lambda v: "%.2f" % v) for c in df.columns if c.endswith("comp")})
        pd.set_option("display.width", 250)
        print(df.to_string(index=False, formatters=fmt))
        summary = df.groupby("overlays").agg(beats_all=("beats BTC", lambda x: int((x == 3).sum())),
                                             worst_mdd=("Y0 mdd", "max"))
        print(summary)
        return

    rows = []
    for lookback, k, rebalance, regime in itertools.product((72, 168, 336, 720), (1, 2, 3, 5), (24, 168), (False, True)):
        row = {"lookback": lookback, "k": k, "rebal": rebalance, "regime": regime}
        for p, (s, e) in PERIODS.items():
            st = simulate(close, mask, s, e, lookback, k, rebalance, regime)
            row[p] = st["total_return"]
            row[p + " mdd"] = st["max_drawdown"]
        row["beats BTC"] = sum(row[p] > hold[p] for p in PERIODS)
        rows.append(row)
    df = pd.DataFrame(rows)
    pct = lambda v: "%+.1f%%" % (v * 100)
    fmt = {c: pct for c in df.columns if c in PERIODS or c.endswith("mdd")}
    pd.set_option("display.width", 200)
    print(df.sort_values(["beats BTC", "Y0"], ascending=False).to_string(index=False, formatters=fmt))
    print("\nsettings beating BTC in all 3 periods: %d of %d" % ((df["beats BTC"] == 3).sum(), len(df)))
    print("beating BTC in Y0 (the bull year): %d of %d" % ((df["Y0"] > hold["Y0"]).sum(), len(df)))


if __name__ == "__main__":
    main()
