"""H14: can full-exposure trend following on BTC beat simply holding BTC?

Each rule turns hourly BTC closes into a position between -1 and 1 (decided at an hour's
close, held through the next hour). Long/flat versions go to cash when the trend is down;
long/short versions go short (Roostoo shorts are 1x). Costs: 0.1% of every unit traded plus
0.02% slippage, the taker fee, which also applies to shorts.

Rule families:
  ema       long while the fast EMA is above the slow EMA
  above     long while the close is above an EMA
  tsmom     long while the return over the last k hours is positive
  donchian  long after a close above the N-hour high, out after a close below the N/2-hour low

Judged against holding BTC in each period. A rule only counts if its neighbours (nearby
settings) also work, otherwise it was luck.

    python -m research.h14_trend
"""
import numpy as np
import pandas as pd

from bot.metrics import summarize
from research.h1_signals import HOLD_OUT_START
from research.panel import load_panel

PERIODS = {"Y0 Oct24-Sep25": ("2024-10-01", "2025-10-01"),
           "Y1 Oct25-Jun26": ("2025-10-01", HOLD_OUT_START),
           "Jul-Sep26 (seen)": (HOLD_OUT_START, "2026-10-01")}
COST = 0.001 + 0.0002


def positions(close: pd.Series, high: pd.Series, low: pd.Series, family: str, a: int, b: int,
              short: bool) -> pd.Series:
    down = -1.0 if short else 0.0
    if family == "ema":
        fast = close.ewm(span=a, adjust=False).mean()
        slow = close.ewm(span=b, adjust=False).mean()
        return pd.Series(np.where(fast > slow, 1.0, down), index=close.index)
    if family == "above":
        ema = close.ewm(span=a, adjust=False).mean()
        return pd.Series(np.where(close > ema, 1.0, down), index=close.index)
    if family == "tsmom":
        past = close / close.shift(a) - 1
        return pd.Series(np.where(past > 0, 1.0, down), index=close.index)
    if family == "donchian":
        upper = high.rolling(a).max().shift(1)
        lower = low.rolling(max(a // 2, 2)).min().shift(1)
        pos = np.zeros(len(close))
        state = 0.0
        c, u, l = close.values, upper.values, lower.values
        for i in range(len(c)):
            if np.isnan(u[i]) or np.isnan(l[i]):
                pos[i] = 0.0
                continue
            if c[i] > u[i]:
                state = 1.0
            elif c[i] < l[i]:
                state = down
            pos[i] = state
        return pd.Series(pos, index=close.index)
    raise ValueError(family)


def evaluate(close: pd.Series, pos: pd.Series, start: str, end: str) -> dict:
    sel = (close.index >= start) & (close.index < end)
    r = close.pct_change().shift(-1)          # next hour's return, earned by this hour's position
    p = pos.where(sel)
    turnover = p.diff().abs()
    turnover.iloc[np.argmax(sel)] = abs(p[sel].iloc[0])  # entering at the start
    ret = (p * r - turnover * COST)[sel].fillna(0.0)
    equity = 100000.0 * (1 + ret).cumprod()
    curve = [(int(t.value // 1_000_000) + 3_600_000, v) for t, v in equity.items()]
    trades = [(int(t.value // 1_000_000), 1.0) for t, x in turnover[sel].items() if x > 0]
    st = summarize(curve, 100000.0, trades)
    st["exposure"] = float(p[sel].abs().mean())
    return st


def main() -> None:
    panel = load_panel(["BTC/USD"])
    close, high, low = panel["close"]["BTC/USD"], panel["high"]["BTC/USD"], panel["low"]["BTC/USD"]
    grid = ([("ema", f, s) for f, s in ((12, 48), (24, 96), (24, 168), (50, 200), (72, 288), (100, 400), (168, 672))]
            + [("above", n, 0) for n in (48, 96, 200, 400, 720)]
            + [("tsmom", k, 0) for k in (24, 72, 168, 336, 720)]
            + [("donchian", n, 0) for n in (48, 96, 168, 336)])
    hold = {}
    for label, (start, end) in PERIODS.items():
        x = close[(close.index >= start) & (close.index < end)]
        hold[label] = x.iloc[-1] / x.iloc[0] - 1
    print("Hold BTC: " + "   ".join("%s %+.1f%%" % (k, v * 100) for k, v in hold.items()))
    rows = []
    for family, a, b in grid:
        for short in (False, True):
            pos = positions(close, high, low, family, a, b, short)
            row = {"rule": "%s %s%s %s" % (family, a, "/%d" % b if b else "", "L/S" if short else "L/flat")}
            for label, (start, end) in PERIODS.items():
                st = evaluate(close, pos, start, end)
                row[label] = st["total_return"]
                row[label + " mdd"] = st["max_drawdown"]
                row[label + " tr/d"] = st["trades_per_day"]
            row["beats hold (dev)"] = sum(row[l] > hold[l] for l in list(PERIODS)[:2])
            rows.append(row)
    df = pd.DataFrame(rows).set_index("rule")
    pct = lambda v: "%+.1f%%" % (v * 100)
    pd.set_option("display.width", 220)
    cols = [c for c in df.columns if not c.endswith("tr/d")] + ["Y0 Oct24-Sep25 tr/d"]
    fmt = {c: (pct if not c.endswith("tr/d") and c != "beats hold (dev)" else (lambda v: "%.2f" % v)) for c in cols}
    fmt["beats hold (dev)"] = lambda v: "%d/2" % v
    print(df[cols].to_string(formatters=fmt))


if __name__ == "__main__":
    main()
