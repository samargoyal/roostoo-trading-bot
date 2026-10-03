"""H16: a multi-strategy blend that beats holding BTC with less risk than BTC itself.

Three books, each run continuously from October 2024 to October 2026 on its own capital:
  rotation  dual momentum: top 2 coins by 336h return, daily, BTC slow-trend filter with a
            fast exit (research.h15_rotation)
  defence   the bot's low-volatility long book plus the 15% short sleeve (research.sim);
            "defence long-only" is the same book without the short sleeve
  btc trend BTC while its 168h EMA is above its 672h EMA, else cash (research.h14_trend)
Blends hold fixed shares of capital in each book, rebalanced daily (0.12% on what moves).
Each period's figures come from the continuous curve, compared with holding BTC.

    python -m research.h16_blend
"""
import itertools

import numpy as np
import pandas as pd

from bot.metrics import max_drawdown, summarize
from research.h14_trend import positions
from research.h15_rotation import simulate as rotation
from research.run_variants import data, score
from research.sim import SimConfig, simulate

START, END = "2024-10-01", "2026-10-01"
PERIODS = {"Y0": ("2024-10-01", "2025-10-01"), "Y1": ("2025-10-01", "2026-07-01"),
           "HO (seen)": ("2026-07-01", "2026-10-01")}
COST = 0.0012


def books() -> pd.DataFrame:
    c = data()
    close = c["panel"]["close"]
    _, rot = rotation(close, c["mask"], START, END, 336, 2, 24, True, fast_exit=True, return_curve=True)
    cfg = SimConfig(short_on=0.15, short_off=0.15, short_trend_filter=False, short_stop_atr=10.0)
    _, defence = simulate(close, c["mask"], score("low_vol"), cfg, START, END,
                          high=c["panel"]["high"], low=c["panel"]["low"])
    _, defence_long = simulate(close, c["mask"], score("low_vol"), SimConfig(), START, END,
                               high=c["panel"]["high"], low=c["panel"]["low"])
    btc = close["BTC/USD"].ffill()
    pos = positions(btc, c["panel"]["high"]["BTC/USD"], c["panel"]["low"]["BTC/USD"], "ema", 168, 672, False)
    sel = (btc.index >= START) & (btc.index < END)
    r = btc.pct_change().shift(-1)
    turn = pos.diff().abs().fillna(pos.abs())
    trend = 100000.0 * (1 + (pos * r - turn * COST)[sel].fillna(0)).cumprod()
    hold = 100000.0 * btc[sel] / btc[sel].iloc[0]
    df = pd.DataFrame({"rotation": rot, "defence": defence, "defence long-only": defence_long,
                       "btc trend": trend, "hold BTC": hold}).dropna()
    return df / df.iloc[0]


def blend(curves: pd.DataFrame, weights: dict) -> pd.Series:
    rets = curves.pct_change().fillna(0.0)
    names = list(weights)
    target = np.array([weights[n] for n in names])
    w = target.copy()
    value, out = 1.0, []
    for t, row in rets[names].iterrows():
        if t.hour == 0:
            value -= value * np.abs(w - target).sum() / 2 * COST
            w = target.copy()
        g = float(w @ row.values)
        value *= 1 + g
        w = w * (1 + row.values) / (1 + g)
        out.append(value)
    return pd.Series(out, index=rets.index)


def stats(curve: pd.Series, start: str, end: str) -> dict:
    x = curve[(curve.index >= start) & (curve.index < end)]
    pts = [(int(t.value // 1_000_000), v) for t, v in x.items()]
    st = summarize(pts, x.iloc[0])
    return {"ret": st["total_return"], "mdd": st["max_drawdown"], "comp": st["composite"]}


def main() -> None:
    curves = books()
    print("correlation of daily returns between books:")
    print(curves.resample("1D").last().pct_change().corr().round(2).to_string())
    candidates = {"rotation only": {"rotation": 1.0}, "defence only": {"defence": 1.0},
                  "btc trend only": {"btc trend": 1.0}}
    for a in (0.3, 0.4, 0.5, 0.6):
        candidates["rot %.0f%% + def LONG-ONLY %.0f%%" % (a * 100, (1 - a) * 100)] = {
            "rotation": a, "defence long-only": 1 - a}
    for a, b in itertools.product((0.3, 0.4, 0.5, 0.6, 0.7), repeat=2):
        if a + b <= 1.0 + 1e-9:
            rest = round(1.0 - a - b, 2)
            name = "rot %.0f%% def %.0f%% trend %.0f%%" % (a * 100, b * 100, rest * 100)
            candidates[name] = {"rotation": a, "defence": b, "btc trend": rest}
    hold = {p: stats(curves["hold BTC"], s, e) for p, (s, e) in PERIODS.items()}
    print("\nhold BTC: " + "  ".join("%s %+.1f%% (mdd %.1f%%)" % (p, h["ret"] * 100, h["mdd"] * 100) for p, h in hold.items()))
    rows = []
    for name, w in candidates.items():
        curve = blend(curves, {k: v for k, v in w.items() if v > 0})
        row = {"blend": name}
        beats, safer = 0, 0
        for p, (s, e) in PERIODS.items():
            st = stats(curve, s, e)
            row[p + " ret"], row[p + " mdd"], row[p + " comp"] = st["ret"], st["mdd"], st["comp"]
            beats += st["ret"] > hold[p]["ret"]
            safer += st["mdd"] < hold[p]["mdd"]
        row["beats BTC"], row["less drawdown"] = beats, safer
        full = stats(curve, START, END)
        row["2y ret"], row["2y mdd"] = full["ret"], full["mdd"]
        rows.append(row)
    df = pd.DataFrame(rows)
    pct = lambda v: "%+.1f%%" % (v * 100)
    fmt = {c: pct for c in df.columns if c.endswith("ret") or c.endswith("mdd")}
    fmt.update({c: (lambda v: "%.2f" % v) for c in df.columns if c.endswith("comp")})
    pd.set_option("display.width", 260)
    print(df.sort_values(["beats BTC", "less drawdown", "2y ret"], ascending=False).to_string(index=False, formatters=fmt))
    full_hold = stats(curves["hold BTC"], START, END)
    print("\nhold BTC over two years: %+.1f%%, max drawdown %.1f%%" % (full_hold["ret"] * 100, full_hold["mdd"] * 100))


if __name__ == "__main__":
    main()
