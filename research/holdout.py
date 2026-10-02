"""The final check: finalists on the hold-out months (July to September 2026), run once.

Nothing in the development research looked at these months. A finalist that only works
on the development period should fall apart here.

    python -m research.holdout
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

from research.h1_signals import HOLD_OUT_START
from research.run_variants import data, score
from research.sim import SimConfig, simulate, with_fees

HOLD_OUT_END = "2026-10-01"
FINALISTS = [
    ("A current bot (momentum ranking)", "current", {}),
    ("C low-vol ranking", "low_vol", {}),
    ("F1 low-vol + vol target 2%", "low_vol", {"vol_target": 0.02}),
    ("F2 F1 + 40% shorts in bear", "low_vol", {"vol_target": 0.02, "short_off": 0.40, "short_regime_ema": 720}),
    ("L low-vol optimiser (MVO)", "low_vol", {"construction": "mvo", "decide_every": 24, "target_vol": 0.01}),
]


def run(job):
    name, score_name, overrides, maker = job
    c = data()
    cfg = with_fees(replace(SimConfig(), **overrides), maker)
    stats, _ = simulate(c["panel"]["close"], c["mask"], score(score_name), cfg,
                        HOLD_OUT_START, HOLD_OUT_END, high=c["panel"]["high"], low=c["panel"]["low"])
    return name, maker, stats


def main() -> None:
    jobs = [(n, s, o, m) for n, s, o in FINALISTS for m in (False, True)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, jobs))
    rows = {}
    for name, maker, st in results:
        row = rows.setdefault(name, {})
        if maker:
            row["return (maker)"] = st["total_return"]
        else:
            row.update({"return (taker)": st["total_return"], "max drawdown": st["max_drawdown"],
                        "sharpe": st["sharpe"], "sortino": st["sortino"], "composite": st["composite"],
                        "14d windows positive": st.get("window_positive_share", float("nan"))})
    df = pd.DataFrame(rows).T[["return (taker)", "return (maker)", "max drawdown", "sharpe", "sortino",
                               "composite", "14d windows positive"]]
    btc = data()["panel"]["close"]["BTC/USD"].loc[HOLD_OUT_START:HOLD_OUT_END]
    print("Hold-out %s to %s (BTC %+.1f%%)" % (HOLD_OUT_START, HOLD_OUT_END, (btc.iloc[-1] / btc.iloc[0] - 1) * 100))
    pct = lambda x: "{:.1%}".format(x)
    num = lambda x: "{:.2f}".format(x)
    print(df.to_string(formatters={c: (pct if c in ("return (taker)", "return (maker)", "max drawdown", "14d windows positive") else num) for c in df.columns}))


if __name__ == "__main__":
    main()
