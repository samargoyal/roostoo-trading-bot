"""H143 (part 1): daily TimesFM 3.0 forecasts for every Roostoo crypto pair (the user's suggestion).

For each pair and each UTC day from 15 September 2023, TimesFM 3.0 (google/timesfm-3.0-pytorch,
zero-shot) forecasts the next three daily log closes from up to 512 daily log closes ending at
that day's close, so nothing after the day is seen. Saved per day and pair:
  ret1, ret3: the median forecast's change in log price over one and three days
  rr3: ret3 over the width of the 10th-90th percentile band at three days (a risk-adjusted forecast)
to runs/research/h143/timesfm_daily.csv, keyed by the day's open time (ms). research/h143_timesfm.py
backtests the all-in rotation ranked on these.

TimesFM's weights are under a non-commercial, non-production licence; this is research only. It
needs its own environment (torch, timesfm[torch]); run with that interpreter:

    <timesfm env>/python -m research.h143_timesfm_forecasts
"""
import csv
import os
import time

import numpy as np
import pandas as pd

OUT = os.path.join("runs", "research", "h143")
START = pd.Timestamp("2023-09-15", tz="UTC")
CONTEXT, MIN_CONTEXT, HORIZON = 512, 64, 3


def pairs():
    with open(os.path.join("research", "candidates.csv"), encoding="utf-8") as f:
        return [r["pair"] for r in csv.DictReader(f) if r["asset_type"] == "crypto" and r["pair"] not in ("PAXG/USD", "MUB/USD")]


def daily_log_closes(pair):
    path = os.path.join("data", "binance", pair.replace("/USD", "USDT") + "_1h.csv")
    if not os.path.exists(path):
        return None
    h = pd.read_csv(path).drop_duplicates("ts").sort_values("ts")
    s = pd.Series(h["close"].values, index=pd.to_datetime(h["ts"], unit="ms", utc=True))
    counts = s.resample("1D").count()
    d = s.resample("1D").last()[counts >= 24]                    # whole days only
    return np.log(d.astype("float64"))


def main() -> None:
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    os.makedirs(OUT, exist_ok=True)
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=256,
                                          device="cuda"))
    jobs = []
    for pair in pairs():
        d = daily_log_closes(pair)
        if d is None:
            continue
        values = d.values.astype(np.float32)
        for i, day in enumerate(d.index):
            if day < START or i + 1 < MIN_CONTEXT:
                continue
            jobs.append((pair, int(day.value // 10 ** 6), values[max(0, i + 1 - CONTEXT):i + 1]))
    print("forecasting %d pair-days" % len(jobs), flush=True)
    started = time.time()
    rows = []
    chunk = 4096
    for k in range(0, len(jobs), chunk):
        part = jobs[k:k + chunk]
        outs = list(model.predict_batch([c for _, _, c in part], horizon=HORIZON, return_quantiles=True))
        for (pair, day, ctx), o in zip(part, outs):
            last = float(ctx[-1])
            f = np.asarray(o.forecast, dtype=float).reshape(-1)
            q = np.asarray(o.quantiles, dtype=float).reshape(HORIZON, -1)
            width = float(q[-1, -1] - q[-1, 0])
            ret3 = float(f[-1] - last)
            rows.append((day, pair, float(f[0] - last), ret3, ret3 / width if width > 0 else 0.0))
        print("  %d / %d (%.0fs)" % (min(k + chunk, len(jobs)), len(jobs), time.time() - started), flush=True)
    pd.DataFrame(rows, columns=["day", "pair", "ret1", "ret3", "rr3"]).to_csv(
        os.path.join(OUT, "timesfm_daily.csv"), index=False)
    print("saved", os.path.join(OUT, "timesfm_daily.csv"))


if __name__ == "__main__":
    main()
