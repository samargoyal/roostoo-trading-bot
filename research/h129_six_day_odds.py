"""H129: the odds of a 6.6% gain in six days for the kept competition setups.

On 12 October 2026 the account stood at -6% with six days of the competition left; gaining 6.6%
would bring it back to +0.2%. Cash cannot do that, so the question was which kept setup most often
gains 6.6% in six days, and how often it loses 5% or 10% instead. Each setup runs through the
production backtester from October 2023 to October 2026 (taker fees, the conservative case), and
every six-day window starting on an hour is counted.

    python -m research.h129_six_day_odds
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

OUT = os.path.join("runs", "research", "sixday")
SETUPS = ["comp_book_cap3", "comp_b1", "comp_b1_60", "comp_k2_3", "comp_r54b"]
NEED, HOURS = 0.066, 144
YEARS = ["2023-10-01", "2024-10-01", "2025-10-01", "2026-10-01"]


def curve(name):
    path = os.path.join(OUT, name, "taker", "equity.csv")
    if not os.path.exists(path):
        subprocess.run([sys.executable, "-m", "bot.backtest", "--config", os.path.join("config", name + ".json"),
                        "--start", YEARS[0], "--end", YEARS[-1], "--out", os.path.join(OUT, name)],
                       check=True, stdout=subprocess.DEVNULL)
    e = pd.read_csv(path)
    return e.set_index(pd.to_datetime(e["time"], utc=True))["equity"].resample("1h").last().ffill()


def line(r, low):
    return ("gain >= 6.6%% %4.1f%% | loss >= 5%% %4.1f%% | loss >= 10%% %4.1f%% | median %+5.1f%% | "
            "dips 8%% on the way %4.1f%%" % (100 * (r >= NEED).mean(), 100 * (r <= -0.05).mean(),
                                             100 * (r <= -0.10).mean(), 100 * r.median(), 100 * (low <= -0.08).mean()))


def main() -> None:
    print("Six-day windows, %s to %s, taker fees" % (YEARS[0], YEARS[-1]))
    for name in SETUPS:
        s = curve(name)
        r = (s.shift(-HOURS) / s - 1).dropna()
        low = (s[::-1].rolling(HOURS + 1).min()[::-1] / s - 1).reindex(r.index)
        print("%-15s all       %s" % (name, line(r, low)))
        for a, b in zip(YEARS, YEARS[1:]):
            sel = (r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))
            print("%-15s %s-%s %s" % ("", a[:4], b[2:4], line(r[sel], low[sel])))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
