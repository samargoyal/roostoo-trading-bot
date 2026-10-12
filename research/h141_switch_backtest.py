"""H141: the H140 switch run inside the strategy (strategy.switch_sma_hours), through the production
backtester, against B1 and the book alone.

H140 chose its rule (the book alone with fast shorts while BTC is below its 200-day average, B1
otherwise) by switching between the two strategies' equity curves. Built into the strategy, the
switch shares one portfolio and one state, so the real run can differ; this checks it, with the
same yardsticks: the eight yearly folds from October 2018 (5,200 hours of warm-up so the 200-day
average is ready from the first day; B1 and the book re-run the same way), significance of the
mean daily return in validation and the holdout, the eight crashes, six-day odds from October 2023
and the competition's first week (config/comp_switch.json).

    python -m research.h141_switch_backtest
"""
import copy
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research import h139_book_significance as h139
from research.folds import FOLDS
from research.h98_crash_validation import chained, crashes
from research.h61_swing import load
from research.holdout2018 import HOLDOUT

OUT = os.path.join("runs", "research", "h141")
WARMUP = 5200
SWITCH = json.load(open("config/comp_switch.json"))["strategy"]
DESIGNS = {"switch (B1 above BTC's 200-day, book below)": SWITCH,
           "B1 60/40": h139.REFERENCE["B1 60/40 (live now)"],
           "book 240/960, shorts 24/72": h139.DESIGNS["H139 book 240/960, shorts 24/72"]}


def curve(job):
    from bot.backtest import run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import HOUR_MS, BinanceClient, load_history
    from research.folds import candidates, funding_table, ms
    tag, strategy, (start, end) = job
    path = os.path.join(OUT, "%s_%s.csv" % (tag.split()[0].replace("/", "-"), start))
    if not os.path.exists(path):
        cfg = load_config()
        apply_overrides(cfg, copy.deepcopy({"strategy": strategy, "backtest": {"warmup_bars": WARMUP}}))
        s, e = ms(start), ms(end)
        client = BinanceClient()
        slip = candidates(cfg)
        bars = {p: load_history(client, p, s - WARMUP * HOUR_MS, e, cfg.backtest.data_dir) for p in slip}
        bars = {p: b for p, b in bars.items() if b}
        r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                         monthly_universe=True, slippage_by_pair=slip, external_scores=funding_table(72))
        pd.DataFrame(r.curve, columns=["ts", "equity"]).to_csv(path, index=False)
    df = pd.read_csv(path)
    return tag, start, pd.Series(df["equity"].values / df["equity"].values[0],
                                 index=pd.to_datetime(df["ts"], unit="ms", utc=True))


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    folds = sorted(HOLDOUT + FOLDS)
    with ProcessPoolExecutor(max_workers=12) as pool:
        got = {(t, s): c for t, s, c in pool.map(curve, [(n, st, f) for n, st in DESIGNS.items() for f in folds])}
    D = load()
    btc = D.c["BTC/USD"].resample("1D").last().dropna()
    btc = btc[(btc.index >= pd.Timestamp(folds[0][0], tz="UTC")) & (btc.index < pd.Timestamp(folds[-1][1], tz="UTC"))]
    events = crashes(btc)
    print("%-44s %6s | %6s %7s %8s | %6s %7s | %7s %4s | %6s %6s %6s %5s | %s" % (
        "design", "devSR", "valSR", "val p", "val ret", "hldSR", "hold p", "crash", "lost", "g6.6", "l5", "l10", "DD",
        "years"))
    for n in DESIGNS:
        eq = chained([got[(n, s)] for s, _ in folds])
        cr = []
        for a, b, _ in events:
            seg = eq[(eq.index >= a) & (eq.index <= b + pd.Timedelta(days=1))]
            cr.append(seg.iloc[-1] / seg.iloc[0] - 1)
        s = eq.resample("1h").last().ffill()
        s = s[s.index >= pd.Timestamp("2023-10-01", tz="UTC")]
        six = (s.shift(-144) / s - 1).dropna()
        dv, vl, ho = h139.daily(eq, *h139.DEV), h139.daily(eq, *h139.VAL), h139.daily(eq, *h139.HOLD)
        yrs = " ".join("%+.0f%%" % (100 * (np.prod(1 + h139.daily(eq, a, b)) - 1)) for a, b in folds)
        print("%-44s %6.2f | %6.2f %7.4f %+7.0f%% | %6.2f %7.4f | %+6.1f%% %4d | %5.1f%% %5.1f%% %5.1f%% %4.0f%% | %s" % (
            n, h139.sharpe(dv), h139.sharpe(vl), h139.p_mean(vl), 100 * (np.prod(1 + vl) - 1), h139.sharpe(ho),
            h139.p_mean(ho), 100 * np.mean(cr), sum(x < 0 for x in cr), 100 * (six >= 0.066).mean(),
            100 * (six <= -0.05).mean(), 100 * (six <= -0.10).mean(), 100 * (1 - (s / s.cummax()).min()), yrs))


if __name__ == "__main__":
    main()
