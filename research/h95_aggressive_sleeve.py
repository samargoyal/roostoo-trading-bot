"""H95: an aggressive sleeve beside the rotation and the book (the user: "introduce aggressive
strats, swings too; the final split momentum 30, aggressive 30, book 40").

The live bot runs the rotation at 55% and the efficiency-weighted long-short book at 45%. The
user's target is three sleeves: the rotation 30%, an aggressive sleeve 30% and the book 40%.
Candidates for the aggressive sleeve, written down before running (long only, at most a quarter
of the sleeve per coin, the fee and half the spread on every change, hourly bars of the bot's
coins from 2018):

  A1  H61's swing ensemble: its nine Stage A survivors (breakout and retest, VCP, relative
      strength on BTC down days, Darvas boxes, all-time highs, negative funding in a bull
      market, the hottest meme, meme hype phases, the golden cross), equally weighted
  A2  breakout swings (H88's WR5): the 14-day Williams %R crossing up through -20 buys; out
      below -50, a stop one daily ATR below, or after 14 days
  A3  the daily dip-buy the user sent (H89): RSI(3) below 22, the 100-day EMA above its level 5
      days ago, the close 1% below the 10-day VWAP; out when RSI(3) closes above 55
  A4  fast momentum: a second rotation, the 2 coins best on 3- and 7-day returns together,
      re-picked daily, while BTC's filter is on (the bot's own rotation code, at 99%)
  A5  A1-A4 in equal parts

Portfolios, each from the full backtester (the rotation and book, with funding) and the sleeve
mixed in at 30% with daily rebalancing:

  live   55% rotation, 45% book (config/comp.json)
  T(A)   30% rotation, 40% book (the backtester at a 3/7 rotation share) and 30% sleeve A
  C      30% rotation, 70% book, no sleeve (a control: the 30% to the book instead)

A sleeve earns the three-way split if T(A) beats live on the median 14-day composite in at
least 5 of 6 folds, with a worst drawdown at most 2 points deeper, and in both holdout years.
Return and drawdown are reported for the user's choice either way.

    python -m research.h95_aggressive_sleeve
"""
import copy
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research.folds import FOLDS, funding_table
from research.h50_long_short import blend
from research.h60_swing_mft import costs, curve_stats, rsi, simulate
from research.h61_swing import REGISTRY, Daily, atr_frac, engine, load
from research.holdout2018 import HOLDOUT

OUT = os.path.join("runs", "research", "h95")
LIVE = json.load(open("config/comp.json"))["strategy"]
CONFIGS = {
    "live": LIVE,
    "core 30/40": dict(LIVE, rotation_weight=3 / 7),
    "control 30/70": dict(LIVE, rotation_weight=0.30),
    "fast momentum": dict(LIVE, rotation_weight=0.99, rotation_top=2, rotation_horizons=[72, 168],
                          rotation_horizon_weights=[1.0, 1.0]),
}
SURVIVORS = ["SW27", "SW29", "SW32", "SW38", "SW39", "SW41", "SW45", "SW47", "SW51"]


def bot_curve(job):
    """The bot's hourly equity (from 1.0) for one fold with these strategy settings."""
    from bot.backtest import run_backtest
    from bot.config import apply_overrides, load_config
    from bot.market_data import HOUR_MS, BinanceClient, load_history
    from research.folds import candidates, ms
    tag, strategy, (start, end) = job
    path = os.path.join(OUT, "%s_%s.csv" % (tag.replace(" ", "_").replace("/", "-"), start))
    if not os.path.exists(path):
        cfg = load_config()
        apply_overrides(cfg, copy.deepcopy({"strategy": strategy}))
        s, e = ms(start), ms(end)
        client = BinanceClient()
        slip = candidates(cfg)
        bars = {p: load_history(client, p, s - cfg.backtest.warmup_bars * HOUR_MS, e, cfg.backtest.data_dir)
                for p in slip}
        bars = {p: b for p, b in bars.items() if b}
        r = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage, "taker",
                         monthly_universe=True, slippage_by_pair=slip, external_scores=funding_table(72))
        pd.DataFrame(r.curve, columns=["ts", "equity"]).to_csv(path, index=False)
    df = pd.read_csv(path)
    return tag, start, pd.Series(df["equity"].values / df["equity"].values[0],
                                 index=pd.to_datetime(df["ts"], unit="ms", utc=True))


def a2_breakouts(D, Y):
    hh, ll = D.h.rolling(336, min_periods=300).max(), D.l.rolling(336, min_periods=300).min()
    wr = -100 * (hh - D.c) / (hh - ll).replace(0, np.nan)
    entry = (wr > -20) & (wr.shift(1) <= -20)
    return engine(D, entry, stop=atr_frac(D, Y, 1.0), max_hours=336, exit_when=wr < -50)


def a3_dip_buys(D, Y):
    r3 = rsi(Y.c, 3)
    ema = Y.c.ewm(span=100, adjust=False, min_periods=100).mean()
    vol = D.vol.groupby(D.idx.normalize()).sum(min_count=1)
    vwap = Y.v.rolling(10).sum() / vol.rolling(10).sum()
    entry = Y.event((r3 < 22) & (ema > ema.shift(5)) & (Y.c <= vwap * 0.99))
    leave = Y.level(r3 > 55) > 0.5
    return engine(D, entry, exit_when=leave)


def hourly_net(sig, D, cost):
    g, f, _, _ = simulate(sig, D, cost)
    net = g - f
    net.index = net.index + pd.Timedelta(hours=1)               # by bar close, like the bot's curve
    return net


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    folds = FOLDS + HOLDOUT
    jobs = [(tag, st, f) for tag, st in CONFIGS.items() for f in folds]
    with ProcessPoolExecutor(max_workers=8) as pool:
        curves = {(tag, start): c for tag, start, c in pool.map(bot_curve, jobs)}
    print("bot curves ready", flush=True)
    D = load()
    Y = Daily(D)
    cost = costs(list(D.c.columns))
    nets = {}
    reg = {name.split()[0]: (fn, params) for name, fn, params, _ in REGISTRY}
    swing = [hourly_net(reg[k][0](D, Y, **reg[k][1]), D, cost) for k in SURVIVORS]
    nets["A1 swing ensemble"] = sum(swing) / len(swing)
    nets["A2 breakout swings"] = hourly_net(a2_breakouts(D, Y), D, cost)
    nets["A3 daily dip-buys"] = hourly_net(a3_dip_buys(D, Y), D, cost)
    print("sleeves ready", flush=True)

    def sleeve(name, start, end):
        if name == "A4 fast momentum":
            return curves[("fast momentum", start)]
        if name == "A5 all four":
            parts = [sleeve(n, start, end) for n in ("A1 swing ensemble", "A2 breakout swings",
                                                     "A3 daily dip-buys", "A4 fast momentum")]
            idx = parts[0].index
            for p in parts[1:]:
                idx = idx.intersection(p.index)
            r = sum(p.loc[idx].pct_change().fillna(0.0) for p in parts) / len(parts)
            return (1 + r).cumprod()
        sl = slice(pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=1), pd.Timestamp(end, tz="UTC"))
        return (1 + nets[name].loc[sl]).cumprod()

    names = ["A1 swing ensemble", "A2 breakout swings", "A3 daily dip-buys", "A4 fast momentum", "A5 all four"]
    rows = {}
    for start, end in folds:
        live = curves[("live", start)]
        core = curves[("core 30/40", start)]
        rows.setdefault("live 55/45", {})[start[:4]] = curve_stats(live)
        rows.setdefault("control 30/70", {})[start[:4]] = curve_stats(curves[("control 30/70", start)])
        for n in names:
            s = sleeve(n, start, end).reindex(core.index).ffill().fillna(1.0)
            rows.setdefault("sleeve " + n, {})[start[:4]] = curve_stats(s / s.iloc[0])
            rows.setdefault("30/30/40 with " + n, {})[start[:4]] = curve_stats(blend(core, s / s.iloc[0], 0.30))
    years = [s[:4] for s, _ in FOLDS]
    hy = [s[:4] for s, _ in HOLDOUT]
    base = rows["live 55/45"]
    bdd = max(base[y]["mdd"] for y in years)
    print("\nH95: yearly return (worst drawdown) by fold %s | six years | worst DD | median 14d composite (mean) | "
          "holdout %s" % (" ".join(years), " ".join(hy)))
    for n, by in rows.items():
        total = np.prod([1 + by[y]["ret"] for y in years]) - 1
        dd = max(by[y]["mdd"] for y in years)
        verdict = ""
        if n.startswith("30/30/40"):
            b14 = sum(by[y]["w14"] > base[y]["w14"] for y in years)
            h14 = sum(by[y]["w14"] > base[y]["w14"] for y in hy)
            ok = b14 >= 5 and dd <= bdd + 0.02 and h14 == len(hy)
            verdict = "| 14d better %d/6, holdout %d/2, DD %+.0f pts -> %s" % (b14, h14, (dd - bdd) * 100,
                                                                              "PASS" if ok else "fail")
        print("  %-36s %s | %+9.0f%% | %3.0f%% | %5.2f | %s %s" % (
            n, " ".join("%+6.0f%%(%2.0f)" % (by[y]["ret"] * 100, by[y]["mdd"] * 100) for y in years), total * 100,
            dd * 100, np.mean([by[y]["w14"] for y in years]),
            " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in hy), verdict), flush=True)


if __name__ == "__main__":
    main()
