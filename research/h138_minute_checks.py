"""H138: B1 checked every minute (the user: stops and the rotation's exit every minute, not hourly).

The live bot acts once an hour. With 30 calls a minute it could read every price each minute, so:
  stops: B1's stops (10 ATRs and none on longs; with 8-ATR long stops as a variant) checked on each
      minute's close instead of the hour's close. Simulated on hourly bars as they are (so the ATR
      is unchanged) with each hour's lowest and highest minute close from 1-minute data
      (research/klines1m.py) as the levels a minute check would see (bot.research_rules.
      INTRABAR_EXTREMES), filled at the stop level; against the hourly close (live) and every wick
      (H121). October 2025 to 11 October 2026, the competition week reported apart.
  the rotation's exit: BTC's 168-hour EMA crossing below its 672-hour EMA, seen within the hour
      instead of at its close. For each such hour from 2018, BTC's move from the minute the cross
      would first show (on the minute's close, the EMAs updated with it) to the hour's close is what
      checking every minute would have saved or cost the rotation; only 2025-26 has minute data.

    python -m research.h138_minute_checks
"""
import json
import os
import tempfile

import pandas as pd

from bot import research_rules
from bot.backtest import main as backtest

S = tempfile.mkdtemp()


def minute_closes(symbol):
    parts = [pd.read_parquet(p)[["ts", "close"]] for p in (os.path.join("data", "binance_1m", symbol + ".parquet"),
                                                           os.path.join("data", "binance_1m_recent", symbol + ".parquet"))
             if os.path.exists(p)]
    return pd.concat(parts).drop_duplicates("ts").sort_values("ts") if parts else None


def load_extremes():
    for f in os.listdir(os.path.join("data", "binance_1m")):
        sym = f.replace(".parquet", "")
        m = minute_closes(sym)
        m["hour"] = m["ts"] // 3_600_000 * 3_600_000
        g = m.groupby("hour")["close"].agg(["min", "max"])
        research_rules.INTRABAR_EXTREMES[sym.replace("USDT", "/USD")] = dict(zip(g.index, zip(g["min"], g["max"])))


def run(name, extra):
    cfg = json.load(open(os.path.join("config", "comp_b1_60.json")))
    cfg["strategy"].update(extra)
    path = os.path.join(S, name + ".json")
    json.dump(cfg, open(path, "w"))
    out = os.path.join("runs", "research", "h138", name)
    backtest(["--config", path, "--start", "2025-10-01", "--end", "2026-10-12", "--out", out])
    e = pd.read_csv(os.path.join(out, "taker", "equity.csv"))
    e["time"] = pd.to_datetime(e["time"])
    s = e.set_index("time")["equity"]
    t = pd.read_csv(os.path.join(out, "taker", "trades.csv"))
    c = s[s.index >= "2026-10-04 13:00"]
    return "year %+6.1f%%  worst drawdown %4.1f%%  stop exits %4d | competition week %+5.1f%%" % (
        100 * (s.iloc[-1] / s.iloc[0] - 1), 100 * (1 - (s / s.cummax()).min()), t["reason"].str.contains("stop").sum(),
        100 * (c.iloc[-1] / c.iloc[0] - 1))


def filter_flips():
    btc = pd.read_csv(os.path.join("data", "binance", "BTCUSDT_1h.csv")).drop_duplicates("ts").sort_values("ts")
    c = btc.set_index("ts")["close"]
    fast, slow = c.ewm(span=168, adjust=False).mean(), c.ewm(span=672, adjust=False).mean()
    on = fast > slow
    offs = on.index[(on.shift(1) == True) & (on == False)]           # noqa: E712 - the hour it turned off
    ons = on.index[(on.shift(1) == False) & (on == True)]            # noqa: E712
    years = (c.index[-1] - c.index[0]) / (365 * 86_400_000)
    print("BTC's rotation filter: %d turns off and %d on in %.1f years (%.1f a year)" % (
        len(offs), len(ons), years, (len(offs) + len(ons)) / years))
    m = minute_closes("BTCUSDT").set_index("ts")["close"]
    a_f, a_s = 2 / 169, 2 / 673
    saved = []
    for kind, hours in (("off", offs), ("on", ons)):
        for h in hours:
            if h < m.index[0] + 3_600_000:
                continue
            prev_f, prev_s = fast.loc[h - 3_600_000], slow.loc[h - 3_600_000]
            mins = m[(m.index >= h) & (m.index < h + 3_600_000)]
            for ts, px in mins.items():
                f, s_ = prev_f + a_f * (px - prev_f), prev_s + a_s * (px - prev_s)
                if (f < s_) if kind == "off" else (f > s_):
                    move = c.loc[h] / px - 1                         # from that minute to the hour's close
                    saved.append((kind, pd.to_datetime(h, unit="ms"), (ts - h) // 60_000, move))
                    break
    if saved:
        print("  minute-checked flips since September 2025 (rotation's 60% acting that much sooner):")
        for kind, when, minute, move in saved:
            effect = -move if kind == "off" else move                  # leaving earlier saves a fall
            print("    %s turned %-3s at minute %2d; BTC to the hour's close %+.2f%% -> minute checks %+.2f%% on the rotation"
                  % (when, kind, minute, 100 * move, 100 * 0.6 * effect))


def main() -> None:
    load_extremes()
    print("B1 60/40, October 2025 to 11 October 2026 (taker fees), stops checked:")
    print("  on the hour's close (live)          %s" % run("hourly", {}))
    print("  every minute (minute closes)        %s" % run("minute", {"intrabar_stops": True}))
    print("  with 8-ATR long stops, hourly       %s" % run("hourly_l8", {"ls_long_stop_atr": 8.0}))
    print("  with 8-ATR long stops, every minute %s" % run("minute_l8", {"intrabar_stops": True, "ls_long_stop_atr": 8.0}))
    research_rules.INTRABAR_EXTREMES.clear()
    print("  every wick (H121)                   %s" % run("wick", {"intrabar_stops": True}))
    filter_flips()


if __name__ == "__main__":
    main()
