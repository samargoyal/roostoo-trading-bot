"""H96: more aggressive sleeves (the user: "make something aggressive"), judged as H95: each at
30% beside a 30% rotation and a 40% book, against the live bot (55/45), on the same rule.
Written down before running:

  A6  all-in momentum: the single best coin on 3- and 7-day returns, re-picked daily, while
      BTC's filter is on (the bot's rotation code at 99%, one pick)
  A7  all-in K2: the single best coin on the live ranking (7, 14 and 21 days, 2/2/1)
  A8  meme momentum: the 2 meme coins with the best positive 7-day return, re-picked daily,
      while BTC's 168h EMA is above its 672h
  A9  a concentrated trend book: the live book alone, only its 5 strongest trends (long or
      short, round 92's option)
  A10 the sharpest efficiency book: the live book alone with the tilt cubed (round 98's power 3)
  A11 high-beta breakouts: H95's A2 (the 14-day Williams %R breakout) only in the most
      volatile third of the coins (30-day volatility, ranked that day)

    python -m research.h96_more_aggressive
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from research.folds import FOLDS
from research.h50_long_short import blend
from research.h60_swing_mft import MEMES_1H, costs, curve_stats
from research.h61_swing import Daily, atr_frac, engine, load
from research.h95_aggressive_sleeve import CONFIGS, LIVE, bot_curve, hourly_net
from research.holdout2018 import HOLDOUT

BOOK = dict(LIVE, rotation_weight=0.0)
MORE = {
    "A6 all-in momentum": dict(LIVE, rotation_weight=0.99, rotation_top=1, rotation_horizons=[72, 168],
                               rotation_horizon_weights=[1.0, 1.0]),
    "A7 all-in K2": dict(LIVE, rotation_weight=0.99, rotation_top=1),
    "A9 concentrated trend book": dict(BOOK, ls_top_n=5),
    "A10 sharpest efficiency book": dict(BOOK, ls_er_power=3.0),
}


def a8_memes(D):
    memes = [p for p in MEMES_1H if p in D.c.columns]
    day_end = D.idx.hour == 23
    r7 = np.log(D.c[memes] / D.c[memes].shift(168))
    btc = D.c["BTC/USD"]
    on = (btc.ewm(span=168, adjust=False).mean() > btc.ewm(span=672, adjust=False).mean())
    pick = pd.DataFrame(np.nan, index=D.idx, columns=D.c.columns)
    ranks = r7.where(D.ok[memes]).rank(axis=1, ascending=False)
    chosen = ((ranks <= 2) & (r7 > 0)).astype(float).mul(on.astype(float), axis=0)
    pick.loc[day_end, memes] = chosen.loc[day_end].values
    pick.loc[day_end] = pick.loc[day_end].fillna(0.0)
    return (pick.ffill().fillna(0.0) > 0.5) * 2.0        # each pick half the sleeve (the engine caps at 1/4 a unit)


def a11_high_beta(D, Y):
    vol = np.log(D.c).diff().rolling(720, min_periods=480).std()
    wild = vol.where(D.ok).rank(axis=1, pct=True) >= 2 / 3
    hh, ll = D.h.rolling(336, min_periods=300).max(), D.l.rolling(336, min_periods=300).min()
    wr = -100 * (hh - D.c) / (hh - ll).replace(0, np.nan)
    entry = (wr > -20) & (wr.shift(1) <= -20) & wild
    return engine(D, entry, stop=atr_frac(D, Y, 1.0), max_hours=336, exit_when=wr < -50)


def main() -> None:
    warnings.filterwarnings("ignore")
    folds = FOLDS + HOLDOUT
    jobs = [(tag, st, f) for tag, st in list(CONFIGS.items())[:2] + list(MORE.items()) for f in folds]
    with ProcessPoolExecutor(max_workers=8) as pool:
        curves = {(tag, start): c for tag, start, c in pool.map(bot_curve, jobs)}
    D = load()
    Y = Daily(D)
    cost = costs(list(D.c.columns))
    nets = {"A8 meme momentum": hourly_net(a8_memes(D), D, cost),
            "A11 high-beta breakouts": hourly_net(a11_high_beta(D, Y), D, cost)}
    names = list(MORE) + list(nets)
    rows = {}
    for start, end in folds:
        core = curves[("core 30/40", start)]
        rows.setdefault("live 55/45", {})[start[:4]] = curve_stats(curves[("live", start)])
        for n in names:
            if n in nets:
                sl = slice(pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=1), pd.Timestamp(end, tz="UTC"))
                s = (1 + nets[n].loc[sl]).cumprod()
            else:
                s = curves[(n, start)]
            s = s.reindex(core.index).ffill().fillna(1.0)
            s = s / s.iloc[0]
            rows.setdefault("sleeve " + n, {})[start[:4]] = curve_stats(s)
            rows.setdefault("30/30/40 with " + n, {})[start[:4]] = curve_stats(blend(core, s, 0.30))
    years = [s[:4] for s, _ in FOLDS]
    hy = [s[:4] for s, _ in HOLDOUT]
    base = rows["live 55/45"]
    bdd = max(base[y]["mdd"] for y in years)
    print("H96: yearly return (worst drawdown) by fold %s | six years | worst DD | median 14d composite (mean) | "
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
        print("  %-40s %s | %+10.0f%% | %3.0f%% | %5.2f | %s %s" % (
            n, " ".join("%+6.0f%%(%2.0f)" % (by[y]["ret"] * 100, by[y]["mdd"] * 100) for y in years), total * 100,
            dd * 100, np.mean([by[y]["w14"] for y in years]),
            " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in hy), verdict), flush=True)


if __name__ == "__main__":
    main()
