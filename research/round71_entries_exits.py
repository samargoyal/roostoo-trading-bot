"""Round 71: the rotation's entries and exits, re-tested on the live bot.

The user asked for better entry and exit points. Most of these rules were tested in rounds 20-48,
but on the bot of that time (a 40-50% rotation beside the defensive book); the live bot is R54b
(70% rotation, 30% long-short book), so they are tested again on it. Written down before
running, each on top of the live bot and judged as rounds 65-70 were (paired gains in at least 5
of 6 folds, worst drawdown at most 2 points deeper, both neighbours in 4 of 6, then both holdout
years), with the 14-day windows shown beside:

  exits
  R71a ATR trailing stop: a pick that closes 8 hourly ATRs below its high since picked leaves,
       barred for a day (neighbours 6, 12)
  R71b percentage trailing stop: the same at 15% below the high (neighbours 10%, 20%)
  R71c channel exit: a pick that closes below its 72-hour low leaves (neighbours 48, 120 hours)
  R71d rank buffer: a held pick stays while it ranks in the top 3, not only the top 2, so a
       leader is not sold on a small slip (neighbours top 4, top 5)
  R71l the long-short book's neutral band: no position while a coin's EMAs are within 1% of
       each other, where the trend is unclear (neighbours 0.5%, 2%)
  entries
  R71e breakout entry: a pick is bought only at its 72-hour high (neighbours 48, 120 hours)
  R71f own uptrend: a pick must close above its slow EMA with its fast EMA above the slow one
       (neighbours: the close alone, the EMAs alone)
  R71g not overextended: skip a pick more than 3 standard deviations above its 168-hour mean
       (neighbours 2.5, 3.5)
  R71h steady climb: a pick's 14-day log return over its volatility must be at least 1, so a
       single spike does not qualify (neighbours 0.75, 1.5)
  R71i multi-horizon ranking: candidates ranked on their 7-, 14- and 21-day returns together
       (neighbours 7+14, 14+21+30 days)
  R71j re-entry delay: after BTC's filter turns on, wait 24 hours before buying, to skip false
       starts (neighbours 12, 48)
  R71k filter hysteresis: BTC's 7-day EMA must clear its 28-day EMA by 1% to turn the filter on or
       off (neighbours 0.5%, 2%)

    python -m research.round71_entries_exits
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round66_attention import design
from research.round68_profits import plain
from research.rounds import results_for, verdict

DESIGNS = {
    "R71a ATR trailing stop": (dict(rotation_stop_atr=8.0), [dict(rotation_stop_atr=6.0), dict(rotation_stop_atr=12.0)]),
    "R71b 15% trailing stop": (dict(rotation_stop_pct=0.15), [dict(rotation_stop_pct=0.10), dict(rotation_stop_pct=0.20)]),
    "R71c 72-hour channel exit": (dict(rotation_donchian_exit=72),
                                  [dict(rotation_donchian_exit=48), dict(rotation_donchian_exit=120)]),
    "R71d rank buffer (top 3)": (dict(rotation_buffer=3), [dict(rotation_buffer=4), dict(rotation_buffer=5)]),
    "R71l book neutral band 1%": (dict(ls_band=0.01), [dict(ls_band=0.005), dict(ls_band=0.02)]),
    "R71e 72-hour breakout entry": (dict(rotation_donchian_entry=72),
                                    [dict(rotation_donchian_entry=48), dict(rotation_donchian_entry=120)]),
    "R71f own uptrend": (dict(rotation_pick_trend="both"),
                         [dict(rotation_pick_trend="close"), dict(rotation_pick_trend="ema")]),
    "R71g not overextended (3 sd)": (dict(rotation_max_z=3.0), [dict(rotation_max_z=2.5), dict(rotation_max_z=3.5)]),
    "R71h steady climb (t >= 1)": (dict(rotation_min_tstat=1.0),
                                   [dict(rotation_min_tstat=0.75), dict(rotation_min_tstat=1.5)]),
    "R71i multi-horizon ranking": (dict(rotation_ranking="multi", rotation_horizons=[168, 336, 504]),
                                   [dict(rotation_ranking="multi", rotation_horizons=[168, 336]),
                                    dict(rotation_ranking="multi", rotation_horizons=[336, 504, 720])]),
    "R71j re-entry delay 24h": (dict(rotation_reentry_hours=24),
                                [dict(rotation_reentry_hours=12), dict(rotation_reentry_hours=48)]),
    "R71k filter hysteresis 1%": (dict(rotation_filter_band=0.01),
                                  [dict(rotation_filter_band=0.005), dict(rotation_filter_band=0.02)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"live bot (R54b)": design()}
    names.update({n: plain(**o) for n, (o, _) in DESIGNS.items()})
    res = results_for(names)
    years = [f[0][:4] for f in FOLDS]
    base = res["live bot (R54b)"]
    for name, by in res.items():
        ok, med, better, worst = verdict(by, base, paired=True)
        six = np.prod([1 + by[y]["ret"] for y in years]) - 1
        w14 = sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in years)
        print("%-30s %s | 6y %+.0f%% DD %.0f%% | yearly better %d/6, 14-day better %d/6%s" % (
            name, " ".join("%+5.0f%%" % (by[y]["ret"] * 100) for y in years), six * 100, worst * 100, better, w14,
            "" if name.startswith("live") else ("  -> candidate" if ok else "  -> fail")), flush=True)
    for name, (opts, neighbours) in DESIGNS.items():
        ok, _, _, _ = verdict(res[name], base, paired=True)
        if not ok:
            continue
        nres = results_for({"%s / neighbour %d" % (name, i + 1): plain(**o) for i, o in enumerate(neighbours)})
        fine = []
        for n, by in nres.items():
            nok, med, better, worst = verdict(by, base, strict=False, paired=True)
            fine.append(nok)
            print("    %-44s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
        if not all(fine):
            print("  %s: not robust" % name)
            continue
        h = results_for({"live": design(), name: plain(**opts)}, HOLDOUT)
        hy = sorted(h["live"])
        wins = sum(h[name][y]["comp"] > h["live"][y]["comp"] for y in hy)
        print("  holdout: live %s, design %s -> %s" % (
            " ".join("%+.0f%%" % (h["live"][y]["ret"] * 100) for y in hy),
            " ".join("%+.0f%%" % (h[name][y]["ret"] * 100) for y in hy), "CONFIRMED" if wins == len(hy) else "not confirmed"))


if __name__ == "__main__":
    main()
