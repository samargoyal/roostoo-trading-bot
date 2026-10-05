"""Round 72: the long-short book's entries and exits, longs and shorts (the user: "make entry and
exit for both long and short").

Rounds 63 and 64 tested four on the live bot (R54b): a funding margin for new shorts, no new
short while oversold, trailing stops on longs, and shorts on a faster trend; their results are
shown again here (from the fold cache). Written down before running, four more on top of the
live bot, judged as rounds 65-71 were:

  R72b turtle shorts: a coin is shorted only on a close below its 20-day low, and covered above
       its 10-day high (Dennis and Eckhardt's breakout rules; neighbours 10/5 and 30/15 days)
  R72c a tighter stop on shorts: covered 6 ATRs above the lowest close since entry instead of 10
       (neighbours 5, 8)
  R72d shorts only in a confirmed bear market: BTC's filter off and BTC below its 90-day average
       (neighbours 70 and 100 days)
  R72e size by trend strength, both sides: a coin's position grows with the gap between its
       EMAs, full at 5%, so weak trends hold less (neighbours 3%, 8%)

    python -m research.round72_book_entries_exits
"""
import warnings

import numpy as np

from research.folds import FOLDS
from research.holdout2018 import HOLDOUT
from research.round66_attention import design
from research.round68_profits import plain
from research.rounds import results_for, verdict

EARLIER = {
    "R63a new shorts need funding 0.005%": dict(short_entry_min_funding=0.00005),
    "R63b no new short while oversold": dict(short_entry_rsi_min=30.0),
    "R64a trailing stops on longs": dict(ls_long_stop_atr=8.0),
    "R64b shorts on a faster trend": dict(ls_short_trend=[168, 672]),
}
DESIGNS = {
    "R72b turtle shorts (20/10 days)": (dict(short_entry_channel=480, short_exit_channel=240),
                                        [dict(short_entry_channel=240, short_exit_channel=120),
                                         dict(short_entry_channel=720, short_exit_channel=360)]),
    "R72c short stop 6 ATRs": (dict(short_stop_atr=6.0), [dict(short_stop_atr=5.0), dict(short_stop_atr=8.0)]),
    "R72d shorts only in a bear market": (dict(short_regime_hours=2160),
                                          [dict(short_regime_hours=1680), dict(short_regime_hours=2400)]),
    "R72e size by trend strength": (dict(ls_full_gap=0.05), [dict(ls_full_gap=0.03), dict(ls_full_gap=0.08)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"live bot (R54b)": design()}
    names.update({"(earlier) " + n: plain(**o) for n, o in EARLIER.items()})
    names.update({n: plain(**o) for n, (o, _) in DESIGNS.items()})
    res = results_for(names)
    years = [f[0][:4] for f in FOLDS]
    base = res["live bot (R54b)"]
    for name, by in res.items():
        ok, med, better, worst = verdict(by, base, paired=True)
        six = np.prod([1 + by[y]["ret"] for y in years]) - 1
        w14 = sum(by[y]["w14_comp"] > base[y]["w14_comp"] for y in years)
        print("%-44s %s | 6y %+.0f%% DD %.0f%% | yearly better %d/6, 14-day better %d/6%s" % (
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
            print("    %-48s better %d/6 -> %s" % (n, better, "holds" if nok else "breaks"))
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
