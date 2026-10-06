"""Round 76: session filters (RESEARCH_QUEUE.md part C), on both bots.

The queue's survey rates session seasonality as weak evidence, and h76_measurements found no
consistent weekend reversal (the weekend's return against Monday-Tuesday's: correlations from
-0.45 to +0.21 by fold), so the prior is low. Written down before running:

  C1  the rotation's daily re-pick at each UTC hour 0-23 (live: 00:00). The hour changes only
      if a block of 4 or more hours in a row all beat 00:00 on the median 14-day composite in
      5 of 6 folds: a single best hour among 24 is noise
  C2  new entries and adds only in the US session (13:00-21:00 UTC; neighbours 12-22, 14-20),
      or in Europe's and the US's (08:00-21:00; neighbours 07-22, 09-20); exits at any hour
  C3a no new entries or adds at weekends (neighbours: Saturdays only, Sundays only, by hour set)
  C3b everything at 0.7 at weekends (neighbours 0.6, 0.8)
  C6  no new entries in the hours funding settles (00:00, 08:00, 16:00), run only if C1 shows
      those hours worse: reported from C1 for information

Judged on C1-C5 (research/queue.py) except C1's hour, judged by its own guard.

    python -m research.round76_sessions
"""
import warnings

import numpy as np

from research.queue import BOTS, YEARS, better_14d, design, judge
from research.rounds import results_for

US, EU_US = list(range(13, 21)), list(range(8, 21))
DESIGNS = {
    "C2 entries in the US session": (dict(entry_hours=US),
                                     [dict(entry_hours=list(range(12, 22))), dict(entry_hours=list(range(14, 20)))]),
    "C2 entries in Europe's and the US's": (dict(entry_hours=EU_US),
                                            [dict(entry_hours=list(range(7, 22))), dict(entry_hours=list(range(9, 20)))]),
    "C3a no weekend entries": (dict(weekend_no_entries=True),
                               [dict(weekend_no_entries=True, entry_hours=list(range(12, 24))),
                                dict(weekend_no_entries=True, entry_hours=list(range(0, 12)))]),
    "C3b weekends at 0.7": (dict(weekend_scale=0.7), [dict(weekend_scale=0.6), dict(weekend_scale=0.8)]),
}


def rebalance_hours() -> None:
    for bot in BOTS:
        names = {"h%02d" % h: design(bot, rotation_rebalance_offset=h) for h in range(1, 24)}
        names["h00"] = design(bot)
        res = results_for(names)
        base = res["h00"]
        wins = {h: better_14d(res["h%02d" % h], base) for h in range(1, 24)}
        print("\nC1 on the %s bot: folds (of 6) in which re-picking at each hour beats 00:00 on the 14-day composite" % bot)
        print("  " + " ".join("%02d:%d" % (h, wins[h]) for h in range(1, 24)))
        print("  median 14-day composite, mean over folds: " + " ".join(
            "%02d:%.1f" % (h, np.mean([res["h%02d" % h][y]["w14_comp"] for y in YEARS])) for h in range(24)))
        good = [h for h in range(1, 24) if wins[h] >= 5]
        blocks, run = [], []
        for h in range(1, 24):
            run = run + [h] if h in good else []
            if len(run) >= 4:
                blocks.append(list(run))
        print("  blocks of 4+ hours each better in 5/6: %s" % (", ".join("%02d-%02d" % (b[0], b[-1]) for b in blocks) or "none"))
        print("  C6 (funding hours): 08:00 beats 00:00 in %d/6, 16:00 in %d/6" % (wins[8], wins[16]))


def main() -> None:
    warnings.filterwarnings("ignore")
    rebalance_hours()
    judge(DESIGNS, "Round 76, session filters")


if __name__ == "__main__":
    main()
