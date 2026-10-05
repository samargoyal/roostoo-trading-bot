"""H68a: securing profits within the competition's 14-day window.

The competition scores one 14-day window. Written down before running: run on every 14-day window
(each starting at a UTC day's close) of the live bot's hourly equity (R54b, the six folds and the
2018-2020 holdout years), with these window rules applied on top:

  lock 10%, half   once the window is up 10%, halve the exposure for the rest of the window
  lock 10%, cash   once up 10%, go to cash for the rest (lock it all)
  lock 20%, half   once up 20%, halve
  cut -10%, half   once down 10%, halve (cutting losses)
  lock+cut         the first and the fourth together

Halving costs 0.1% of the half sold. Scored as the competition does (0.4 Sortino + 0.3 Sharpe +
0.3 Calmar on the window), with the window's return, drawdown and share of winning windows.
A rule is worth adopting if it raises the median window composite in at least 5 of the 6
folds without cutting the mean window return by more than a tenth, and holds in the holdout.

    python -m research.h68_lock_in
"""
import os

import numpy as np
import pandas as pd

from bot.metrics import calmar, composite, max_drawdown, sharpe, sortino
from research.folds import FOLDS
from research.holdout2018 import HOLDOUT

DAY = 24
RULES = {
    "no rule (live bot)": dict(),
    "lock 10%, half": dict(lock=0.10, lock_scale=0.5),
    "lock 10%, cash": dict(lock=0.10, lock_scale=0.0),
    "lock 20%, half": dict(lock=0.20, lock_scale=0.5),
    "cut -10%, half": dict(cut=-0.10, cut_scale=0.5),
    "lock+cut": dict(lock=0.10, lock_scale=0.5, cut=-0.10, cut_scale=0.5),
}
COST = 0.001


def curve(start):
    df = pd.read_csv(os.path.join("runs", "research", "h60", "r54b_%s.csv" % start))
    return df["equity"].values


def window(rets, lock=None, lock_scale=1.0, cut=None, cut_scale=1.0):
    """Hourly returns of one window -> (return, max drawdown, composite) under the rule."""
    a, s, path = 1.0, 1.0, [1.0]
    for r in rets:
        a *= 1 + s * r
        if lock is not None and s == 1.0 and a - 1 >= lock:
            a *= 1 - COST * (1 - lock_scale)
            s = lock_scale
        if cut is not None and s == 1.0 and a - 1 <= cut:
            a *= 1 - COST * (1 - cut_scale)
            s = cut_scale
        path.append(a)
    daily = path[::DAY]
    if (len(path) - 1) % DAY:
        daily.append(path[-1])
    d = [b / x - 1 for x, b in zip(daily, daily[1:])]
    total = path[-1] - 1
    mdd = max_drawdown(path)
    return total, mdd, composite(sortino(d), sharpe(d), calmar(total, 14, mdd))


def fold_windows(eq, rule):
    hourly = eq[1:] / eq[:-1] - 1
    out = []
    for start in range(0, len(hourly) - 14 * DAY, DAY):
        out.append(window(hourly[start:start + 14 * DAY], **rule))
    return np.array(out)


def main() -> None:
    for label, folds in (("six folds", FOLDS), ("holdout", HOLDOUT)):
        print("\n" + label)
        base = {}
        for name, rule in RULES.items():
            rows, wins_comp = [], 0
            for start, _ in folds:
                w = fold_windows(curve(start), rule)
                med_comp, mean_ret = float(np.median(w[:, 2])), float(w[:, 0].mean())
                if name.startswith("no rule"):
                    base[start] = (med_comp, mean_ret)
                else:
                    wins_comp += med_comp > base[start][0]
                rows.append((start[:4], mean_ret, float(np.median(w[:, 0])), float((w[:, 0] > 0).mean()),
                             float(np.percentile(w[:, 0], 10)), float(w[:, 1].max()), med_comp))
            mean_all = np.mean([r[1] for r in rows])
            comp_all = np.median([r[6] for r in rows])
            verdict = "" if name.startswith("no rule") else " | median composite better in %d/%d folds" % (wins_comp, len(folds))
            print("  %-20s mean window %+.1f%%, median composite %.2f%s" % (name, mean_all * 100, comp_all, verdict))
            for y, mean_r, med_r, pos, p10, worst_dd, mc in rows:
                print("      %s mean %+5.1f%% median %+5.1f%% won %3.0f%% p10 %+6.1f%% worst DD %3.0f%% median comp %5.2f" % (
                    y, mean_r * 100, med_r * 100, pos * 100, p10 * 100, worst_dd * 100, mc))


if __name__ == "__main__":
    main()
