"""H72: trading the account's swings (the user: "we went from +1% to +3% three times").

Two days into the competition the account had swung between +1% and +3% three times, and the
user asked what selling near the top of each swing and buying back lower would have made. Swing
trading pays only if the account's moves revert. On the live bot's hourly equity (R54b, the six
folds and the two holdout years): first, what follows a move over the last 6 to 72 hours; then
the rule itself, run on that equity: sell once it is up from its 24-hour low, buy back after a
pullback from the sale, or if it keeps rising, so that a trend is not missed; 0.15% a side.

    python -m research.h72_swings
"""
import numpy as np
import pandas as pd

from bot.metrics import calmar, composite, max_drawdown, sharpe, sortino
from research.folds import FOLDS
from research.holdout2018 import HOLDOUT

COST = 0.0015            # a side: the 0.1% fee and about half the spread


def run(eq, up, back, stop, sell=1.0):
    """The rule on one year's hourly equity: (return, Sharpe, composite, trades)."""
    held, a, path = 1.0, 1.0, [1.0]
    sold_at, trades = None, 0
    for i in range(1, len(eq)):
        a *= 1 + held * (eq[i] / eq[i - 1] - 1)
        low = eq[max(0, i - 24):i + 1].min()
        if held == 1.0 and eq[i] >= low * (1 + up):
            held, sold_at = 1.0 - sell, eq[i]
            a *= 1 - COST * sell
            trades += 1
        elif held < 1.0 and (eq[i] <= sold_at * (1 - back) or eq[i] >= sold_at * (1 + stop)):
            held = 1.0
            a *= 1 - COST * sell
            trades += 1
        path.append(a)
    daily = path[::24]
    d = [y / x - 1 for x, y in zip(daily, daily[1:])]
    total, mdd = path[-1] - 1, max_drawdown(path)
    return total, sharpe(d), composite(sortino(d), sharpe(d), calmar(total, len(eq) / 24, mdd)), trades



def reversal(curves, years):
    print("what follows a move of the bot's equity over the last N hours (each year from 2018):")
    for back, ahead in ((6, 6), (12, 12), (24, 24), (24, 72), (72, 72)):
        cors, after_up = [], []
        for start in years:
            eq = curves[start]
            past = eq[back:-ahead] / eq[:-back - ahead] - 1
            fut = eq[back + ahead:] / eq[back:-ahead] - 1
            p, f = past[::6], fut[::6]                    # every 6 hours, to limit overlap
            cors.append(np.corrcoef(p, f)[0, 1])
            after_up.append(f[p >= 0.02 * (back / 24) ** 0.5].mean())
        print("  last %2dh -> next %2dh: correlation %s | next after a rise %s" % (
            back, ahead, " ".join("%+.2f" % x for x in cors), " ".join("%+.1f%%" % (x * 100) for x in after_up)))


def main() -> None:
    years = [s for s, _ in list(HOLDOUT) + list(FOLDS)]
    curves = {s: pd.read_csv("runs/research/h60/r54b_%s.csv" % s)["equity"].values for s in years}
    reversal(curves, years)
    print()
    rules = [("hold (the live bot)", None), ("sell all after +2%, back after -1.5% or +3%", (0.02, 0.015, 0.03, 1.0)),
             ("sell all after +3%, back after -2% or +4%", (0.03, 0.02, 0.04, 1.0)),
             ("sell all after +1.5%, back after -1% or +2%", (0.015, 0.01, 0.02, 1.0)),
             ("sell half after +2%, back after -1.5% or +3%", (0.02, 0.015, 0.03, 0.5))]
    print("year:" + "".join("%11s" % s[:4] for s in years))
    base = {}
    for name, rule in rules:
        out = {s: run(curves[s], *rule) if rule else (curves[s][-1] / curves[s][0] - 1, None, None, 0) for s in years}
        if rule is None:
            base = {s: run(curves[s], 1.0, 0, 0, 0.0) for s in years}      # no trades: the bot itself
            out = base
        better = sum(out[s][0] > base[s][0] for s in years)
        sh = sum(out[s][1] > base[s][1] for s in years)
        print("%-46s %s | return better %d/8, Sharpe better %d/8, trades a year %s" % (
            name, " ".join("%+9.0f%%" % (out[s][0] * 100) for s in years), better, sh,
            "-" if rule is None else "%.0f" % np.mean([out[s][3] for s in years])))


if __name__ == "__main__":
    main()
