"""Round 107: a crash detector that switches the account to the book alone and back (the user:
"can you make a detector which detects this and switches automatically?").

The live bot runs 55% rotation and 45% efficiency book. Each detector, while it says "bear",
gives the rotation's share to the book (the book alone, which made +26% in the 2021-22 crash
year), and hands it back when it clears. Written down before running, on the live bot (55/45),
reported against the fixed 55/45 and the book alone on return, drawdown, the crash year, the
14-day yardstick and the holdout:

  B1  the rotation's own BTC filter (168h EMA below 672h): when it turns off, the rotation's
      share runs the book instead of PAXG or cash (round 61's option)
  B2  B1, but only while BTC is also below its 200-day average (round 62's confirmation)
  B3  a fast BTC detector: BTC below its 50h and 200h EMAs with the 50h below the 200h
  B4  breadth: under 40% of the coins up over 72 hours
  B5  B3 and B4 together

B3-B5 use round 106's option with the shares 55/55/0 (bull, neutral, bear).

    python -m research.round107_crash_detector
"""
import warnings

import numpy as np

from research.holdout2018 import HOLDOUT
from research.queue import BOTS, YEARS, design
from research.rounds import results_for

BOT = "live (55/45, efficiency book)"
BOTS[BOT] = __import__("json").load(open("config/comp_er_55.json"))["strategy"]
DESIGNS = {
    "book alone": dict(rotation_weight=0.0),
    "B1 BTC filter off -> book": dict(ls_absorb_rotation=1.0),
    "B2 B1 + below 200-day average": dict(ls_absorb_rotation=1.0, ls_absorb_sma_hours=4800),
    "B3 fast BTC (50h/200h) -> book": dict(split_regime="btc", split_shares=[0.55, 0.55, 0.0]),
    "B4 breadth under 40% -> book": dict(split_regime="breadth", split_shares=[0.55, 0.55, 0.0]),
    "B5 both -> book": dict(split_regime="both", split_shares=[0.55, 0.55, 0.0]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    names = {"fixed 55/45": design(BOT)}
    for n, o in DESIGNS.items():
        d = design(BOT, **o)
        if "ls_absorb_sma_hours" in o:
            d["backtest"] = {"warmup_bars": 6500}
        names[n] = d
    res, hold = results_for(names), results_for(names, HOLDOUT)
    hy = sorted(hold["fixed 55/45"])
    b = res["fixed 55/45"]
    print("Round 107: yearly return by fold %s | total | worst DD | 14d better | won | holdout %s [14d]"
          % (" ".join(YEARS), " ".join(hy)))
    for n, by in res.items():
        print("  %-32s %s | %+8.0f%% | %3.0f%% | %d/6 | %3.0f%% | %s [%s]" % (
            n, " ".join("%+6.0f%%" % (by[y]["ret"] * 100) for y in YEARS),
            (np.prod([1 + by[y]["ret"] for y in YEARS]) - 1) * 100, max(by[y]["mdd"] for y in YEARS) * 100,
            sum(by[y]["w14_comp"] > b[y]["w14_comp"] for y in YEARS),
            np.mean([by[y]["w14_pos"] for y in YEARS]) * 100,
            " ".join("%+5.0f%%" % (hold[n][y]["ret"] * 100) for y in hy),
            " ".join("%.2f" % hold[n][y]["w14_comp"] for y in hy)), flush=True)


if __name__ == "__main__":
    main()
