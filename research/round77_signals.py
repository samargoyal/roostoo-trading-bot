"""Round 77: new signals (RESEARCH_QUEUE.md part E), on both bots.

Written down before running, each on top of the live bot and of the previous one, judged on
C1-C5 (research/queue.py):

  E1  beta hedge: short BTC against half the rotation's beta (336-hour betas), the short's
      collateral taken from the rotation, at most half of it (neighbours 0.25, 1.0). Rounds 7-8
      hedged the ranking and rounds 50-64 built long-short books; neither shorted BTC against
      the picks
  E2  macro filter: while BTC's 30-day correlation with QQQ is above 0.5 and QQQ is below its
      50-day average (about 13% of days), everything at half (neighbours: correlation 0.3,
      scale 0.75)
  E4  permutation entropy (order 4, 168 hours): rotation candidates only below the candidates'
      median entropy, cleaner trends (neighbours: orders 3 and 5); the opposite, above the
      median, is the falsification arm
  E5  capitulation: 2% of equity bought after an hourly -3 sd bar on 3 times its hour-of-week
      volume that closes 40% or more off its low, at most 3 at once, out at +1.5 ATRs or after
      24 hours, only while BTC's filter is on, where h76 found the next day positive in 5 of 6
      folds (neighbours 1%, 3%); while it is off, for information
  E3  trade dependence: a new rotation pick or trend-book leg is taken only if the coin's last
      paper trade (every signal, taken or not) lost; the opposite, only after a win, is the
      falsification arm (neighbours: the rotation alone, the book alone). Expect about half the
      entries, so a difference may be noise
  E6  meta-labelled sizing (research/h77_meta_labels.py, run first): each rotation pick at
      clip(2p, 0.5, 1) of its weight, p a walk-forward logistic model's probability that it beats
      its costs over 72 hours (neighbour: gradient-boosted trees of depth 3). Its walk-forward AUC
      was 0.48 to 0.55, so little is expected
  E7  funding against the coin's own norm: no long (rotation or book) while its funding is 1.5
      standard deviations above its 30-day mean (round 56 used an absolute level; neighbours
      1.0, 2.0)
  E8  volume acceleration: longs grow only while the second difference of 24-hour average
      dollar volume is positive (rounds 12 and 45-46 tested volume surfaces and TradingView's
      volume indicators, not this; neighbours 12, 48 hours)

    python -m research.round77_signals
"""
import warnings

from research.queue import judge

MACRO, FZ = {"macro": True}, {"funding_z": True}
DESIGNS = {
    "E1 beta hedge 0.5": (dict(beta_hedge=0.5), [dict(beta_hedge=0.25), dict(beta_hedge=1.0)]),
    "E2 macro filter": (dict(macro_rho=0.5, macro_scale=0.5, research=MACRO),
                        [dict(macro_rho=0.3, macro_scale=0.5, research=MACRO),
                         dict(macro_rho=0.5, macro_scale=0.75, research=MACRO)]),
    "E4 low permutation entropy": (dict(rotation_entropy_order=4),
                                   [dict(rotation_entropy_order=3), dict(rotation_entropy_order=5)]),
    "E4 falsification: high entropy": (dict(rotation_entropy_order=4, rotation_entropy_low=False),
                                       [dict(rotation_entropy_order=3, rotation_entropy_low=False),
                                        dict(rotation_entropy_order=5, rotation_entropy_low=False)]),
    "E5 capitulation 2%, BTC filter on": (dict(capitulation_size=0.02),
                                          [dict(capitulation_size=0.01), dict(capitulation_size=0.03)]),
    "E5 capitulation 2%, BTC filter off": (dict(capitulation_size=0.02, capitulation_regime="off"),
                                           [dict(capitulation_size=0.01, capitulation_regime="off"),
                                            dict(capitulation_size=0.03, capitulation_regime="off")]),
    "E3 entries after a losing signal only": (dict(trade_dependence="after_loss"),
                                              [dict(trade_dependence="after_loss", trade_dependence_scope="rotation"),
                                               dict(trade_dependence="after_loss", trade_dependence_scope="book")]),
    "E3 falsification: after a win only": (dict(trade_dependence="after_win"),
                                           [dict(trade_dependence="after_win", trade_dependence_scope="rotation"),
                                            dict(trade_dependence="after_win", trade_dependence_scope="book")]),
    "E6 meta-labelled sizing (logistic)": (dict(rotation_meta_sizing=True, research={"meta": "logistic"}),
                                           [dict(rotation_meta_sizing=True, research={"meta": "trees"}),
                                            dict(rotation_meta_sizing=True, research={"meta": "trees"})]),
    "E7 funding z below 1.5": (dict(long_max_funding_z=1.5, research=FZ),
                               [dict(long_max_funding_z=1.0, research=FZ), dict(long_max_funding_z=2.0, research=FZ)]),
    "E8 volume acceleration": (dict(volume_accel=True),
                               [dict(volume_accel=True, volume_accel_hours=12),
                                dict(volume_accel=True, volume_accel_hours=48)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 77, new signals")


if __name__ == "__main__":
    main()
