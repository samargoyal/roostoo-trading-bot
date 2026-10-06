"""Round 75: risk circuit breakers (RESEARCH_QUEUE.md part B), on both bots.

The account holds about 70% in two rotation picks, so a breaker that cuts risk at the right time
could matter more than any new signal; but each one that cuts risk also cuts return, and the
rotation brake (round 23) and the profit locks (rounds 68-70) failed for that reason. Written
down before running, each on top of the live bot and of the previous one, judged on C1-C5
(research/queue.py). "No new entries" lets a position shrink or close but not grow.

  B1  daily loss limit: 3% below the account's 00:00 UTC value, no new entries until the next
      00:00; at 5% everything at half (neighbours 2%/5%, 4%/5%). B1b: 3% without the second tier
  B2  drawdown ladder: 4%, 7% and 10% below the account's peak scale everything to 0.5, then
      0.25, then the rotation goes to cash for 24 hours; each step released at half its level
      after 12 hours (neighbours 3/5/8%, 6/10/14%)
  B3  position loss: a position whose last 24 hours cost the account 2.5% of equity is halved
      and not added to for a day (neighbours 1.5%, 4%)
  B5  short squeeze guard: a short whose coin rose 10% in 24 hours or 3 ATRs in an hour is
      covered and not shorted again for 48 hours; shorts capped at 15% of equity (neighbours
      7%/2.5 ATRs, 15%/4 ATRs)
  B6  BTC shock: BTC down 2.5% in an hour or 5% in four: no new longs for 6 hours and the
      rotation at half (neighbours 2%/4%, 3.5%/7%)
  B7  volatility regime: BTC's 24-hour realised variance 2.5 times its 30-day median scales
      everything by 2.5 over the ratio, at least 0.4 (round 19 scaled continuously by a
      forecast; neighbours 2.0, 3.0)

    python -m research.round75_risk_breakers
"""
import warnings

from research.queue import judge

DESIGNS = {
    "B1 daily loss 3% (5% halves)": (dict(day_loss_stop=0.03, day_loss_cut=0.05),
                                     [dict(day_loss_stop=0.02, day_loss_cut=0.05),
                                      dict(day_loss_stop=0.04, day_loss_cut=0.05)]),
    "B1b daily loss 3% only": (dict(day_loss_stop=0.03),
                               [dict(day_loss_stop=0.02), dict(day_loss_stop=0.04)]),
    "B2 drawdown ladder 4/7/10%": (dict(dd_ladder=[0.04, 0.07, 0.10]),
                                   [dict(dd_ladder=[0.03, 0.05, 0.08]), dict(dd_ladder=[0.06, 0.10, 0.14])]),
    "B3 position loss 2.5% of equity": (dict(coin_loss_cap=0.025),
                                        [dict(coin_loss_cap=0.015), dict(coin_loss_cap=0.04)]),
    "B5 short squeeze guard": (dict(squeeze_rise=0.10, squeeze_atr=3.0),
                               [dict(squeeze_rise=0.07, squeeze_atr=2.5), dict(squeeze_rise=0.15, squeeze_atr=4.0)]),
    "B6 BTC shock 2.5%/5%": (dict(btc_shock_1h=0.025, btc_shock_4h=0.05),
                             [dict(btc_shock_1h=0.02, btc_shock_4h=0.04), dict(btc_shock_1h=0.035, btc_shock_4h=0.07)]),
    "B7 volatility regime 2.5x": (dict(vol_regime=2.5), [dict(vol_regime=2.0), dict(vol_regime=3.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 75, risk circuit breakers")


if __name__ == "__main__":
    main()
