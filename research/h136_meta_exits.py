"""H136: H135's meta-labeled dip buys with wider exits, so the model's picks can run further.

H135's model, trained on development only, sorted its candidates well: at its cutoff (0.67) the
validation trades won 63.4% with wins 1.41 times losses, about one a day, but the holdout's ratio
was 0.98. This re-runs it unchanged except for the exit, fixed before running: targets of 2, 2.5
and 3 daily moves (the stop still 2), held 12 or 24 hours; the cutoff found the same way
(development out-of-fold, the lowest probability whose trades win 60% or more).

    python -m research.h136_meta_exits
"""
import warnings

from research import h135_meta_labels as m


def main() -> None:
    warnings.filterwarnings("ignore")
    m.WIN_CUT = 0.60
    for target in (2.0, 2.5, 3.0):
        for hold in (12, 24):
            m.TARGET, m.HOLD = target, hold
            print("\n==== target %.1f, stop %.1f daily moves, held %dh" % (target, m.STOP, hold), flush=True)
            m.main()


if __name__ == "__main__":
    main()
