"""Round 78: the queue's last two session ideas (RESEARCH_QUEUE.md C3c and C4), on both bots.

Written down before running, each on top of the live bot and of the multi-horizon one, judged on
C1-C5 (research/queue.py):

  C3c the rotation ranks on returns without weekend hours, thin-volume moves that h76 found no
      more likely to reverse than weekday ones, so the prior is low (neighbours: without
      Saturdays only, without Sundays only)
  C4  session-decomposed momentum: candidates ranked on the normal score of their ranking value
      plus half that of their US-session return (13:00-21:00 UTC) less their Asian-session
      return (00:00-08:00) over 7 days: strength driven by US hours persists (neighbours: a
      quarter and a whole of the tilt); the same over 14 days; and both against it, the
      falsification arm. With 4 tries, C1-C5 and the neighbours are required of each

    python -m research.round78_session_momentum
"""
import warnings

from research.queue import judge


def tilt(weight, days):
    return (dict(session_tilt=weight, session_tilt_days=days),
            [dict(session_tilt=weight / 2, session_tilt_days=days), dict(session_tilt=weight * 2, session_tilt_days=days)])


DESIGNS = {
    "C3c weekend hours out of the ranking": (dict(rank_skip_days=[5, 6]),
                                             [dict(rank_skip_days=[5]), dict(rank_skip_days=[6])]),
    "C4 US-session strength, 7 days": tilt(0.5, 7),
    "C4 US-session strength, 14 days": tilt(0.5, 14),
    "C4 falsification: Asian strength, 7 days": tilt(-0.5, 7),
    "C4 falsification: Asian strength, 14 days": tilt(-0.5, 14),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 78, session momentum")


if __name__ == "__main__":
    main()
