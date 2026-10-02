"""Performance statistics, including the competition's composite score.

Sharpe, Sortino and volatility use daily returns (UTC days, 365 per year, zero risk-free
rate). Calmar is annualised return over maximum drawdown, with the drawdown measured on
hourly equity. The organisers have not published their exact conventions.
"""
import math
from typing import Dict, List, Sequence, Tuple

DAY_MS = 24 * 3600 * 1000
COMPETITION_DAYS = 14


def max_drawdown(values: Sequence[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, 1.0 - value / peak)
    return worst


def daily_closes(curve: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Last equity value of each UTC day, as (day number, equity)."""
    by_day: Dict[int, float] = {}
    for ts, equity in curve:
        by_day[ts // DAY_MS] = equity
    return sorted(by_day.items())


def returns(values: Sequence[float]) -> List[float]:
    return [b / a - 1.0 for a, b in zip(values, values[1:]) if a > 0]


def sharpe(rets: Sequence[float], periods_per_year: float = 365.0) -> float:
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return mean / math.sqrt(var) * math.sqrt(periods_per_year) if var > 0 else 0.0


def sortino(rets: Sequence[float], periods_per_year: float = 365.0) -> float:
    if not rets:
        return 0.0
    mean = sum(rets) / len(rets)
    downside = math.sqrt(sum(min(r, 0.0) ** 2 for r in rets) / len(rets))
    return mean / downside * math.sqrt(periods_per_year) if downside > 0 else 0.0


def calmar(total_return: float, days: float, drawdown: float) -> float:
    if days <= 0 or drawdown <= 0 or total_return <= -1:
        return 0.0
    annual = (1.0 + total_return) ** (365.0 / days) - 1.0
    return annual / drawdown


def composite(sortino_ratio: float, sharpe_ratio: float, calmar_ratio: float) -> float:
    """The competition's score: 0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar."""
    return 0.4 * sortino_ratio + 0.3 * sharpe_ratio + 0.3 * calmar_ratio


def summarize(curve: Sequence[Tuple[int, float]], initial: float,
              trades: Sequence[Tuple[int, float]] = ()) -> Dict[str, float]:
    """Statistics for an hourly equity curve [(ts, equity)] and trades [(ts, notional)]."""
    if not curve:
        return {}
    values = [initial] + [v for _, v in curve]
    days = (curve[-1][0] - curve[0][0]) / DAY_MS + 1.0 / 24
    total = values[-1] / initial - 1.0
    daily = [initial] + [v for _, v in daily_closes(curve)]
    day_rets = returns(daily)
    mdd = max_drawdown(values)
    stats = {
        "days": days,
        "total_return": total,
        "annual_return": (1.0 + total) ** (365.0 / days) - 1.0 if total > -1 else -1.0,
        "max_drawdown": mdd,
        "daily_volatility": _stdev(day_rets),
        "sharpe": sharpe(day_rets),
        "sortino": sortino(day_rets),
        "calmar": calmar(total, days, mdd),
    }
    stats["composite"] = composite(stats["sortino"], stats["sharpe"], stats["calmar"])

    mean_equity = sum(values) / len(values)
    # Activity counts only days the curve covers in full (24 hourly points).
    points: Dict[int, int] = {}
    for ts, _ in curve:
        points[ts // DAY_MS] = points.get(ts // DAY_MS, 0) + 1
    per_day = {d: 0 for d, n in points.items() if n >= 24}
    for ts, _ in trades:
        if ts // DAY_MS in per_day:
            per_day[ts // DAY_MS] += 1
    stats.update({
        "trades": float(len(trades)),
        "trades_per_day": len(trades) / days,
        "min_trades_in_a_day": float(min(per_day.values())) if per_day else 0.0,
        "active_day_share": sum(1 for n in per_day.values() if n) / len(per_day) if per_day else 0.0,
        "turnover_per_day": sum(n for _, n in trades) / mean_equity / days if mean_equity > 0 else 0.0,
    })
    stats.update(window_stats(curve))
    return stats


def window_stats(curve: Sequence[Tuple[int, float]],
                 length_days: int = COMPETITION_DAYS) -> Dict[str, float]:
    """Return, drawdown and composite score over every rolling window the length of the competition.

    A window starts at the close of one UTC day and covers the next `length_days` full days.
    """
    daily = daily_closes(curve)
    hourly_by_day: Dict[int, List[float]] = {}
    for ts, value in curve:
        hourly_by_day.setdefault(ts // DAY_MS, []).append(value)
    days = [d for d, _ in daily]
    closes = [v for _, v in daily]

    rets, mdds, scores = [], [], []
    for i in range(len(daily) - length_days):
        if days[i + length_days] - days[i] != length_days:
            continue  # a gap in the data
        window_closes = closes[i:i + length_days + 1]
        path = [closes[i]] + [v for d in days[i + 1:i + length_days + 1] for v in hourly_by_day[d]]
        total = window_closes[-1] / window_closes[0] - 1.0
        mdd = max_drawdown(path)
        day_rets = returns(window_closes)
        rets.append(total)
        mdds.append(mdd)
        scores.append(composite(sortino(day_rets), sharpe(day_rets), calmar(total, length_days, mdd)))
    if not rets:
        return {}
    return {
        "window_count": float(len(rets)),
        "window_return_median": _quantile(rets, 0.5),
        "window_return_p10": _quantile(rets, 0.1),
        "window_return_worst": min(rets),
        "window_positive_share": sum(1 for r in rets if r > 0) / len(rets),
        "window_drawdown_median": _quantile(mdds, 0.5),
        "window_drawdown_worst": max(mdds),
        "window_composite_median": _quantile(scores, 0.5),
    }


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))


def _quantile(xs: Sequence[float], q: float) -> float:
    ordered = sorted(xs)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
