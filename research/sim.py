"""A fast hourly portfolio simulator for comparing strategy ideas.

It follows the live bot's mechanics closely enough to rank ideas: decisions at each
hourly close, the BTC regime cap, the 5% PAXG core, the drawdown brake, the rebalance
threshold and taker or maker fees. Ideas that survive here are then implemented in bot/
and confirmed with the real backtester (python -m bot.backtest).

Two ways to build the portfolio from a score:
  topn  hold the best-scoring eligible coins, sized by inverse volatility (today's bot)
  mvo   mean-variance optimisation: maximise expected return minus risk and turnover
        penalties under long-only, per-coin, budget and core constraints
"""
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from bot.metrics import summarize


@dataclass
class SimConfig:
    construction: str = "topn"       # "topn" or "mvo"
    regime: bool = True              # cap exposure when BTC is below its 200h EMA
    exposure_on: float = 0.75
    exposure_off: float = 0.25
    trend_filter: bool = True        # only coins with EMA50 > EMA200 and close > EMA200
    max_positions_on: int = 4        # topn only
    max_positions_off: int = 3
    max_weight: float = 0.15
    core_weight: float = 0.05        # PAXG held at all times
    brake_dd: float = 0.04
    brake_release: float = 0.02
    brake_factor: float = 0.5
    rebalance_threshold: float = 0.04
    min_trade: float = 0.0001        # fraction of equity ($10 on $100k)
    fee: float = 0.001
    slippage: float = 0.0002
    decide_every: int = 1            # hours between decisions
    rsi_max_entry: float = 70.0      # no new entries above this RSI(14)
    stop_atr: float = 8.0            # trailing stop below the highest close since entry
    cooldown: int = 24               # hours without re-entry after a stop
    # Short sleeve (Roostoo /v6 shorts: 1x, 0.1% fee to open and to close)
    short_on: float = 0.0            # short exposure when risk-on
    short_off: float = 0.0           # short exposure when risk-off
    max_shorts: int = 3
    short_rsi_min: float = 30.0      # no new shorts into oversold coins
    short_stop_atr: float = 8.0      # trailing stop above the lowest close since entry
    short_fee: float = 0.001
    short_regime_ema: int = 200      # shorts only while BTC is below this EMA (hours)
    short_trend_filter: bool = True  # False: short the lowest scores whatever their trend (BAB)
    vol_target: float = 0.0          # > 0: scale exposure by min(1, target / BTC daily vol)
    # Sizing of the selected coins (topn)
    weighting: str = "inverse_vol"   # "inverse_vol" or "erc" (equal risk contribution)
    vol_source: str = "rolling"      # "rolling" (168h) or "forecast" (the vol_forecast panel)
    vol_managed: float = 0.0         # > 0: scale the trend sleeve to this ex-ante daily vol
    tc_bands: bool = False           # per-coin no-trade bands from transaction-cost theory
    tc_gamma: float = 2.0            # risk aversion in the band formula
    # mvo only
    ic: float = 0.08                 # expected skill: alpha = ic x volatility x z-score
    horizon: int = 24                # hours the alpha and risk are measured over
    risk_aversion: float = 10.0
    turnover_aversion: float = 0.0
    cov_window: int = 336
    shrink: float = 0.3              # pull correlations towards zero
    min_weight: float = 0.01         # drop optimiser weights smaller than this
    target_vol: float = 0.0          # > 0: pick risk aversion so ex-ante horizon vol is this
    full_budget: bool = False        # invest the whole budget (minimum-variance style)


def project(v: np.ndarray, lo: np.ndarray, hi: np.ndarray, budget: float,
            equality: bool = False) -> np.ndarray:
    """Euclidean projection onto {lo <= w <= hi, sum(w) <= budget} (or == budget)."""
    w = np.clip(v, lo, hi)
    if equality:
        budget = min(budget, float(hi.sum()))
    elif w.sum() <= budget + 1e-12:
        return w
    a, b = float(np.min(v - hi)), float(np.max(v - lo))
    for _ in range(60):
        m = (a + b) / 2
        if np.clip(v - m, lo, hi).sum() > budget:
            a = m
        else:
            b = m
    return np.clip(v - b, lo, hi)


def erc_weights(cov: np.ndarray, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Equal-risk-contribution weights (sum 1) by cyclical coordinate descent on the strictly
    convex problem  min 1/2 y'Cy - sum(log y) / n,  then normalising y."""
    n = len(cov)
    b = np.full(n, 1.0 / n)
    y = 1.0 / np.sqrt(np.diag(cov))
    for _ in range(iters):
        y_old = y.copy()
        for i in range(n):
            c = cov[i] @ y - cov[i, i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * cov[i, i] * b[i])) / (2 * cov[i, i])
        if np.max(np.abs(y - y_old)) < tol:
            break
    return y / y.sum()


def no_trade_band(weight: np.ndarray, cost: float, gamma: float) -> np.ndarray:
    """Half-width of the optimal no-trade region around a target weight under proportional
    costs (the Janecek-Shreve asymptotics of the Davis-Norman problem):
    (3 / (2 gamma) * w^2 (1 - w)^2 * cost) ** (1/3)."""
    w = np.abs(weight)
    return (1.5 / gamma * w * w * (1 - w) ** 2 * cost) ** (1.0 / 3.0)


def solve_mvo(alpha: np.ndarray, cov: np.ndarray, w_prev: np.ndarray, lo: np.ndarray,
              hi: np.ndarray, budget: float, risk_aversion: float, turnover_aversion: float,
              iters: int = 300, equality: bool = False) -> np.ndarray:
    """max alpha'w - (ra/2) w'Cw - (ta/2)|w - w_prev|^2 over the constraint set (FISTA)."""
    lipschitz = risk_aversion * float(np.linalg.eigvalsh(cov)[-1]) + turnover_aversion
    lipschitz = max(lipschitz, 1e-9)
    x = project(w_prev, lo, hi, budget, equality)
    y, t = x.copy(), 1.0
    for _ in range(iters):
        grad = -alpha + risk_aversion * cov @ y + turnover_aversion * (y - w_prev)
        x_new = project(y - grad / lipschitz, lo, hi, budget, equality)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = x_new + (t - 1) / t_new * (x_new - x)
        if np.max(np.abs(x_new - x)) < 1e-7:
            x = x_new
            break
        x, t = x_new, t_new
    return x


def simulate(close: pd.DataFrame, mask: pd.DataFrame, score: pd.DataFrame,
             cfg: SimConfig, start: str, end: str, initial: float = 100000.0,
             defensive: str = "PAXG/USD", regime_pair: str = "BTC/USD",
             high: Optional[pd.DataFrame] = None,
             low: Optional[pd.DataFrame] = None,
             vol_forecast: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, float], pd.Series]:
    close = close.ffill()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    # Same rules as bot/strategy.py: enter on EMA50 > EMA200 and close > EMA200 with RSI <= max;
    # stay in until EMA50 < EMA200 or the trailing stop.
    entry_trend = ((ema50 > ema200) & (close > ema200)).values
    hold_trend = (ema50 > ema200).values
    short_entry_trend = ((ema50 < ema200) & (close < ema200)).values
    regime_on = (close[regime_pair] > ema200[regime_pair]).values
    bear = (close[regime_pair] < close[regime_pair].ewm(span=cfg.short_regime_ema, adjust=False).mean()).values
    logret = np.log(close).diff().fillna(0.0).values
    sig = pd.DataFrame(logret, index=close.index).rolling(168, min_periods=100).std().values
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-diff).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).fillna(50).values
    hi_ = close if high is None else high.reindex_like(close).ffill()
    lo_ = close if low is None else low.reindex_like(close).ffill()
    tr = pd.concat([hi_ - lo_, (hi_ - close.shift()).abs(), (lo_ - close.shift()).abs()]).groupby(level=0).max()
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().reindex(close.index).values
    # Daily volatility per coin for sizing: the 168h estimate, or a forecast where available.
    vol24 = sig * np.sqrt(24)
    if cfg.vol_source == "forecast" and vol_forecast is not None:
        fc = np.sqrt(vol_forecast.reindex(index=close.index, columns=close.columns).values)
        vol24 = np.where(np.isfinite(fc) & (fc > 0), fc, vol24)

    times = close.index
    rows = np.where((times >= start) & (times < end))[0]
    pairs = list(close.columns)
    n = len(pairs)
    d = pairs.index(defensive)
    prices = close.values
    m = mask.reindex(index=close.index, columns=pairs).fillna(False).values
    sc = score.reindex(index=close.index, columns=pairs).values

    qty = np.zeros(n)
    cash = initial
    peak, brake = initial, False
    held = np.zeros(n, dtype=bool)
    top = np.zeros(n)                    # highest close since entry
    cool_until = np.full(n, -1)
    held_s = np.zeros(n, dtype=bool)     # open shorts
    bottom = np.full(n, np.inf)          # lowest close since a short was opened
    cool_s = np.full(n, -1)
    curve, trades = [], []
    for k, i in enumerate(rows):
        p = np.nan_to_num(prices[i])
        equity = cash + float(qty @ p)
        w_cur = qty * p / equity
        peak = max(peak, equity)
        dd = 1 - equity / peak
        if brake and dd <= cfg.brake_release:
            brake = False
        elif not brake and dd >= cfg.brake_dd:
            brake = True

        top = np.where(held, np.maximum(top, p), 0.0)
        stopped = held & (p < top - cfg.stop_atr * np.nan_to_num(atr[i]))
        cool_until[stopped] = i + cfg.cooldown
        bottom = np.where(held_s, np.minimum(bottom, p), np.inf)
        stopped_s = held_s & (p > bottom + cfg.short_stop_atr * np.nan_to_num(atr[i]))
        cool_s[stopped_s] = i + cfg.cooldown
        if k % cfg.decide_every == 0 or stopped.any() or stopped_s.any():
            cap = cfg.exposure_on if (regime_on[i] or not cfg.regime) else cfg.exposure_off
            if cfg.vol_target > 0:
                btc_vol = np.nan_to_num(sig[i, pairs.index(regime_pair)]) * np.sqrt(24)
                if btc_vol > 0:
                    cap = cfg.core_weight + (cap - cfg.core_weight) * min(1.0, cfg.vol_target / btc_vol)
            base = m[i] & ~np.isnan(sc[i]) & (p > 0)
            entry_ok = base & (rsi[i] <= cfg.rsi_max_entry) & (cool_until < i)
            hold_ok = base & ~stopped
            if cfg.trend_filter:
                entry_ok &= entry_trend[i]
                hold_ok &= hold_trend[i]
            ok = np.where(held, hold_ok, entry_ok)
            core = np.zeros(n)
            if m[i, d]:
                core[d] = cfg.core_weight
            factor = cfg.brake_factor if brake else 1.0
            if cfg.construction == "topn":
                keep = _select(cfg, sc[i], ok, held, regime_on[i] or not cfg.regime, d)
                target = _allocate(cfg, keep, vol24[i], logret, i, cap - core.sum(), d) * factor + core
            else:
                target = _mvo(cfg, sc[i], logret, i, sig[i], ok, m[i], w_cur, core,
                              core.sum() + (cap - core.sum()) * factor)
            short_cap = (cfg.short_off if bear[i] else cfg.short_on) * factor
            if short_cap > 0:
                s_base = m[i] & ~np.isnan(sc[i]) & (p > 0)
                s_base[d] = False                       # never short the defensive asset
                s_entry = s_base & (rsi[i] >= cfg.short_rsi_min) & (cool_s < i)
                s_hold = s_base & ~stopped_s
                if cfg.short_trend_filter:
                    s_entry &= short_entry_trend[i]
                    s_hold &= ~hold_trend[i]
                target = target + _shorts(cfg, sc[i], sig[i], np.where(held_s, s_hold, s_entry), held_s, short_cap)

            delta = target - w_cur
            small = np.abs(delta) < cfg.min_trade
            closing = (np.abs(target) <= 1e-9) & (np.abs(w_cur) > 0)
            opening = (np.abs(w_cur) < 0.005) & (np.abs(target) > 0)
            band = (no_trade_band(target, cfg.fee + cfg.slippage, cfg.tc_gamma) if cfg.tc_bands
                    else cfg.rebalance_threshold)
            go = ~small & (closing | opening | (np.abs(delta) >= band))
            for j in list(np.where(go & (delta < 0))[0]) + list(np.where(go & (delta > 0))[0]):
                dq = -qty[j] if closing[j] else delta[j] * equity / p[j]
                notional = abs(dq) * p[j]
                is_short = target[j] < 0 or w_cur[j] < 0
                rate = (cfg.short_fee if is_short else cfg.fee) + cfg.slippage
                if dq > 0 and not is_short:
                    notional = min(notional, max(cash, 0.0) / (1 + rate))
                    dq = notional / p[j]
                    if notional <= cfg.min_trade * equity:
                        continue
                cash -= dq * p[j] + notional * rate
                qty[j] += dq
                trades.append((times[i], notional))
            equity_after = cash + float(qty @ p)
            w_after = qty * p / equity_after
            now_held = (w_after - core) > 0.005
            top = np.where(now_held & ~held, p, top)
            held = now_held
            now_short = w_after < -0.005
            bottom = np.where(now_short & ~held_s, p, bottom)
            held_s = now_short
        curve.append((times[i], cash + float(qty @ p)))

    series = pd.Series([v for _, v in curve], index=[t for t, _ in curve])
    to_ms = lambda t: int(t.value // 1_000_000) + 3_600_000
    stats = summarize([(to_ms(t), v) for t, v in curve], initial, [(to_ms(t), x) for t, x in trades])
    return stats, series


def _shorts(cfg: SimConfig, score, sig, ok, held_s, cap) -> np.ndarray:
    """Negative weights for the lowest-scoring coins in downtrends, sized by inverse volatility."""
    n = len(score)
    order = [j for j in np.argsort(np.nan_to_num(score, nan=np.inf)) if ok[j]]
    rank = {j: r for r, j in enumerate(order)}
    keep = sorted([j for j in np.where(held_s & ok)[0]], key=lambda j: rank[j])[:cfg.max_shorts]
    for j in order:
        if len(keep) >= cfg.max_shorts:
            break
        if j not in keep:
            keep.append(j)
    w = np.zeros(n)
    inv = np.array([1 / sig[j] if sig[j] > 0 else 0 for j in keep])
    if keep and inv.sum() > 0:
        w[keep] = -np.minimum(cap * inv / inv.sum(), cfg.max_weight)
    return w


def _select(cfg: SimConfig, score, ok, held, risk_on, d) -> list:
    """The coins to hold: held ones that still qualify, then the best-scoring new ones."""
    limit = cfg.max_positions_on if risk_on else cfg.max_positions_off
    order = [j for j in np.argsort(-np.nan_to_num(score, nan=-np.inf)) if ok[j]]
    if not risk_on and ok[d] and d in order:
        order.remove(d)
        order.insert(0, d)
    rank = {j: r for r, j in enumerate(order)}
    keep = sorted([j for j in np.where(held & ok)[0]], key=lambda j: rank.get(j, 1e9))[:limit]
    for j in order:
        if len(keep) >= limit:
            break
        if j not in keep:
            keep.append(j)
    return keep


def _cov(cfg: SimConfig, logret, i, keep, vol) -> np.ndarray:
    """Daily covariance of the kept coins: shrunk hourly correlation scaled by daily vols."""
    window = logret[max(0, i - cfg.cov_window + 1):i + 1][:, keep]
    corr = np.nan_to_num(np.corrcoef(window, rowvar=False)) if len(keep) > 1 else np.ones((1, 1))
    corr = (1 - cfg.shrink) * corr + cfg.shrink * np.eye(len(keep))
    return corr * np.outer(vol, vol)


def _allocate(cfg: SimConfig, keep, vol24, logret, i, budget, d) -> np.ndarray:
    n = len(vol24)
    w = np.zeros(n)
    if not keep:
        return w
    vol = np.nan_to_num(vol24[keep])
    if np.any(vol <= 0):
        return w
    cov = None
    if cfg.weighting == "erc" and len(keep) > 1:
        cov = _cov(cfg, logret, i, keep, vol)
        base = erc_weights(cov)
    else:
        base = (1 / vol) / np.sum(1 / vol)
    caps = np.full(len(keep), cfg.max_weight)
    if d in keep:
        caps[keep.index(d)] -= cfg.core_weight
    if cfg.vol_managed > 0:
        cov = _cov(cfg, logret, i, keep, vol) if cov is None else cov
        port_vol = float(np.sqrt(base @ cov @ base))
        x = base * (cfg.vol_managed / port_vol if port_vol > 0 else 0.0)
    else:
        x = base * budget
    x = np.minimum(x, np.maximum(caps, 0.0))
    if x.sum() > budget:
        x *= budget / x.sum()
    w[keep] = x
    return w


def _mvo(cfg: SimConfig, score, logret, i, sig, ok, in_universe, w_cur, core, budget) -> np.ndarray:
    n = len(score)
    idx = np.where(in_universe & ~np.isnan(score))[0]
    target = core.copy()
    if len(idx) < 2 or i < cfg.cov_window:
        return target
    s = score[idx]
    z = (s - s.mean()) / (s.std() + 1e-12)
    vol_h = np.nan_to_num(sig[idx]) * np.sqrt(cfg.horizon)
    alpha = cfg.ic * vol_h * z
    window = logret[i - cfg.cov_window + 1:i + 1][:, idx]
    cov = np.cov(window, rowvar=False) * cfg.horizon
    cov = (1 - cfg.shrink) * cov + cfg.shrink * np.diag(np.diag(cov))
    lo = core[idx]
    hi = np.where(ok[idx], cfg.max_weight, 0.0)
    hi = np.maximum(hi, lo)
    solve = lambda ra: solve_mvo(alpha, cov, w_cur[idx], lo, hi, budget, ra,
                                 cfg.turnover_aversion, equality=cfg.full_budget)
    if cfg.target_vol > 0:
        # Risk aversion that brings ex-ante volatility down to the target (bisection in log space).
        lo_ra, hi_ra = 0.1, 1e5
        w = solve(lo_ra)
        if np.sqrt(w @ cov @ w) > cfg.target_vol:
            for _ in range(18):
                mid = np.sqrt(lo_ra * hi_ra)
                if np.sqrt(solve(mid) @ cov @ solve(mid)) > cfg.target_vol:
                    lo_ra = mid
                else:
                    hi_ra = mid
            w = solve(hi_ra)
    else:
        w = solve(cfg.risk_aversion)
    w = np.where(w - lo < cfg.min_weight, lo, w)
    target[idx] = w
    return target


def with_fees(cfg: SimConfig, maker: bool) -> SimConfig:
    return replace(cfg, fee=0.0005, slippage=0.0) if maker else cfg
