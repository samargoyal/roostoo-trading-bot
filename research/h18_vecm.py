"""H18: the three engines of VECM-ARB (github.com/samargoyal/VECM-ARB), ported to crypto and
tested walk-forward on the six folds.

  1. N-dimensional alpha engine. A Johansen trace test on a basket's log prices gives the
     cointegration rank r; a VECM at that rank gives the hedge ratios (beta) and the speeds of
     adjustment (alpha). The spread beta' log p is traded back towards its mean on its z-score:
     enter at |z| >= 2, take profit at |z| <= 0.5, and leave early if the spread's slow mean
     drifts against the position (VECM-ARB's trend guard).
  2. Dynamic hedging. If a leg cannot be traded when the spread is entered (halted, or
     impossible to borrow for a short), re-solve a minimum-variance hedge on the other N-1
     assets, keeping each survivor's side and a $0.5 long / $0.5 short book (VECM-ARB phase 4).
  3. Cointegration breakdown protocol. At every re-fit the Johansen test is re-run on the
     traded basket's trailing window; rank 0 liquidates the spread, and nothing new is entered
     until a basket is cointegrated again (VECM-ARB rolling_johnson.py).

VECM-ARB's tear sheet (Sharpe 1.02 on six Indian bank stocks) estimates beta on the whole of
2020-2026 and backtests on the same data. Here every beta is estimated only from data before
it is used.

Written down before running:
  Basket   at each re-fit, the 6 crypto pairs with the highest 30-day USD volume among the
           frozen candidates (spread <= 0.1%, PAXG excluded) with a complete window.
  Prices   log closes.
  Costs    each leg pays the taker fee (0.1%) and slippage (half its spread, at least 0.02%)
           on entry and exit. Shorts are 1x as on Roostoo; no borrowing cost is charged (none
           is documented).
  Sizing   dollar weights proportional to beta, gross exposure equal to the sleeve's capital.
  Designs  A  VECM-ARB's own settings on daily closes: a 500-day window re-fitted every 30
              days, z-score over 25 days, guard over 60 days lagged 10, threshold 1.5.
           B  the same logic on hourly closes: a 30-day window re-fitted weekly, z-score over
              72 hours, guard over 168 hours lagged 24.
           C  B, entering only when the VECM's alpha and beta give the spread a half-life
              under one week.
  Adopt    a 20% sleeve, blended with the incumbent and rebalanced daily, must beat the
           incumbent alone on the usual rule: a higher median composite, a higher composite
           in at least 4 of the 6 folds, and a worst-fold drawdown no more than 2 points
           worse. Even then it is not deployed unless the organisers confirm that spread
           trading is not the "arbitrage" the rules ban.
  Hedge    on B, every entry loses its largest short leg; the N-1 re-hedge is compared with
           simply dropping that leg: the realised volatility of the position while held (the
           lower, the better hedged), and the sleeve's composite.

    python -m research.h18_vecm
"""
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import cvxpy as cp
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

from bot.backtest import run_backtest
from bot.config import load_config
from bot.market_data import HOUR_MS, BinanceClient, load_history
from bot.metrics import DAY_MS, summarize
from research.folds import FOLDS, candidates, ms
from research.panel import load_panel

OUT = os.path.join("runs", "research")
DEFENSIVE = "PAXG/USD"
BASKET = 6
VOLUME_HOURS = 30 * 24
FEE = 0.001
SLEEVE = 0.2
INITIAL = 100_000.0
ENTRY_Z, EXIT_Z, GUARD = 2.0, 0.5, 1.5
DESIGNS = {
    "A daily, VECM-ARB settings": dict(hours=24, window=500, refit=30, zwin=25, trend=60, lag=10),
    "B hourly": dict(hours=1, window=720, refit=168, zwin=72, trend=168, lag=24),
    "C hourly, half-life < 1 week": dict(hours=1, window=720, refit=168, zwin=72, trend=168, lag=24,
                                         max_half_life=168),
}


def johansen(logp: np.ndarray):
    """Cointegration rank at 95% (trace test), then the first cointegrating vector (beta) and its
    loadings (alpha) from a VECM at that rank; (0, None, None) when there is none."""
    n = logp.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = coint_johansen(logp, 0, 1)
            rank = next((r for r in range(n) if res.lr1[r] < res.cvt[r, 1]), n)
            if rank == 0:
                return 0, None, None
            fit = VECM(logp, k_ar_diff=1, coint_rank=min(rank, n - 1), deterministic="co").fit()
        except (np.linalg.LinAlgError, ValueError):
            return 0, None, None
    beta, alpha = fit.beta[:n, 0], fit.alpha[:, 0]
    if not (np.isfinite(beta).all() and np.isfinite(alpha).all()):
        return 0, None, None
    return rank, beta, alpha


def half_life(beta: np.ndarray, alpha: np.ndarray) -> float:
    """Bars for the spread s = beta'y to halve, from the VECM: delta s = (beta'alpha) s_{t-1}."""
    phi = 1.0 + float(beta @ alpha)
    return float(np.log(0.5) / np.log(phi)) if 0.0 < phi < 1.0 else np.inf


def rehedge(cov: np.ndarray, w: np.ndarray, frozen: int):
    """VECM-ARB phase 4: minimum-variance weights with leg `frozen` pinned at 0, every other
    leg kept on its side, $0.5 long and $0.5 short. None when infeasible."""
    n = len(w)
    longs = [i for i in range(n) if w[i] > 0 and i != frozen]
    shorts = [i for i in range(n) if w[i] < 0 and i != frozen]
    if not longs or not shorts:
        return None
    x = cp.Variable(n)
    cons = [x[frozen] == 0, x[longs] >= 0, x[shorts] <= 0,
            cp.sum(x[longs]) == 0.5, cp.sum(x[shorts]) == -0.5]
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(cov))), cons)
    try:
        problem.solve()
    except cp.SolverError:
        return None
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        return None
    return np.asarray(x.value).ravel()


def freeze_largest_short(w: np.ndarray, rets: np.ndarray, mode: str) -> np.ndarray:
    """The spread's weights when its largest short leg cannot be traded: dropped (the rest
    rescaled to the same gross) or re-hedged on the other N-1 legs."""
    shorts = np.where(w < 0)[0]
    if len(shorts) == 0:
        return w
    frozen = shorts[np.argmin(w[shorts])]
    dropped = w.copy()
    dropped[frozen] = 0.0
    dropped /= np.abs(dropped).sum()
    if mode == "drop":
        return dropped
    solved = rehedge(np.cov(rets.T), w, frozen)
    return dropped if solved is None else solved


def simulate(close: pd.DataFrame, dollar: pd.DataFrame, cost: np.ndarray, start: str, end: str,
             d: dict, freeze: str = "", breakdown: bool = True):
    """One fold of the sleeve, walk-forward. `close` (forward-filled) and `dollar` (trailing
    30-day USD volume known at each bar's close) are indexed by bar open time. `breakdown`
    False switches the breakdown protocol off, for a diagnostic."""
    bar_ms = d["hours"] * HOUR_MS
    times = close.index
    first = times.searchsorted(pd.Timestamp(start, tz="UTC"))
    last = times.searchsorted(pd.Timestamp(end, tz="UTC"))
    prices = close.values
    logp = np.log(prices)
    vol = dollar.values
    n = prices.shape[1]
    cash = INITIAL
    units = np.zeros(n)
    pos, model, held = 0, None, None
    curve, held_rets = [], []
    stats = dict(fits=0, cointegrated=0, trades=0, breakdowns=0, guard_exits=0, held_bars=0)
    need = max(d["zwin"], d["trend"] + d["lag"])

    def trade(target: np.ndarray, p: np.ndarray) -> None:
        nonlocal cash, units
        notional = np.nan_to_num((target - units) * p)   # unlisted coins: no price, no trade
        cash -= notional.sum() + (np.abs(notional) * cost).sum()
        units = target

    for t in range(first, last):
        p = prices[t]
        if pos != 0:
            prev = cash + np.nansum(units * prices[t - 1])
            held_rets.append(np.nansum(units * (p - prices[t - 1])) / prev)
            stats["held_bars"] += 1
        equity = cash + np.nansum(units * p)
        if (t - first) % d["refit"] == 0:
            window = logp[t - d["window"] + 1:t + 1]
            complete = ~np.isnan(window).any(axis=0) & ~np.isnan(vol[t])
            order = [j for j in np.argsort(-np.nan_to_num(vol[t], nan=-1.0)) if complete[j]][:BASKET]
            stats["fits"] += 1
            rank, beta, alpha = johansen(window[:, order]) if len(order) == BASKET else (0, None, None)
            model = None
            if rank > 0:
                stats["cointegrated"] += 1
                if half_life(beta, alpha) <= d.get("max_half_life", np.inf):
                    model = (order, beta, window)
            if pos != 0 and breakdown:
                cols = held[0]
                held_rank = rank if cols == order else johansen(window[:, cols])[0]
                if held_rank == 0:          # breakdown protocol: liquidate
                    trade(np.zeros(n), p)
                    pos, held = 0, None
                    stats["breakdowns"] += 1
                    stats["trades"] += 1
        spec = held if pos != 0 else model
        if spec is not None:
            cols, beta = spec[0], spec[1]
            ect = logp[t - need + 1:t + 1][:, cols] @ beta
            recent = ect[-d["zwin"]:]
            sd = recent.std()
            if sd > 0 and np.isfinite(sd):
                z = (ect[-1] - recent.mean()) / sd
                drift = (ect[-d["trend"]:].mean() - ect[-d["trend"] - d["lag"]:-d["lag"]].mean()) / sd
                new = pos
                if pos == 0:
                    new = 1 if z <= -ENTRY_Z else -1 if z >= ENTRY_Z else 0
                elif pos == 1 and (drift < -GUARD or z >= -EXIT_Z):
                    new = 0
                    stats["guard_exits"] += drift < -GUARD
                elif pos == -1 and (drift > GUARD or z <= EXIT_Z):
                    new = 0
                    stats["guard_exits"] += drift > GUARD
                if new != pos:
                    if new == 0:
                        trade(np.zeros(n), p)
                        held = None
                    else:
                        w = new * beta / np.abs(beta).sum()
                        if freeze:
                            w = freeze_largest_short(w, np.diff(spec[2][:, cols], axis=0), freeze)
                        target = np.zeros(n)
                        target[cols] = w * equity / p[cols]
                        trade(target, p)
                        held = (cols, beta)
                    pos = new
                    stats["trades"] += 1
        curve.append((int(times[t].value // 10 ** 6) + bar_ms, cash + np.nansum(units * p)))
    bars_per_year = 365 * 24 / d["hours"]
    stats["held_vol"] = float(np.std(held_rets) * np.sqrt(bars_per_year)) if held_rets else 0.0
    return curve, stats


def incumbent_curve(fold):
    """The bot's default configuration on one fold (the same run as research/folds.py)."""
    start, end = fold
    path = os.path.join(OUT, "h18_incumbent_%s.json" % start[:4])
    if os.path.exists(path):
        with open(path) as f:
            return start, [tuple(x) for x in json.load(f)]
    cfg = load_config()
    s, e = ms(start), ms(end)
    warm = s - cfg.backtest.warmup_bars * HOUR_MS
    client = BinanceClient()
    slippage = candidates(cfg)
    bars = {}
    for pair in slippage:
        series = load_history(client, pair, warm, e, cfg.backtest.data_dir)
        if series:
            bars[pair] = series
    result = run_backtest(cfg, bars, s, e, cfg.backtest.taker_fee, cfg.backtest.taker_slippage,
                          "taker", monthly_universe=True, slippage_by_pair=slippage)
    with open(path, "w") as f:
        json.dump(result.curve, f)
    return start, result.curve


def blend(inc, sleeve, share: float):
    """(1 - share) of the incumbent plus `share` of the sleeve, rebalanced at 00:00 UTC."""
    by_ts = dict(sleeve)
    out = []
    ref_e = ref_i = ref_s = last_s = INITIAL
    for ts, i in inc:
        last_s = by_ts.get(ts, last_s)
        e = ref_e * ((1 - share) * i / ref_i + share * last_s / ref_s)
        out.append((ts, e))
        if ts % DAY_MS == 0:
            ref_e, ref_i, ref_s = e, i, last_s
    return out


def daily_returns(curve):
    s = pd.Series(dict(curve))
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s.resample("1D").last().pct_change().dropna()


def main() -> None:
    cfg = load_config()
    slip = candidates(cfg)
    pairs = sorted(p for p in slip if p != DEFENSIVE)
    client = BinanceClient()
    for pair in pairs:   # design A's 500-day window needs history from 2019
        load_history(client, pair, ms("2019-01-01"), ms("2026-10-01"), cfg.backtest.data_dir)

    with ProcessPoolExecutor(max_workers=6) as pool:
        incumbent = dict(pool.map(incumbent_curve, FOLDS))

    panel = load_panel(pairs, cfg.backtest.data_dir)
    close_h = panel["close"][pairs].ffill()
    dollar_h = (panel["close"][pairs] * panel["volume"][pairs]).rolling(VOLUME_HOURS, min_periods=VOLUME_HOURS).sum()
    frames = {1: (close_h, dollar_h),
              24: (close_h.resample("1D").last(), dollar_h.resample("1D").last())}
    cost = np.array([FEE + slip[p] for p in pairs])

    years = [f[0][:4] for f in FOLDS]
    table = {"incumbent": {}}
    for start, _ in FOLDS:
        st = summarize(incumbent[start], INITIAL)
        table["incumbent"][start[:4]] = {"ret": st["total_return"], "mdd": st["max_drawdown"],
                                         "comp": st["composite"]}
    runs = [(name, d, "") for name, d in DESIGNS.items()]
    runs += [("B, largest short frozen: dropped", DESIGNS["B hourly"], "drop"),
             ("B, largest short frozen: N-1 re-hedge", DESIGNS["B hourly"], "rehedge")]
    sleeve_rows, blend_rows = [], []
    for name, d, freeze in runs:
        close, dollar = frames[d["hours"]]
        table[name] = {}
        row = {"sleeve": name}
        comps, rets, held_vols, corr = [], [], [], []
        counts = dict(fits=0, cointegrated=0, trades=0, breakdowns=0, guard_exits=0)
        for start, end in FOLDS:
            curve, stats = simulate(close, dollar, cost, start, end, d, freeze)
            st = summarize(curve, INITIAL)
            comps.append(st["composite"])
            rets.append(st["total_return"])
            held_vols.append(stats["held_vol"])
            row[start[:4]] = "%+.0f%% (%.0f%%)" % (st["total_return"] * 100, st["max_drawdown"] * 100)
            for k in counts:
                counts[k] += stats[k]
            inc_r, slv_r = daily_returns(incumbent[start]), daily_returns(curve)
            both = pd.concat([inc_r, slv_r], axis=1).dropna()
            corr.append(both.corr().iloc[0, 1] if len(both) > 2 and both.iloc[:, 1].std() > 0 else 0.0)
            mixed = summarize(blend(incumbent[start], curve, SLEEVE), INITIAL)
            table[name][start[:4]] = {"sleeve_ret": st["total_return"], "sleeve_mdd": st["max_drawdown"],
                                      "sleeve_comp": st["composite"], "ret": mixed["total_return"],
                                      "mdd": mixed["max_drawdown"], "comp": mixed["composite"],
                                      **stats}
        row["median comp"] = "%.2f" % float(np.median(comps))
        row["cointegrated"] = "%d/%d" % (counts["cointegrated"], counts["fits"])
        row["trades"] = counts["trades"]
        row["breakdown exits"] = counts["breakdowns"]
        row["guard exits"] = counts["guard_exits"]
        row["held vol"] = "%.0f%%" % (np.mean(held_vols) * 100)
        row["corr"] = "%.2f" % float(np.mean(corr))
        sleeve_rows.append(row)

    base = table["incumbent"]
    for name in ["incumbent"] + [r[0] for r in runs]:
        by_year = table[name]
        comps = [by_year[y]["comp"] for y in years]
        blend_rows.append({
            "design": name if name == "incumbent" else "incumbent + 20% " + name,
            **{y: "%+.0f%% (%.0f%%) %.2f" % (by_year[y]["ret"] * 100, by_year[y]["mdd"] * 100,
                                             by_year[y]["comp"]) for y in years},
            "median comp": "%.2f" % float(np.median(comps)),
            "better": "%d/6" % sum(by_year[y]["comp"] > base[y]["comp"] for y in years),
            "worst mdd": "%.0f%%" % (max(by_year[y]["mdd"] for y in years) * 100)})
    with open(os.path.join(OUT, "h18_vecm.json"), "w") as f:
        json.dump(table, f, indent=1, default=float)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print("The sleeve alone: return per fold (max drawdown); folds start in October")
    print(pd.DataFrame(sleeve_rows).to_string(index=False))
    print("\nBlended with the incumbent: return (max drawdown) composite per fold")
    print(pd.DataFrame(blend_rows).to_string(index=False))

    # Information only: is the loss the costs or the signal, and did the breakdown exit help?
    print("\nDiagnostics: the sleeve's return per fold")
    zero = np.zeros(len(pairs))
    checks = [(name + ", no costs", d, zero, True) for name, d in DESIGNS.items()]
    checks.append(("B hourly, breakdown protocol off", DESIGNS["B hourly"], cost, False))
    for name, d, c, on in checks:
        close, dollar = frames[d["hours"]]
        rets = [summarize(simulate(close, dollar, c, s, e, d, breakdown=on)[0], INITIAL)["total_return"]
                for s, e in FOLDS]
        print("  %-40s %s   mean %+.1f%%" % (name, "  ".join("%+4.0f%%" % (r * 100) for r in rets),
                                            float(np.mean(rets)) * 100))


if __name__ == "__main__":
    main()
