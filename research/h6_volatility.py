"""H6 and H7: do time-series volatility models beat the bot's simple 168-hour estimate?

H6, forecast accuracy: each day at 00:00 UTC, every model forecasts the next 24 hours'
realised variance (sum of squared hourly log returns) for each coin in the universe.
Scored with QLIKE, the standard robust loss for variance forecasts (lower is better), and
the R-squared of log realised variance on the log forecast.

H7, ranking: the daily cross-sectional rank IC of minus each forecast with the next 24h
and 72h returns. Does a better volatility forecast make a better low-volatility ranking?

Models, all walk-forward (the parametric ones refitted monthly on the trailing 180 days):
  roll168  variance of the last 168 hourly returns (what the bot ranks by today)
  ewma     exponentially weighted squared returns, half-life 24 hours
  garch    GARCH(1,1) with Student-t errors
  gjr      GJR-GARCH(1,1,1), where falls raise volatility more than rises
  har      HAR-RV: next day's log realised variance on the last day, week and month
           (pooled OLS across coins, refitted monthly on all earlier data)

    python -m research.h6_volatility
"""
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from arch import arch_model

from bot.config import UniverseConfig
from research.h1_signals import daily_ic
from research.panel import cached_pairs, load_panel, monthly_universe, targets

OUT = os.path.join("runs", "research", "vol_forecasts.pkl")
FIRST_MONTH, LAST_MONTH = "2024-10-01", "2026-09-01"
WINDOW = 180 * 24       # hours of returns each GARCH fit sees
MIN_HISTORY = 2000      # hours a coin needs before its first fit
PERIODS = {"Y0 Oct24-Sep25": ("2024-10-01", "2025-10-01"),
           "Y1 Oct25-Jun26": ("2025-10-01", "2026-07-01"),
           "Jul-Sep26 (seen)": ("2026-07-01", "2026-10-01")}


def fit_coin(job):
    """Monthly GARCH and GJR-GARCH parameters for one coin: {model: [(month, params)]}."""
    pair, returns, month_starts = job
    out = {"garch": [], "gjr": []}
    for month in month_starts:
        train = returns[(returns.index < month) & (returns.index >= month - pd.Timedelta(hours=WINDOW))].dropna()
        for name, o in (("garch", 0), ("gjr", 1)):
            params = None
            if len(train) >= MIN_HISTORY:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        res = arch_model(train.values * 100, mean="Zero", vol="GARCH", p=1, o=o, q=1,
                                         dist="t", rescale=False).fit(disp="off", show_warning=False)
                        p = res.params
                        params = (p["omega"], p["alpha[1]"], p.get("gamma[1]", 0.0), p["beta[1]"])
                    except Exception:  # noqa: BLE001 - a failed fit leaves the month empty
                        params = None
            out[name].append((month, params))
    return pair, out


def filter_forecasts(returns: pd.Series, fits) -> pd.Series:
    """Hourly forecasts of the next 24 hours' variance from each month's parameters."""
    r = returns.values * 100
    index = returns.index
    out = np.full(len(r), np.nan)
    months = [m for m, _ in fits] + [index[-1] + pd.Timedelta(hours=1)]
    for (month, params), next_month in zip(fits, months[1:]):
        if params is None:
            continue
        omega, alpha, gamma, beta = params
        start = index.searchsorted(month)
        end = index.searchsorted(next_month)
        lo = max(0, start - WINDOW)
        history = r[lo:start]
        history = history[~np.isnan(history)]
        if len(history) < MIN_HISTORY:
            continue
        sigma2 = float(np.var(history))
        phi = alpha + gamma / 2 + beta
        long_run = omega / (1 - phi) if phi < 1 else None
        for t in range(lo, end):
            eps = 0.0 if np.isnan(r[t]) else r[t]
            sigma2 = omega + (alpha + (gamma if eps < 0 else 0.0)) * eps * eps + beta * sigma2
            if t >= start:
                if long_run is None:
                    out[t] = 24 * sigma2
                else:
                    out[t] = 24 * long_run + (sigma2 - long_run) * (1 - phi ** 24) / (1 - phi)
    return pd.Series(out / 1e4, index=index)


def har_forecasts(r: pd.DataFrame, mask: pd.DataFrame, month_starts) -> pd.DataFrame:
    sq = r ** 2
    rv_d = sq.rolling(24).sum()
    rv_w = sq.rolling(168).sum() / 7
    rv_m = sq.rolling(720).sum() / 30
    rv_next = sq.rolling(24).sum().shift(-24)
    daily = r.index[r.index.hour == 0]
    def stack(frame):
        return frame.loc[daily].where(mask.loc[daily]).stack()
    X = pd.DataFrame({"d": np.log(stack(rv_d)), "w": np.log(stack(rv_w)), "m": np.log(stack(rv_m))})
    y = np.log(stack(rv_next)).rename("y")
    data = X.join(y, how="inner").replace([np.inf, -np.inf], np.nan)
    times = data.index.get_level_values(0)
    every_hour = pd.DataFrame({"d": np.log(rv_d.stack()), "w": np.log(rv_w.stack()),
                               "m": np.log(rv_m.stack())}).replace([np.inf, -np.inf], np.nan)
    hour_times = every_hour.index.get_level_values(0)
    pieces = []
    for month, next_month in zip(month_starts, list(month_starts[1:]) + [r.index[-1] + pd.Timedelta(hours=1)]):
        train = data[times < month - pd.Timedelta(hours=24)].dropna()
        A = np.column_stack([np.ones(len(train)), train[["d", "w", "m"]].values])
        beta, *_ = np.linalg.lstsq(A, train["y"].values, rcond=None)
        resid_var = np.var(train["y"].values - A @ beta)
        test = every_hour[(hour_times >= month) & (hour_times < next_month)].dropna()
        pred = np.exp(beta[0] + test.values @ beta[1:] + resid_var / 2)
        pieces.append(pd.Series(pred, index=test.index))
    return pd.concat(pieces).unstack().reindex(index=r.index, columns=r.columns)


def main() -> None:
    pairs = cached_pairs()
    panel = load_panel(pairs)
    mask = monthly_universe(panel, UniverseConfig(), pairs)
    r = np.log(panel["close"]).diff()
    month_starts = list(pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS", tz="UTC"))

    forecasts = {
        "roll168": r.rolling(168, min_periods=150).var() * 24,
        "ewma": (r ** 2).ewm(halflife=24, adjust=False).mean() * 24,
    }
    print("fitting GARCH and GJR-GARCH: %d coins x %d months x 2 models" % (len(pairs), len(month_starts)), flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        fits = dict(pool.map(fit_coin, [(p, r[p], month_starts) for p in pairs]))
    for name in ("garch", "gjr"):
        forecasts[name] = pd.DataFrame({p: filter_forecasts(r[p], fits[p][name]) for p in pairs})
    forecasts["har"] = har_forecasts(r, mask, month_starts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"forecasts": forecasts, "fits": fits}, f)

    rv_next = (r ** 2).rolling(24).sum().shift(-24)
    daily = r.index[r.index.hour == 0]
    in_universe = mask.loc[daily]
    print("\nH6: forecasting the next 24 hours' realised variance (daily, universe coins)")
    print("%-9s" % "model" + "".join("%28s" % p for p in PERIODS))
    for name, fc in forecasts.items():
        cells = []
        for start, end in PERIODS.values():
            rows = daily[(daily >= start) & (daily < end)]
            F = fc.loc[rows].where(in_universe.loc[rows]).stack()
            RV = rv_next.loc[rows].where(in_universe.loc[rows]).stack()
            both = pd.concat([F.rename("f"), RV.rename("rv")], axis=1).dropna()
            both = both[(both.f > 0) & (both.rv > 0)]
            ratio = both.rv / both.f
            qlike = float(np.mean(ratio - np.log(ratio) - 1))
            r2 = float(np.corrcoef(np.log(both.f), np.log(both.rv))[0, 1] ** 2)
            cells.append("QLIKE %.3f  R2 %.3f" % (qlike, r2))
        print("%-9s" % name + "".join("%28s" % c for c in cells))

    print("\nH7: rank IC of low forecast volatility with the next 24h / 72h return")
    print("%-9s" % "model" + "".join("%28s" % p for p in PERIODS))
    fwd = {h: targets(panel, h)["fwd"] for h in (24, 72)}
    for name, fc in forecasts.items():
        ics = {h: daily_ic(-fc, fwd[h], mask) for h in fwd}
        cells = []
        for start, end in PERIODS.values():
            vals = []
            for h in (24, 72):
                x = ics[h][(ics[h].index >= start) & (ics[h].index < end)]
                vals.append("%.3f (t %.1f)" % (x.mean(), x.mean() / x.std() * np.sqrt(len(x))))
            cells.append(" / ".join(vals))
        print("%-9s" % name + "".join("%28s" % c for c in cells))


if __name__ == "__main__":
    main()
