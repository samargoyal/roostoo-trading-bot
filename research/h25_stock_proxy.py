"""H25: build the tokenized-stock strategy on the underlying stocks' long history.

Roostoo's 21 tokenized stocks (NVDAB, TSLAB, ...) have only been listed on Binance since
June-July 2026, too little history to test anything on. The underlying shares have years of
it. If the tokens track their shares, a strategy can be tested on the shares and traded on
the tokens.

Part A, tracking: each token's price at the US close (the close of Binance's 19:00 UTC bar,
20:00 UTC while New York is on summer time) against the share's official close, every trading
day since the token listed: the average gap (basis), how much it varies, and the correlation
of daily returns.

Part B, strategies on the shares (daily closes, decided and traded at the close, as the
tokens trade around the clock; 0.15% per trade for the fee, spread and tracking error). A share
is eligible once it has a year of history, so the newest never count. Fixed before running:
  S0  hold every eligible share equally, rebalanced monthly (the benchmark)
  S1  trend: a 1/N share of each eligible stock while its 50-day average is above its 200-day
      average and it closes above the 200-day; cash otherwise
  S2  momentum rotation, like the crypto sleeve: every 5 trading days, the 2 eligible shares
      with the best positive 63-day return, while the equal-weight basket of eligible shares
      is above its 200-day average; cash otherwise
Folds run October to October from 2011. Then each of S1 and S2, as a 20% sleeve beside the
crypto bot (rebalanced daily), faces the usual rule on the six crypto folds.

Caveat: the 21 shares are the ones tokenized in 2026, chosen for being popular after big runs
(NVDA, PLTR, MSTR, MU...), so every result here is flattered; comparisons between designs on
the same list are fairer than their absolute returns. README round 16 adds the check that
settles it: the same rotation on the 40 largest US companies of 2011.

    python -m research.h25_stock_proxy
"""
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

from bot.metrics import summarize
from research.folds import FOLDS, candidate_table
from research.h18_vecm import blend

COST = 0.0015
SLEEVE = 0.2
INITIAL = 100_000.0

STOCK_CACHE = os.path.join("data", "stocks")


def underlying(pair: str) -> str:
    """NVDAB/USD -> NVDA (the B marks the tokenized share)."""
    return pair.split("/")[0][:-1]


def daily_stock(ticker: str, start: str = "2010-01-01") -> pd.Series:
    """Daily closes adjusted for splits and dividends, cached as CSV."""
    os.makedirs(STOCK_CACHE, exist_ok=True)
    path = os.path.join(STOCK_CACHE, ticker + ".csv")
    if not os.path.exists(path):
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        close = df["Close"]
        close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        close.rename("close").to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    return s.dropna()


def token_closes(pair: str) -> pd.Series:
    """The token's price at 20:00 UTC each day (close of the 19:00 bar), from the candle cache."""
    path = os.path.join("data", "binance", pair.split("/")[0] + "USDT_1h.csv")
    df = pd.read_csv(path)
    t = pd.to_datetime(df["ts"], unit="ms", utc=True)
    s = pd.Series(df["close"].values, index=t)
    s = s[s.index.hour == 19]
    s.index = s.index.tz_localize(None).normalize()
    return s


def simulate(closes: pd.DataFrame, weights: pd.DataFrame, start: str, end: str):
    """Daily equity from target weights decided at each close (held to the next close)."""
    rets = closes.pct_change().fillna(0.0)
    days = closes.index[(closes.index >= start) & (closes.index < end)]
    equity, held, curve = INITIAL, pd.Series(0.0, index=closes.columns), []
    for d in days:
        equity *= 1 + float((held * rets.loc[d]).sum())
        drifted = held * (1 + rets.loc[d])
        drifted = drifted / (1 + float((held * rets.loc[d]).sum())) if held.sum() > 0 else drifted
        want = weights.loc[d].fillna(0.0)
        equity *= 1 - COST * float((want - drifted).abs().sum())
        held = want
        curve.append((int(pd.Timestamp(d).tz_localize("UTC").value // 10 ** 6) + 21 * 3_600_000, equity))
    return curve


def designs(closes: pd.DataFrame) -> dict:
    eligible = closes.notna() & (closes.notna().cumsum() >= 252)
    n = eligible.sum(axis=1).replace(0, np.nan)
    month_end = pd.Series(closes.index.to_period("M"), index=closes.index)
    first = month_end != month_end.shift(1)
    s0 = eligible.div(n, axis=0).where(first).ffill()
    sma50, sma200 = closes.rolling(50).mean(), closes.rolling(200).mean()
    trend = eligible & (sma50 > sma200) & (closes > sma200)
    s1 = trend.astype(float).div(n, axis=0)
    basket = (closes.pct_change().where(eligible).mean(axis=1).fillna(0.0) + 1).cumprod()
    market_on = basket > basket.rolling(200).mean()
    mom = closes / closes.shift(63) - 1
    s2 = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    for i, d in enumerate(closes.index):
        if i % 5 == 0:
            current = pd.Series(0.0, index=closes.columns)
            if market_on.loc[d]:
                m = mom.loc[d].where(eligible.loc[d]).dropna()
                picks = m[m > 0].nlargest(2).index
                current[picks] = 0.5
        s2.loc[d] = current
    return {"S0 hold all equally": s0.fillna(0.0), "S1 trend per stock": s1.fillna(0.0),
            "S2 momentum rotation (top 2)": s2}


def main() -> None:
    rows = []
    for r in candidate_table():
        if r["asset_type"] != "stock":
            continue
        pair, ticker = r["pair"], underlying(r["pair"])
        stock = daily_stock(ticker)
        if stock.empty:
            rows.append({"token": pair, "share": ticker, "share history from": "not found"})
            continue
        token = token_closes(pair)
        both = pd.concat([token.rename("token"), stock.rename("share")], axis=1, join="inner").dropna()
        basis = both["token"] / both["share"] - 1
        rets = np.log(both).diff().dropna()
        rows.append({"token": pair, "share": ticker, "share history from": stock.index[0].date(),
                     "days compared": len(both), "mean gap": "%+.2f%%" % (basis.mean() * 100),
                     "gap sd": "%.2f%%" % (basis.std() * 100),
                     "return corr": round(rets["token"].corr(rets["share"]), 3),
                     "tracking error/day": "%.2f%%" % ((rets["token"] - rets["share"]).std() * 100)})
    pd.set_option("display.width", 260)
    print("Tokens against their shares at the US close, since each token listed")
    print(pd.DataFrame(rows).to_string(index=False))

    tickers = [underlying(r["pair"]) for r in candidate_table() if r["asset_type"] == "stock"]
    closes = pd.DataFrame({t: daily_stock(t) for t in tickers}).sort_index()
    closes = closes[closes.index >= "2010-01-01"]
    qqq = daily_stock("QQQ")
    weights = designs(closes)
    folds = [("%d-10-01" % y, "%d-10-01" % (y + 1)) for y in range(2011, 2026)]
    rows = []
    for name, w in list(weights.items()) + [("QQQ (Nasdaq 100)", None)]:
        row = {"design": name}
        comps = []
        for start, end in folds:
            if w is None:
                q = qqq[(qqq.index >= start) & (qqq.index < end)]
                curve = [(int(pd.Timestamp(d).tz_localize("UTC").value // 10 ** 6), INITIAL * v / q.iloc[0])
                         for d, v in q.items()]
            else:
                curve = simulate(closes, w, start, end)
            st = summarize(curve, INITIAL)
            row[start[2:4]] = "%+.0f%%/%.0f%%" % (st["total_return"] * 100, st["max_drawdown"] * 100)
            comps.append(st["composite"])
        row["median comp"] = round(float(np.median(comps)), 2)
        rows.append(row)
    print("\nOn the shares, return / max drawdown per October-to-October year (20xx)")
    print(pd.DataFrame(rows).to_string(index=False))

    # As a 20% sleeve beside the crypto bot, on the crypto folds.
    table = {}
    for name in ("S1 trend per stock", "S2 momentum rotation (top 2)"):
        table[name] = []
        for start, end in FOLDS:
            with open(os.path.join("runs", "research", "h18_incumbent_%s.json" % start[:4])) as f:
                inc = [tuple(x) for x in json.load(f)]
            sleeve = simulate(closes, weights[name], start, end)
            # Hold the stock sleeve's value flat between closes on the bot's hourly clock.
            stamps = pd.Series(dict(sleeve)).sort_index()
            hourly = [(ts, float(stamps[stamps.index <= ts].iloc[-1]) if (stamps.index <= ts).any() else INITIAL)
                      for ts, _ in inc]
            mixed = summarize(blend(inc, hourly, SLEEVE), INITIAL)
            alone = summarize(inc, INITIAL)
            table[name].append((start[:4], alone["composite"], mixed["composite"], alone["max_drawdown"],
                                mixed["max_drawdown"], mixed["total_return"]))
    print("\nThe crypto bot alone and with a 20% stock sleeve: composite per crypto fold")
    for name, rows_ in table.items():
        better = sum(m > a for _, a, m, _, _, _ in rows_)
        print("  %s: %s | median %.2f vs %.2f, better in %d/6, worst drawdown %.0f%% vs %.0f%%" % (
            name, "  ".join("%s %.2f->%.2f" % (y, a, m) for y, a, m, _, _, _ in rows_),
            float(np.median([m for _, _, m, _, _, _ in rows_])), float(np.median([a for _, a, _, _, _, _ in rows_])),
            better, max(md for _, _, _, _, md, _ in rows_) * 100, max(ad for _, _, _, ad, _, _ in rows_) * 100))


if __name__ == "__main__":
    main()
