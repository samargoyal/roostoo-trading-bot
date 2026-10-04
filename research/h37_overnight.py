"""H37: an overnight-return predictor for the tokenized stocks.

US shares earn most of their return between the close and the next open (2016-2026: about +26%
a year overnight against +5% during the session for the tokenized shares). Holding only
overnight means two trades a day, about 0.2% a night at market-order fees (0.1% with limit
orders), so a predictor must pick nights that earn more than that.

Written down before running, on daily open/close data 2016-2026 (Yahoo), for 2011's 40 largest
US companies (no hindsight) and for the tokenized shares:
  P0  every stock, every night, equally weighted (no predictor)
  P1  overnight momentum (Lou, Polk and Skouras, 2019): the top fifth by the mean overnight
      return of the previous 21 nights
  P2  intraday reversal: the top fifth by the most negative return over today's session
  P3  P1 and P2 together (sum of their cross-sectional ranks)
Each is held from the close to the next open. Compared with buying and holding the same stocks:
better only if, after 0.1% round-trip costs, it beats holding in most years.

    python -m research.h37_overnight
"""
import numpy as np
import pandas as pd
import yfinance as yf

from research.folds import candidate_table

BIG_2011 = ["XOM", "AAPL", "MSFT", "BRK-B", "WMT", "GE", "GOOGL", "CVX", "IBM", "PG", "JNJ", "T", "WFC", "JPM",
            "KO", "PFE", "ORCL", "INTC", "C", "BAC", "PM", "MRK", "VZ", "CSCO", "QCOM", "PEP", "SLB", "COP", "AMZN",
            "CMCSA", "ABT", "MCD", "DIS", "HD", "UNH", "OXY", "V", "MA", "BA", "MMM"]


def data(tickers):
    df = yf.download(tickers, start="2015-11-01", progress=False, auto_adjust=True, group_by="ticker", threads=True)
    o = pd.DataFrame({t: df[t]["Open"] for t in tickers if t in df.columns.get_level_values(0)})
    c = pd.DataFrame({t: df[t]["Close"] for t in tickers if t in df.columns.get_level_values(0)})
    return o, c


def evaluate(name, tickers):
    o, c = data(tickers)
    night = o.shift(-1) / c - 1            # from today's close to tomorrow's open
    session = c / o - 1                    # today's open to close
    past_nights = (o / c.shift(1) - 1).rolling(21).mean()   # known at today's close
    k = lambda row: max(2, int(round(0.2 * row.notna().sum())))

    def top(score):
        r = score.rank(axis=1, ascending=False)
        n = score.notna().sum(axis=1).apply(lambda x: max(2, int(round(0.2 * x))))
        return r.le(n, axis=0) & score.notna()

    picks = {"P0 every stock, every night": night.notna(),
             "P1 overnight momentum": top(past_nights),
             "P2 intraday reversal": top(-session),
             "P3 both": top(past_nights.rank(axis=1) + (-session).rank(axis=1))}
    hold = (c / c.shift(1) - 1).mean(axis=1)          # equal-weight buy and hold, daily
    years = range(2016, 2026)
    print("\n%s (%d stocks)" % (name, c.shape[1]))
    print("  %-28s %9s %11s %13s %13s  %s" % ("", "bp/night", "nights up", "net 0.1%/yr", "net 0.2%/yr",
                                              "years beating hold (0.1%)"))
    hold_years = {y: (1 + hold[hold.index.year == y]).prod() - 1 for y in years}
    for label, sel in picks.items():
        nightly = night.where(sel).mean(axis=1).dropna()
        nightly = nightly[nightly.index.year >= 2016]
        net1, net2 = nightly - 0.001, nightly - 0.002
        yrs = len(nightly) / 252
        ann = lambda s: (1 + s).prod() ** (1 / yrs) - 1
        beat = sum((1 + net1[net1.index.year == y]).prod() - 1 > hold_years[y] for y in years)
        print("  %-28s %9.1f %10.0f%% %+12.1f%% %+12.1f%%  %d/%d" % (
            label, nightly.mean() * 1e4, (nightly > 0).mean() * 100, ann(net1) * 100, ann(net2) * 100, beat, len(years)))
    print("  %-28s %+52.1f%%  (buy and hold, per year)" % ("hold", ((1 + hold[hold.index.year >= 2016]).prod()
                                                                ** (252 / len(hold[hold.index.year >= 2016])) - 1) * 100))


def main() -> None:
    tok = [r["pair"].split("/")[0][:-1] for r in candidate_table() if r["asset_type"] == "stock"]
    evaluate("2011's 40 largest US companies (no hindsight)", BIG_2011)
    evaluate("the tokenized shares (chosen with hindsight)", tok)


if __name__ == "__main__":
    main()
