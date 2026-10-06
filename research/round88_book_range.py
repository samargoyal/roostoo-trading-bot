"""Round 88: the book sized by a range (volatility) forecast (the user: "try making forecasting
range"; research/h91_range_forecast.py tests trading the forecast range itself).

The book weights each coin by 1 / its last week's standard deviation of hourly returns. A HAR
forecast (the mean of the last day's, week's and month's realised variance) predicts the next
day's volatility better than one week's history. Written down before running, judged on C1-C5
on the book alone and on the live bot:

  V1  the book weighted by 1 / the HAR forecast, same gross, a coin capped at 3x its current
      weight (neighbours: caps 2 and 5)

    python -m research.round88_book_range
"""
import warnings

from research.queue import judge

DESIGNS = {
    "V1 book weighted by the HAR volatility forecast": (
        dict(ls_sizing="har"), [dict(ls_sizing="har", ls_sizing_cap=2.0), dict(ls_sizing="har", ls_sizing_cap=5.0)]),
}


def main() -> None:
    warnings.filterwarnings("ignore")
    judge(DESIGNS, "Round 88, the book sized by a volatility forecast",
          bots=["the live book alone", "live (K2, 3 coins, 75%)"])


if __name__ == "__main__":
    main()
