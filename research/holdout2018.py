"""An untouched holdout for rounds 31 and on: October 2018 to October 2020.

Every earlier round chose among designs on the six folds from October 2020, so those years are
used up. Binance's candles go back to 2017 for the coins that existed then, so two more years,
never looked at by any research, are kept for confirmation:

    HOLDOUT = October 2018 - October 2019 (the 2018 bear market's end, the 2019 rally) and
              October 2019 - October 2020 (the COVID crash and the recovery)

Rule, fixed before any of it is looked at: a design that passes the strict rule on the six
folds counts as better only if it also has a higher composite than the incumbent in both
holdout years. No design's holdout result is looked at before it has passed on the six folds.
The universe then is thinner (only today's Roostoo coins that were listed on Binance then).

    python -m research.holdout2018   # fetch the 2018-2020 candles into the cache
"""
from concurrent.futures import ThreadPoolExecutor

from bot.config import load_config
from bot.market_data import BinanceClient, load_history
from research.folds import candidates, ms

HOLDOUT = [("2018-10-01", "2019-10-01"), ("2019-10-01", "2020-10-01")]


def fetch() -> None:
    cfg = load_config()
    client = BinanceClient()

    def one(pair):
        bars = load_history(client, pair, ms("2018-01-01"), ms("2020-10-01"), cfg.backtest.data_dir)
        return pair, len(bars)

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = dict(pool.map(one, sorted(candidates(cfg))))
    listed = [p for p, n in got.items() if n > 0]
    print("%d of %d candidates have candles before October 2020: %s" % (
        len(listed), len(got), ", ".join(p.split("/")[0] for p in listed)))


if __name__ == "__main__":
    fetch()
