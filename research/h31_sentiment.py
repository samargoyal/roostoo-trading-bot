"""H31 (round 41): market sentiment and stablecoin liquidity, data the bot has not used.

  Fear & Greed  the Crypto Fear & Greed index (alternative.me), 0-100, daily since 2018
  Liquidity     the 7-day % change of the total stablecoin supply (DefiLlama), daily since
                2017: new money waiting to buy crypto

Each day's value is used only from the next 00:00 UTC (a day's lag, so it was published).
Written down before running: for each, the Spearman correlation with BTC's and with the
equal-weight universe's next 7 days in each of the six folds; worth a strategy test with the
same sign in at least 5 of 6 folds and a mean |correlation| of at least 0.05. The 2018-2020
holdout is not looked at.

    python -m research.h31_sentiment
"""
import os

import numpy as np
import pandas as pd
import requests

from bot.config import UniverseConfig, load_config
from research.folds import FOLDS, candidates
from research.panel import DEFENSIVE, load_panel, monthly_universe

FNG = os.path.join("data", "fear_greed.csv")
STABLE = os.path.join("data", "stablecoins.csv")


def fear_greed() -> pd.Series:
    if not os.path.exists(FNG):
        d = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=60).json()["data"]
        pd.DataFrame([(int(x["timestamp"]), int(x["value"])) for x in d], columns=["ts", "value"]).to_csv(FNG, index=False)
    df = pd.read_csv(FNG).sort_values("ts")
    s = pd.Series(df["value"].values, index=pd.to_datetime(df["ts"], unit="s", utc=True))
    s.index = s.index + pd.Timedelta(days=1)          # usable from the next day
    return s


def stable_supply() -> pd.Series:
    if not os.path.exists(STABLE):
        d = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=60).json()
        pd.DataFrame([(int(x["date"]), float(x["totalCirculatingUSD"]["peggedUSD"])) for x in d],
                     columns=["ts", "usd"]).to_csv(STABLE, index=False)
    df = pd.read_csv(STABLE).sort_values("ts")
    s = pd.Series(df["usd"].values, index=pd.to_datetime(df["ts"], unit="s", utc=True))
    s.index = s.index + pd.Timedelta(days=1)
    return s


def main() -> None:
    cfg = load_config()
    pairs = sorted(candidates(cfg))
    panel = load_panel(pairs)
    close = panel["close"]
    mask = monthly_universe(panel, UniverseConfig(), list(close.columns))
    mask[DEFENSIVE] = False
    daily = close[close.index.hour == 23]
    daily.index = daily.index + pd.Timedelta(hours=1)   # 00:00 closes
    member = mask[mask.index.hour == 23]
    member.index = member.index + pd.Timedelta(hours=1)
    btc_fwd = np.log(daily["BTC/USD"]).shift(-7) - np.log(daily["BTC/USD"])
    ew = np.log((daily.pct_change().where(member).mean(axis=1).fillna(0) + 1).cumprod())
    ew_fwd = ew.shift(-7) - ew
    signals = {"Fear & Greed level": fear_greed(),
               "stablecoin supply, 7-day change": stable_supply().pct_change(7)}
    for name, sig in signals.items():
        sig = sig.reindex(daily.index, method="ffill")
        for target_name, target in (("BTC next 7d", btc_fwd), ("universe next 7d", ew_fwd)):
            corrs = []
            for s, e in FOLDS:
                both = pd.concat([sig, target], axis=1).dropna()
                both = both[(both.index >= pd.Timestamp(s, tz="UTC")) & (both.index < pd.Timestamp(e, tz="UTC"))]
                corrs.append(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
            mean = float(np.nanmean(corrs))
            same = sum(np.sign(c) == np.sign(mean) for c in corrs)
            print("%-34s vs %-17s %s | mean %+.3f, same sign %d/6 -> %s" % (
                name, target_name, " ".join("%+.3f" % c for c in corrs), mean, same,
                "PASS" if same >= 5 and abs(mean) >= 0.05 else "fail"))


if __name__ == "__main__":
    main()
