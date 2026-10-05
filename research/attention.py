"""Daily Wikipedia page views per coin: a free, long attention series (research/h62_attention.py).

Retail attention is the classic reason given for coins that "move on Twitter", but tweets are not
free to collect, nor available years back. Wikipedia page views are (Kristoufek 2013 used them
for Bitcoin): daily from July 2015, from Wikimedia's public API. Each coin is mapped to its
English article; "Cryptocurrency" stands for the market as a whole. Saved to
data/wiki_views.csv (git-ignored). Requests are paced, with a back-off when rate limited.

    python -m research.attention
"""
import os
import time

import pandas as pd
import requests

HEADERS = {"User-Agent": "Team124-research-bot/1.0 (crypto attention study; github.com/samargoyal)"}
URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
       "{t}/daily/20150701/20261001")
ARTICLES = {
    "MARKET": ["Cryptocurrency"],
    "BTC/USD": ["Bitcoin"], "ETH/USD": ["Ethereum"], "SOL/USD": ["Solana_(blockchain_platform)"],
    "XRP/USD": ["XRP_Ledger"], "DOGE/USD": ["Dogecoin"], "SHIB/USD": ["Shiba_Inu_(cryptocurrency)"],
    "ADA/USD": ["Cardano_(blockchain_platform)"], "LTC/USD": ["Litecoin"], "POL/USD": ["Polygon_(blockchain)"],
    "DOT/USD": ["Polkadot_(cryptocurrency)"], "AVAX/USD": ["Avalanche_(blockchain_platform)"],
    "LINK/USD": ["Chainlink_(blockchain)"], "TRX/USD": ["Tron_(cryptocurrency)"], "XLM/USD": ["Stellar_(payment_network)"],
    "FIL/USD": ["Filecoin"], "ZEC/USD": ["Zcash"], "UNI/USD": ["Uniswap"],
    "NEAR/USD": ["NEAR_Protocol", "Near_Protocol"], "HBAR/USD": ["Hedera_(distributed_ledger)", "Hedera_Hashgraph"],
    "ICP/USD": ["Internet_Computer"], "PEPE/USD": ["Pepe_(cryptocurrency)", "Pepecoin"], "TRUMP/USD": ["$Trump"],
    "WLD/USD": ["World_(blockchain)", "Worldcoin"], "TAO/USD": ["Bittensor"], "APT/USD": ["Aptos_(blockchain)"],
    "ARB/USD": ["Arbitrum"], "AAVE/USD": ["Aave"], "BONK/USD": ["Bonk_(cryptocurrency)"], "SUI/USD": ["Sui_(blockchain)"],
    "ONDO/USD": ["Ondo_Finance"], "PUMP/USD": ["Pump.fun"], "FET/USD": ["Fetch.ai"], "CAKE/USD": ["PancakeSwap"],
    "CRV/USD": ["Curve_Finance"],
}
PATH = os.path.join("data", "wiki_views.csv")


def fetch(title):
    for attempt in range(6):
        r = requests.get(URL.format(t=requests.utils.quote(title, safe="")), headers=HEADERS, timeout=60)
        if r.status_code == 200:
            items = r.json()["items"]
            return pd.Series({pd.Timestamp(i["timestamp"][:8], tz="UTC"): i["views"] for i in items})
        if r.status_code == 404:
            return None
        time.sleep(15 * (attempt + 1))                       # rate limited: back off
    return None


def attention_table(short=7, long=90):
    """{00:00 of day D (ms): {"ATT:" + pair: log(7-day mean / 90-day median of views)}} from views up
    to day D-1 (a day's views are published hours after it ends): the research hook for round 65."""
    import numpy as np
    w = pd.read_csv(PATH, index_col=0, parse_dates=True)
    w.index = pd.to_datetime(w.index, utc=True).normalize()
    def norm(x):
        return np.log(x.rolling(short, min_periods=short).mean() / x.rolling(long, min_periods=long // 2).median())

    with np.errstate(divide="ignore", invalid="ignore"):
        aa = norm(w).shift(1)                                         # attention against its norm
        share = norm(w.drop(columns="MARKET").div(w["MARKET"], axis=0)).shift(1)   # share of all attention
        spike = (w / w.rolling(30, min_periods=15).median()).shift(1)  # yesterday against the month
    out = {}
    for day in aa.index:
        vals = {}
        for prefix, table in (("ATT:", aa), ("SHR:", share), ("SPK:", spike)):
            for p, v in table.loc[day].items():
                if v == v and abs(v) != float("inf") and (prefix != "SHR:" or p != "MARKET"):
                    vals[prefix + p] = float(v)
        if vals:
            out[int(day.timestamp() * 1000)] = vals
    return out


def main() -> None:
    have = pd.read_csv(PATH, index_col=0, parse_dates=True) if os.path.exists(PATH) else pd.DataFrame()
    series = {c: have[c] for c in have.columns}
    for key, titles in ARTICLES.items():
        if key in series:
            continue
        for title in titles:
            s = fetch(title)
            time.sleep(3)
            if s is not None and s.sum() > 1000:
                series[key] = s
                print("%-10s %-32s %d days from %s" % (key, title, len(s), s.index[0].date()), flush=True)
                break
        else:
            print("%-10s no article found" % key, flush=True)
        pd.DataFrame(series).sort_index().to_csv(PATH)


if __name__ == "__main__":
    main()
