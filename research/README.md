# Research scripts

The experiments behind the strategy, described round by round in
[docs/research.md](../docs/research.md). Each script's docstring states its hypothesis and the
rule fixed before it ran. They run from the repository root, for example:

```
pip install -r requirements-research.txt
python -m research.rounds 54        # one research round through the bot's own backtester
python -m research.h50_long_short   # the long-short screen
```

**Infrastructure**

| Script | What it is |
|---|---|
| `folds.py` | Six one-year folds (October 2020 to October 2026) through the bot's own backtester; the shared fold runner |
| `holdout2018.py` | The untouched holdout, October 2018 to October 2020, for rounds 31 and on |
| `holdout.py` | The first hold-out check (July to September 2026), run once |
| `rounds.py` | Rounds 20–64: each design and its neighbours, the adoption rule and the holdout check |
| `round65_attention.py` – `round74_convex.py` | Rounds 65–74, one script each: attention, securing profits, trend exits, entries and exits, the multi-horizon ranking, convex optimisation |
| `queue.py`, `round75_risk_breakers.py` – `round85_shorts_three_coins.py` | The research queue (RESEARCH_QUEUE.md) and the questions after it, each design on both bots, judged on C1–C5 |
| `d1_permutation.py`, `d2_deflated_sharpe.py` | The validation audit: a permutation test and the deflated Sharpe ratio of both bots |
| `h77_meta_labels.py` | Meta-labelled sizing of the rotation's picks (E6): a walk-forward model |
| `h76_measurements.py` | The queue's measurements: weekends, relative volume, shock bars, correlation |
| `competition_rule.py` | Every design of rounds 20 and on, judged on 14-day windows |
| `fullbars.py`, `panel.py`, `stocks.py` | Data: full Binance klines, price panels, US share data |
| `klines5m.py`, `attention.py` | Data: 5-minute Binance candles; daily Wikipedia page views per coin |
| `sim.py`, `run_variants.py` | A fast simulator used in the first rounds |

**Studies, in order**

| Script | Hypothesis |
|---|---|
| `h1_signals.py`, `h2_ml.py` | Which signals, and does machine learning, rank coins for the next 24 hours? |
| `h6_volatility.py`, `h8_regimes.py` | Time-series volatility models and market timing |
| `h14_trend.py`, `h15_rotation.py`, `h16_blend.py` | Beating buy-and-hold: BTC trend timing, momentum rotation, a blend of books |
| `h18_vecm.py`, `h19_halts.py` | VECM-ARB's engines on crypto; what halt handling is worth |
| `h20_screen.py` – `h24_order_flow.py` | Microstructure, Kalman and Bayesian signals, neural networks, order flow |
| `h25_stock_proxy.py` – `h28_cash_overlay.py` | Tokenized stocks, built on their shares' history |
| `h29_funding.py`, `h30_funding_strategy.py` | Perpetual funding rates as a crowding signal |
| `h31_sentiment.py`, `h32_sentiment_strategy.py` | Fear & Greed and stablecoin liquidity |
| `h33_indicators.py` – `h36_orderflow_strategies.py` | TradingView's indicators, order flow and ICT concepts |
| `h37_overnight.py`, `h38_overnight_selective.py` | Overnight returns of the tokenized stocks |
| `rotation_weight.py`, `book_exposure.py` | The rotation's share of the account; the book's exposure |
| `h50_long_short.py` | Long-short books (screen for rounds 50–64; the live long-short book came from round 54) |
| `h60_swing_mft.py`, `h61_swing.py` | 53 swing and 28 medium-frequency strategies, alone and beside the bot |
| `h62_attention.py` | Retail attention (Wikipedia page views): 14 strategies |
| `h68_lock_in.py` | Locking in a 14-day window's gains, the competition's horizon |
| `h69_trend_exit.py` | Do indicators, votes or machine learning tell when a trend is over? (rounds 69 and 70) |
| `h70_secure_profits.py` | Securing profits within the competition's 14-day window without a fixed target |
| `h71_timing.py` | Is there a right time to be short? Fast down-signals against the next days' returns |
| `h72_swings.py` | Trading the account's swings: do its moves revert, and what would selling the tops have made? |
| `h73_swing_sleeve.py` – `h75_gated_short_swings.py` | A swing sleeve beside the bot, long and short (shorts also only in downtrends): where its capital comes from, how much, which setups |

The scripts read cached data under `data/` (git-ignored) and write results under
`runs/research/`.
