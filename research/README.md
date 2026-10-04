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
| `competition_rule.py` | Every design of rounds 20 and on, judged on 14-day windows |
| `fullbars.py`, `panel.py`, `stocks.py` | Data: full Binance klines, price panels, US share data |
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

The scripts read cached data under `data/` (git-ignored) and write results under
`runs/research/`.
