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
| `h86_scalping.py` | Seven scalping strategies on 5-minute bars, longs and shorts judged apart |
| `h87_orb.py` | Opening-range breakouts at the Asia, London and US opens and the UTC day, 252 designs |
| `h88_williams_r.py`, `round86_williams_r.py` | Williams %R alone (five designs) and on the rotation |
| `h89_rsi3_vwap.py` | The RSI(3) + EMA + VWAP dip-buy, daily, 4-hour and hourly |
| `h90_order_flow.py`, `h90_holdout.py` | 17 order-flow signals from taker volume and trade counts, votes and pairs, and the survivors on 2018–20 |
| `round87_book_sde.py` | The long-short book from stochastic calculus: Merton and Kalman sizing, variance ratio, jumps, volatility management |
| `h91_range_forecast.py`, `round88_book_range.py` | A HAR forecast of the daily range: its accuracy, trading its bands, and the book weighted by it |
| `h92_ml_ranking.py`, `h93_ml_book.py` | Ridge, LightGBM and a neural network ranking coins, walk-forward: for the rotation and as a long-short book |
| `h94_ml_book.py`, `h94_q1_robust.py`, `round89_book_breadth.py` | ML for the book alone: direction models, meta-labels, a learned controller, and the breadth rule it found |
| `round90_book_correlation.py` | Correlation on the book alone: ERC, minimum variance, HRP, and a correlation spike halving the book |
| `round91_more_return.py` | Return against risk for the split, 75% to 95% rotation, and the HRP book |
| `round92_book_return.py`, `round93_book_dual.py`, `round94_book_chop.py` | The book for return: concentration, faster longs, fresh trends, dual momentum, hysteresis, efficiency ratio |
| `round95_efficiency.py`, `round96_book_quality.py`, `round97_efficiency_stress.py` | Trend-quality weights for the book (efficiency ratio, R^2), judged on return, and their stress tests |
| `round98_efficiency_more.py`, `round99_lower_split.py`, `round100_book_convex.py` | More from the efficiency ratio, the split with the efficiency book, convex optimisation on it |
| `round101_book_at_55.py` | The squared efficiency tilt and fresh trends at the live split |
| `h95_aggressive_sleeve.py`, `h96_more_aggressive.py` | An aggressive third sleeve (swings, breakouts, dip-buys, fast and all-in momentum, memes, concentrated books) in a 30/30/40 split |
| `h97_two_books.py`, `round103_dollar_neutral.py` | Two books without momentum; a dollar-neutral book |
| `round104_winners_losers.py`, `round105_dynamic_winners.py`, `round106_dynamic_split.py` | Winners against losers (fixed and dynamic counts) and a split set by a fast regime |
| `round107_crash_detector.py`, `round108_faster_book.py`, `h98_crash_validation.py` | Crash detectors switching to the book, faster books, and every design measured over every BTC crash |
| `h99_attribution.py`, `round109_alt_index.py` | B1's return split by sleeve; an altcoin index for the switch |
| `h100_next_day.py` | Predicting the next day's market direction (logistic, LightGBM, simple rules) and trading it daily |
| `round110_hourly_ema.py` | The speed of the bear-market switch, BTC EMAs from 12/48 to 240/960 hours |
| `round111_fast_shorts.py` | A 24/72-hour switch, and the book's shorts on very fast EMAs |
| `h101_book_fast_shorts.py` | The book alone with its shorts on 24/72-hour EMAs (the configuration deployed on 7 October) |
| `h102_book_fast.py` | The book alone on 24/72-hour EMAs for longs and shorts alike |
| `h62_attention.py` | Retail attention (Wikipedia page views): 14 strategies |
| `h68_lock_in.py` | Locking in a 14-day window's gains, the competition's horizon |
| `h69_trend_exit.py` | Do indicators, votes or machine learning tell when a trend is over? (rounds 69 and 70) |
| `h70_secure_profits.py` | Securing profits within the competition's 14-day window without a fixed target |
| `h71_timing.py` | Is there a right time to be short? Fast down-signals against the next days' returns |
| `h72_swings.py` | Trading the account's swings: do its moves revert, and what would selling the tops have made? |
| `h73_swing_sleeve.py` – `h75_gated_short_swings.py` | A swing sleeve beside the bot, long and short (shorts also only in downtrends): where its capital comes from, how much, which setups |

The scripts read cached data under `data/` (git-ignored) and write results under
`runs/research/`.
