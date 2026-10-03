# Roostoo trading bot

An autonomous trend-following bot for the [Roostoo](https://github.com/roostoo/Roostoo-API-Documents)
mock exchange, built by Team124 for the HK vs AU vs IN Quant Trading Hackathon. It trades
through the Roostoo REST API with no manual intervention: every order comes from the
strategy, and every order, decision and hourly equity point is recorded.

- [Strategy](#strategy)
- [Which assets it trades](#which-assets-it-trades)
- [Strategy research](#strategy-research)
- [How it works](#how-it-works)
- [Backtest results](#backtest-results)
- [Setup and running](#setup-and-running)
- [Configuration](#configuration)
- [Records for judging](#records-for-judging)
- [Tests](#tests)

## Strategy

Two books share the account, trading the 45 most traded crypto pairs on Roostoo plus PAXG
(gold-backed):

- **Momentum rotation, 40% of equity.** Holds the 2 coins with the strongest positive
  2-week return, re-chosen daily at 00:00 UTC, while BTC's 168-hour EMA is above its
  672-hour EMA, and leaves at once when it is not. This is the return engine.
- **Defensive trend book, 60% of equity.** Low-volatility coins in uptrends, with a market
  regime filter, trailing stops and a drawdown brake. This is the risk engine.

The two books' daily returns are barely correlated (0.18), so together they keep most of the
rotation's upside with a much smaller drawdown. The competition ranks on return and then
scores `0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar`.

The defensive book's rules:

| Rule | Detail |
|---|---|
| Regime | BTC above its 200-hour EMA: up to 75% invested in up to 4 positions. Otherwise up to 25% in up to 3, with PAXG first in line. |
| Trend filter | A coin is eligible while its 50-hour EMA is above its 200-hour EMA and the close is above the 200-hour EMA. |
| Ranking | Lowest volatility of hourly returns over the last 168 hours first (the low-volatility effect; see [Strategy research](#strategy-research)). Free slots go to the best-ranked eligible coins. |
| Entry timing | No new entry while RSI(14) is above 70, or within 24 hours of a stop-loss exit on that coin. |
| Sizing | Weights proportional to `price / ATR(14)`, so each position carries similar risk, scaled to the exposure limit and capped at 15% per coin. |
| Exits | The 50-hour EMA falls below the 200-hour EMA, or the close drops 8 ATR below the highest close since entry. A held coin is never sold just for ranking lower. |
| Drawdown brake | At 4% below its peak, the defensive book's trend positions are halved until it is back within 2%. The brake follows the defensive book's own value (tracked in the saved state), so swings in the rotation book do not trigger it. |
| PAXG core | 5% of equity stays in PAXG at all times. |
| Short sleeve | Optional, off by default: short the 3 most volatile coins (15% of the defensive book, inverse-ATR sizing) with a trailing stop 10 ATR above the lowest close since entry. Never shorts PAXG or a coin either book holds. |
| Activity rule | The competition requires trades on at least 8 days. If nothing has filled in the current 8-hour UTC block and less than 2 hours of it remain, the position furthest from its target is rebalanced. This guarantees at least 2 trades in every calendar day, whatever time zone is used. |

Trades are only placed when a holding is more than 4% of equity away from its target
(entries and exits always go through), which keeps fees down.

## Which assets it trades

The bot trades the 45 crypto pairs with the highest USD trading volume over the last 30
days, among Roostoo pairs with a bid-ask spread of at most 0.1% and at least 1000 hours of
price history, plus PAXG. As of 2 October 2026 that is:

> BTC, ETH, ZEC, SOL, XRP, NEAR, BNB, SUI, DOGE, UNI, ENA, AVAX, WLD, LINK, ADA, TAO, ARB,
> PUMP, LTC, TRX, ONDO, XLM, TRUMP, HBAR, AAVE, FIL, XPL, FET, ASTER, PENGU, DOT, APT, ICP,
> POL, CAKE, SEI, ZEN, TUT, VIRTUAL, PENDLE, CRV, FORM, EIGEN, WIF, PLUME, and PAXG

The list was 20 coins until a six-year test (see [Out-of-sample validation](#out-of-sample-validation-across-six-years))
showed the rotation book needs a wider net to catch the market's leaders.

The list lives in `bot/config.py`. `python -m bot.universe` ranks every candidate by the
rule and prints a new list. Tokenised stocks are left out: their history is short and their
pricing outside US market hours is unknown. Coins such as PEPE, SHIB and BONK are left out
by the spread limit, because their coarse price steps make every trade cost 0.15% or more.

**Why a rule instead of a hand-picked list.** The project brief proposed 12 coins. Broken
down by coin, the backtest's whole profit for October 2025 to October 2026 came from ZEC,
which rose about 20-fold that year; without ZEC the same strategy lost 2.6%. A list chosen
after the fact flatters the backtest. The volume rule uses only what was known at the time,
so the backtest applies it as of the first day of each test window:

| Universe (market-order fees) | Oct 2025 – Oct 2026 | Oct 2024 – Oct 2025 |
|---|---|---|
| Brief's 12 coins (chosen with hindsight) | +19.8% | +27.0% |
| Brief's 12 without ZEC | -2.6% | +33.4% |
| BTC, ETH, SOL, BNB, XRP and PAXG | -5.9% | +15.4% |
| Top 12 by volume at the start | -2.6% | +21.6% |
| **Top 20 by volume at the start** | **+6.6%** | **+30.2%** |
| All 36 long-listed candidates | +4.7% | +17.6% |

These comparisons used the 36 candidates with two years of history and two risk-off
positions. The top 20 did best in both years, and allowing three risk-off positions instead
of two then improved every universe in both years. The final results, with every Roostoo
pair as a candidate, are under [Backtest results](#backtest-results).

### How the parameters were chosen

The project brief's starting point used EMA 20/100, 24h/72h momentum, a 2.5 ATR stop, and
re-ranked held coins every hour. In backtests it churned: positions were held for a median
of 6 hours, turnover was about 51% of equity per day, and it lost money after fees. Each
change below was kept only if it improved results on **both** test years:

| Version | Oct 2025 – Oct 2026 | Oct 2024 – Oct 2025 |
|---|---|---|
| Brief baseline | -8.1% return, 24.3% max drawdown | +3.3%, 18.2% |
| No exits for losing rank | +0.2%, 15.6% | +8.5%, 18.7% |
| + 6 ATR stop, EMA 50/200 | +14.7%, 14.0% | +17.3%, 15.6% |
| + 8 ATR stop, 72h/168h momentum, 4% rebalance threshold | +19.8%, 15.5% | +27.0%, 14.1% |

These runs used the brief's 12 coins, and assume every order is a market order (0.1% fee
plus slippage). Removing the drawdown brake raised return slightly but pushed max drawdown
to 24–27%, so the brake stays. With the volume-based universe, more positions (5 or 6) and
other stop widths did no better in both years; three risk-off positions did.

### Known weaknesses

- **Survivorship bias.** Backtests can only use coins Roostoo lists today, which over-represents
  coins that did well. A coin the rotation bought in 2021 or 2023 that later collapsed and was
  delisted is missing, and the wider the coin list, the more such coins are missing. The
  six-year results therefore overstate what to expect, the early years most.
- It still loses money in a crash: -40% in October 2021 to October 2022 (BTC -56%). The
  worst drawdown in six years was 42%.
- With 20 coins it lagged BTC in BTC-led rallies (October 2022 to October 2024); with 45 it beat
  BTC in every year tested, but that came from the wider coin list, chosen after seeing those
  years, so treat it as promising rather than proven.
- The rotation book alone draws down 60–70%; the blend relies on the defensive book and the
  low correlation between them.
- Over any 14-day window (the length of the competition) the median backtest return is
  close to zero; the worst 14-day loss was about 17%.
- Results move by several points with small changes to the coin list, so treat any single
  backtest figure as rough.
- The coins are highly correlated, so several positions can behave like one.
- Backtests use Binance candles. Roostoo's prices are streamed from Binance and were
  within a fraction of a percent of them when checked, but fills on Roostoo are not
  guaranteed to match.

## Strategy research

The `research/` folder holds the hypothesis tests behind the current strategy
(`pip install -r requirements-research.txt` to run them). Development used October 2024 to
June 2026. **July to September 2026 was held out**, untouched until the finalists were
chosen, then run once. Every comparison used point-in-time coin lists and walk-forward
models (each month predicted by a model trained only on earlier data).

**H1. Which signals predict the next day?** (`research/h1_signals.py`) Daily rank
correlation (IC) between each signal and the next 24-hour return across the coins:

| Signal | Oct 24 – Sep 25 | Oct 25 – Jun 26 |
|---|---|---|
| Low 168h volatility | **+0.076** (t 3.0) | **+0.094** (t 3.7) |
| Liquidity (30-day USD volume) | +0.049 (t 4.2) | +0.047 (t 2.9) |
| Close to the 168h high | +0.037 | +0.073 (t 3.4) |
| 72h / 168h momentum (the earlier ranking) | -0.033 / -0.014 | +0.018 / +0.041 |
| BTC trend, for market timing | -0.03 | -0.02 |

The momentum ranking the bot used had no reliable power to pick the better coin; low
volatility did, consistently.

**H2. Does machine learning rank coins better?** (`research/h2_ml.py`) Ridge regression and
gradient-boosted trees on 20 ranked features, retrained monthly, out-of-sample IC with the
next 24h / 72h return:

| Model | Oct 24 – Sep 25 | Oct 25 – Jun 26 |
|---|---|---|
| Gradient-boosted trees | 0.100 / 0.105 | 0.086 / 0.102 |
| Ridge regression | 0.083 / 0.102 | 0.087 / 0.133 |
| Fixed blend (low vol, liquidity, near high, 2-week momentum) | 0.068 / 0.103 | 0.096 / 0.150 |
| Low volatility alone | 0.077 / 0.098 | 0.094 / 0.149 |

ML matched a single robust signal but did not beat it, so it does not earn its complexity.

**H3. Does convex optimisation build better portfolios?** (`research/sim.py`) Mean-variance
optimisation (maximise expected return minus risk and turnover penalties, subject to
long-only, 15% per coin, the regime's exposure limit and the PAXG core), solved with an
accelerated projected-gradient method checked against cvxpy. Hourly re-optimisation churned
(12–23 trades a day); daily re-optimisation with a 1% daily volatility target was the best
version, with the smallest drawdowns (8–10%) but much lower returns than simple top-N
selection, and it earned 0.2% in the hold-out quarter. Not adopted.

**H4. Signals and construction in full simulations.** Same rules as the bot, market-order
fees, development periods:

| Ranking, top-N | Oct 24 – Sep 25 | Oct 25 – Jun 26 | Composite |
|---|---|---|---|
| Momentum (earlier bot) | +22.0%, drawdown 21.7% | -9.9%, 15.2% | 1.53 / -1.28 |
| **Low volatility** | **+34.0%, 15.5%** | **-4.8%, 9.7%** | **2.68 / -0.72** |
| ML trees | +34.9%, 16.6% | -5.3%, 10.9% | 2.63 / -0.78 |
| Fixed blend | +30.7%, 15.1% | -9.2%, 14.6% | 2.56 / -1.25 |

Changes on top of low volatility that did **not** improve both periods: no regime filter
(much worse), 3 or 5 positions, other stop widths, 4-hourly or daily decisions, no drawdown
brake, larger positions or exposure, smaller or larger coin lists, blends with other
signals, other rebalance thresholds and PAXG core sizes. Market volatility targeting
improved both development periods slightly but not the hold-out.

**H5. Shorting in falling markets.** Shorting the most volatile coins in downtrends while
BTC was below its 30-day average turned the falling year positive (+6.3%) but cost 15
points in the rising year and 22 points in the hold-out quarter. It is insurance, not an
edge. Not adopted.

**Hold-out, July – September 2026, run once (BTC +42.8%):**

| Finalist | Return | Max drawdown | Composite | 14-day windows positive |
|---|---|---|---|---|
| Momentum ranking (earlier bot) | 34.5% | 6.8% | 14.2 | 65% |
| **Low-volatility ranking (adopted)** | **30.8%** | **5.9%** | **14.2** | **75%** |
| + volatility target | 29.3% | 5.9% | 13.5 | 75% |
| + bear-market shorts | 6.9% | 7.8% | 2.4 | 61% |
| Convex optimiser | 0.2% | 5.9% | 0.2 | 42% |

Low volatility tied momentum on the composite score in the rally while giving lower
drawdowns and more positive 14-day windows in all three periods, so it became the ranking.

### Time-series models and advanced portfolio construction

A second round tested econometric and portfolio-theory ideas, again chosen on the
development period only.

**H6–H7. Volatility forecasting** (`research/h6_volatility.py`). Each model forecast the next
24 hours' realised variance for every coin, refitted monthly:

| Model | QLIKE (lower is better) | Rank IC of low forecast vol, next 24h |
|---|---|---|
| 168h rolling variance (the bot) | 0.353 / 0.423 / 0.429 | 0.077 / 0.094 / 0.069 |
| EWMA, 24h half-life | 0.329 / 0.406 / 0.383 | 0.082 / 0.093 / 0.072 |
| GARCH(1,1), Student-t | 0.336 / 0.434 / 0.368 | 0.084 / 0.101 / 0.056 |
| GJR-GARCH(1,1,1) | 0.337 / 0.438 / 0.366 | 0.085 / 0.098 / 0.057 |
| HAR-RV (day, week, month) | **0.290 / 0.363 / 0.347** | **0.086 / 0.100 / 0.073** |

(Oct 24 – Sep 25 / Oct 25 – Jun 26 / Jul – Sep 26.) GARCH was not reliably better than the
simple estimate; HAR-RV forecast best every time. But ranking or sizing by the HAR forecast
did not improve the simulated strategy, because every model agrees on which coins are calm.

**H8. Regime switching** (`research/h8_regimes.py`). A 2-state Gaussian hidden Markov model
on BTC's 4-hour returns, filtered forward from past data only, switched 1,015 times against
689 for the 200-hour EMA rule, with no better timing. Neither switch predicts the next 1–3
days' return; the EMA rule earns its place by cutting exposure in volatile, falling spells.

**H9. Return prediction over time.** Pooled AR(7) and ARIMA(1,0,1) models had out-of-sample
R-squared between -0.02 and +0.006 and called the next day's direction 48–54% of the time.

**H10–H12. Portfolio construction** (`research/sim.py`):
- Volatility-managed exposure, in the spirit of Merton's optimal fraction `(mu - r) /
  (gamma sigma^2)` with unpredictable `mu`: invest up to 95% when forecast portfolio volatility
  is low. It lowered returns (+13% to +29% against +34% in the rising year) and deepened
  drawdowns (16–22%): in crypto, calm spells often come just before crashes.
- Equal risk contribution weights (Spinu's convex formulation, cyclical coordinate descent):
  better in one year, worse in the other.
- No-trade bands from transaction-cost theory (Janecek–Shreve asymptotics of the
  Davis–Norman problem, about 2.5% for a 15% position): better in one year, worse in the other.

**H13. Betting against beta: a short sleeve.** Low-volatility coins beat high-volatility
ones, so the same effect can be earned from the other side: short the three most volatile
coins whatever their trend, sized by inverse ATR, with a trailing stop above the lowest close
since entry. A grid over the size (10–25%) and stop (6–12 ATR) showed a clear safe zone:
13 of 20 settings beat the long-only strategy on the composite score in both development
years, and every setting cut the drawdown. Tight 6 ATR stops and 25% sizes failed. The
centre of the safe zone, 15% with a 10 ATR stop, was chosen before looking at July to
September 2026, where it returned 28.1% against 30.8% for long-only during a 43% BTC rally,
with more positive 14-day windows (80% against 75%).

In the bot's own backtester, with Roostoo's collateral rules and market-order fees:

| | Long only (default) | Long + 15% short sleeve | Hold BTC |
|---|---|---|---|
| Oct 2024 – Oct 2025 | +37.2%, drawdown 12.7%, composite 3.16 | **+35.8%, 8.5%, 3.66** | +79.5%, 30.9% |
| Oct 2025 – Oct 2026 | +12.9%, drawdown 12.3%, composite 1.26 | **+22.6%, 8.4%, 2.51** | -26.8%, 53.7% |
| Two years | +54.9% | **+66.5%** | +31.4% |

The sleeve ships switched off (`short_exposure = 0`) because Roostoo's `/v6` short endpoints
have not yet been tried on the testing account, and because on October 2023 to October 2024,
which no research had used, the defensive book with the sleeve lost 10.6% as volatile coins
squeezed higher. Turn it on with `--config config/shorts.json` once the endpoints are tried.

### Beating buy-and-hold

The strategies above beat holding BTC in falling markets but not in rallies: the bot was
on average only 22–35% invested. A third round asked what could beat holding BTC outright.

**H14. Trend timing on BTC** (`research/h14_trend.py`). Twenty-one rules (EMA crossovers,
price against an EMA, time-series momentum, breakouts), long/flat and long/short, at full
exposure. Every rule beat holding in the falling year; none beat it in the rising year. The
best, a 168/672-hour EMA crossover, kept 65% of BTC's 80% (and 115% of its 135% in
2023–24) with smaller drawdowns.

**H15. Dual momentum rotation** (`research/h15_rotation.py`). Hold the top 1–5 coins by return
over 3–30 days, if positive, rebalanced daily or weekly, with or without the BTC slow-trend
filter: 64 settings. Nine beat holding BTC in all three development periods, all with a
2- to 4-week lookback and the trend filter, but with drawdowns of 50–70%. Stops, inverse-
volatility weights and a drawdown brake cut the drawdowns only by cutting the returns
below BTC's; leaving at once when the trend filter fails helped.

**H16. A blend of books** (`research/h16_blend.py`). Rotation and the defensive book have a
daily-return correlation of 0.18. With fixed shares rebalanced daily, 40% rotation and 60%
defensive beat holding BTC in all three development periods, with a smaller drawdown than
BTC in the two big ones. In the bot's own backtester this needed two more fixes: the coin
list refreshed monthly (a list frozen on day one cost the rotation 36 points in 2024–25), and
the brake following the defensive book's own value rather than the whole account's.
Rotation became the default at 40%, chosen before the October 2023 to October 2024 check,
where, with 20 coins, it did not beat holding BTC.

### Out-of-sample validation across six years

Tuning until the one unused year looked good would only have overfitted to it. Instead,
`research/folds.py` runs whole designs, fixed in advance, through the bot's own backtester on
six one-year folds from October 2020 to October 2026: the 2020–21 bull run, the 2022 crash,
the 2023 recovery, the 2023–24 and 2024–25 rallies and the 2025–26 decline. The first three
were never used by any earlier research. The rule for choosing was fixed before running:
highest median composite score across the folds, ties to the smaller worst drawdown. The
candidate coins and their spreads are frozen in `research/candidates.csv` so the study
gives the same answer every time.

Eight designs on the 20-coin list (return per fold, from October 2020):

| Design | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median composite |
|---|---|---|---|---|---|---|---|
| 40% rotation + book + short sleeve | +722% | -28% | +6% | +34% | +89% | +64% | 2.02 |
| 60% rotation + book | +1515% | -35% | -1% | +62% | +104% | +66% | 1.73 |
| Rotation only | +3946% | -50% | -10% | +78% | +153% | +94% | 1.65 |
| 40% rotation + book (the default) | +787% | -28% | -1% | +39% | +89% | +47% | 1.60 |
| 40% rotation + momentum-ranked book | +733% | -29% | +1% | +44% | +68% | +42% | 1.50 |
| 20% rotation + book | +359% | -17% | -5% | +25% | +60% | +27% | 1.44 |
| Low-volatility book only | +117% | -10% | -11% | +2% | +30% | +16% | 0.89 |
| Momentum-ranked book only | +105% | -15% | -8% | +7% | +19% | +13% | 0.88 |

The pattern held across all six years, which is what a strategy that is not overfitted looks
like: it beat holding BTC when the market fell or when altcoins boomed, and lagged when BTC
alone led a steady rally (2022–24). Keeping part of the rotation book in BTC did not fix that
(it improved one weak year and hurt others).

The fix came from the coin list. The rotation book had only 20 coins to choose leaders from.
A second set of designs, also fixed before running, varied how wide the net is, charging
coins with wide spreads half their spread on every trade:

| Coins to choose from | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median composite | Beats BTC |
|---|---|---|---|---|---|---|---|---|
| **Top 45 crypto, spread at most 0.1%** | +537% | -40% | +42% | +236% | +150% | +30% | **2.59** | **6/6** |
| Top 30 crypto, spread at most 0.1% | +537% | -40% | +42% | +226% | +127% | +45% | 2.45 | 6/6 |
| Top 60 of all 86, spread at most 0.5% | +539% | -27% | +39% | +299% | +115% | +0% | 2.19 | 6/6 |
| Top 30 crypto, spread at most 0.3% | +539% | -27% | +43% | +244% | +97% | +31% | 2.06 | 6/6 |
| Top 20 crypto (the earlier list) | +786% | -28% | -1% | +39% | +89% | +43% | 1.51 | 4/6 |
| Top 20 crypto and tokenised stocks | +786% | -28% | -1% | +39% | +89% | +27% | 1.24 | 4/6 |

The top 45 won under the rule and became the list. Its 2023–24 profit was spread across many
coins (FLOKI, PENDLE, WLD, SUI, AVAX, SEI and FET each made $17k–$81k on $100k, while ENA, TAO,
ICP and WIF lost $15k–$59k), and no single-hour price move in those coins exceeded 21%, so no
bad candle drove it. Tokenised stocks did not help.

### Convex optimisation of the rotation book (round 5)

All six folds had now been seen, so the bar for change rose: a design written down in advance
replaces the current one only with a higher median composite, a higher composite in at least
4 of the 6 folds, and a worst drawdown no more than 2 points worse. Round 5 asked how the
rotation book should weight its momentum picks, using plain-Python solvers in `bot/optimize.py`
(checked against cvxpy to within 0.003%):

| Rotation book | Median composite | Folds improved | Worst drawdown | Six years |
|---|---|---|---|---|
| **Top 2, equal weights (kept)** | **2.59** | – | 42% | **+5,871%** |
| Top 5, equal weights | 2.42 | 2 of 6 | 40% | +4,040% |
| Top 5, equal risk contribution | 1.92 | 2 of 6 | 39% | +2,858% |
| Top 5, inverse volatility | 1.89 | 2 of 6 | 38% | +2,874% |
| Top 5, minimum variance, 40% cap | 1.56 | 2 of 6 | 34% | +2,475% |

The rotation book earns its return by concentrating on the strongest leaders. Spreading it over
five coins, and above all weighting by risk, trimmed the drawdown but cut returns in four of six
years: minimum variance favours the calmest of the leaders, which are the weakest movers. None
met the bar. The options stay in the code, off (`rotation_weighting`, `rotation_max_weight`).

## How it works

```
bot/
  config.py        every tunable number, with JSON overrides
  roostoo.py       API client: signing, clock offset, rate limit, retries, request log
  market_data.py   Binance hourly candles (live signals, warm-up, backtest cache)
  universe.py      which pairs to trade: the most traded crypto pairs on Roostoo
  indicators.py    EMA, ATR, RSI, rolling volatility, updated one bar at a time
  strategy.py      target weights from signals and current holdings
  planner.py       trades from targets: rebalance threshold, activity rule, sells
                   anything outside the universe
  execution.py     orders on Roostoo: rounding, limit-then-market, fills
  live.py          the hourly loop, state recovery, dry-run mode
  journal.py       logs, append-only trade and equity records, saved state
  backtest.py      replays the same strategy and planner on Binance candles
  metrics.py       return, drawdown, Sharpe, Sortino, Calmar, composite score
scripts/run_bot.sh restarts the bot whenever it exits (run it inside tmux)
tests/             unit tests, plus end-to-end runs against a simulated exchange
```

Every hour, a minute after the candle closes, the live bot:

1. Fetches the last 2500 closed hourly candles per pair from Binance and rebuilds the
   indicators. Roostoo has no history endpoint, but its prices are streamed from Binance.
   If Binance is unreachable, it uses its cached candles plus hourly bars built from
   Roostoo ticker samples taken every 5 minutes.
2. Reads the Roostoo wallet and ticker, and values the portfolio at the last price.
3. Asks the strategy for target weights and the planner for trades.
4. Sells first, then buys with the cash actually available. Each order rests as a limit
   order at the best bid or ask (0.05% maker fee); anything unfilled after 5 minutes is
   cancelled and sent at market (0.1%). Stop-loss exits go straight to market.
5. Reconciles the strategy state with the new wallet, records everything, and saves
   its state.

The bot never assumes it starts flat. On start-up and before each cycle it cancels any
order left open by a crashed run, and it rebuilds positions from the real wallet. Trailing
stops, cooldowns and the drawdown brake survive restarts through `state.json`; the peak
equity is also recovered from the equity journal.

The API client stays under 20 calls per minute (the organisers' limit is 30), retries
network errors and 5xx responses with backoff, and never retries an order placement,
since a timed-out order may still have gone through.

## Backtest results

```
python -m bot.backtest                                   # default window: 2025-10-01 to 2026-10-01
python -m bot.backtest --start 2024-10-01 --end 2025-10-01
```

The first run downloads candles into `data/`; later runs read the cache. The backtest runs
the same `Strategy` and `plan_trades` code as the live bot, in two cost scenarios: every
order as a taker (0.1% fee plus 0.02% slippage) and every order as a maker (0.05%). Live
costs fall between the two. The coins are chosen by the volume rule as of the first day of
the window (`--fixed-universe` tests the configured list instead).

$100,000 starting cash, market-order fees (plus half the spread on coins with wide spreads),
the coin list refreshed by the volume rule on the first day of every month. Each column is a
year starting in October.

| | 2020–21 | 2021–22 | 2022–23 | 2023–24 | 2024–25 | 2025–26 | Six years |
|---|---|---|---|---|---|---|---|
| **Bot (45 coins, 40% rotation)** | **+537%** (25%) | **-40%** (42%) | **+42%** (30%) | **+236%** (23%) | **+150%** (31%) | **+30%** (22%) | **+5,871%** |
| Bot with 20 coins (the earlier list) | +786% (21%) | -28% (35%) | -1% (30%) | +39% (41%) | +89% (33%) | +43% (24%) | +2,258% |
| Defensive book alone | +117% (16%) | -10% (16%) | -11% (14%) | +2% (13%) | +30% (14%) | +16% (10%) | +167% |
| Hold BTC | +306% (55%) | -56% (74%) | +39% (27%) | +135% (32%) | +80% (31%) | -27% (54%) | +674% |

Maximum drawdown in brackets. The bot beat holding BTC in all six years with a smaller
drawdown than BTC in five; its median composite score across the years was 2.59. Read the
[survivorship warning](#known-weaknesses) before trusting the size of these numbers.

Sharpe and Sortino use daily returns annualised over 365 days; Calmar is annualised
return over maximum drawdown. The organisers have not published their exact conventions.
The full report also shows statistics over every rolling 14-day window and monthly returns.
Results are written to `runs/backtest/`.

## Setup and running

Requirements: Python 3.9 or later and `requests` (`pip install -r requirements.txt`).
Nothing else; the code uses only the standard library otherwise.

### Credentials

The bot reads its keys only from the environment variables `ROOSTOO_API_KEY` and
`ROOSTOO_SECRET_KEY`. On the server they live in one file per account, outside the repo:

```
~/.roostoo_test.env    # testing account
~/.roostoo_comp.env    # competition account
```

each containing:

```
export ROOSTOO_API_KEY=...
export ROOSTOO_SECRET_KEY=...
```

Keys never appear in code, config, logs or commits. The API log records parameters and
responses but never headers, which is where the key and signature travel.

### On the AWS server

```
git clone https://github.com/samargoyal/roostoo-trading-bot.git
cd roostoo-trading-bot
pip3 install --user -r requirements.txt

tmux new -s bot
scripts/run_bot.sh test --dry-run    # check one account end to end without trading
scripts/run_bot.sh comp              # the competition account
```

Detach from tmux with `Ctrl-b d`; reattach with `tmux attach -t bot`.
`run_bot.sh` restarts the bot a minute after it exits for any reason.

To deploy new code during the competition, commit and push it, then on the server:

```
git pull && pkill -f "bot.live --account comp"
```

The restart loop starts the new code within a minute, and the bot carries on from the
wallet and its saved state.

### Running directly

```
python -m bot.live --account test --dry-run   # everything except sending orders
python -m bot.live --account test --once      # a single cycle, then exit
python -m bot.live --account test             # trade every hour
```

`--account` names the folder under `runs/` that holds that account's logs, journal and
state; dry runs use `runs/<account>-dry/` so they never mix with real records.

## Configuration

All parameters live in [`bot/config.py`](bot/config.py), with their defaults and a comment
for each. To change some without editing code, pass a JSON file with only the values to
override, to either the bot or the backtest:

```
{"strategy": {"max_positions_risk_on": 5}, "execution": {"limit_timeout_sec": 600}}
```

```
python -m bot.backtest --config my_settings.json
python -m bot.live --account test --config my_settings.json
```

Unknown keys are rejected, so a typo cannot silently fall back to a default.

`config/shorts.json` turns on the short sleeve (see [Strategy research](#strategy-research)):

```
scripts/run_bot.sh comp --config config/shorts.json
```

With the sleeve on, the bot reads `/v6/short_positions` every cycle, values each short at its
collateral plus unrealised profit, opens shorts with `/v6/short_open` (market, collateral
taken from free cash alongside buys) and covers with `/v6/short_close`.

## Records for judging

Each account keeps these under `runs/<account>/`:

| File | Contents |
|---|---|
| `journal/orders.csv` | Every order: pair, side, type, quantity, limit price, status, filled quantity, average price, fee, and the strategy reason (`entry`, `exit_trend`, `exit_stop`, `rebalance`, `activity`, ...) with the current and target weight that triggered it. |
| `journal/decisions.jsonl` | One line per hour: the signals for every pair, ranking scores, regime, brake state, target weights and planned trades. |
| `journal/equity.csv` | Portfolio value, cash, positions and drawdown after every hourly cycle. |
| `logs/bot.log` | What the bot did and why. Rotated. |
| `logs/api.log` | Every API request and response, without keys. Rotated. |
| `state.json` | What the strategy remembers between restarts. |

Together these show that every trade follows from the strategy's logged signals.

## Tests

```
python -m unittest discover -s tests -t .
```

The tests cover request signing (against the example in the Roostoo docs), the rate
limiter, response handling, rounding to each pair's precision, position sizing, every
strategy rule, the activity rule and the metrics. End-to-end tests drive the live loop
against a simulated exchange: a dry run, real fills, a restart after a crash with an order
left open, and a Binance outage.
