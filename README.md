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

- **Momentum rotation, 70% of equity.** Holds the 2 coins with the strongest positive
  2-week return, re-chosen daily at 00:00 UTC, while BTC's 168-hour EMA is above its
  672-hour EMA, and leaves at once when it is not. This is the return engine.
- **Defensive trend book, 30% of equity.** Low-volatility coins in uptrends, with a market
  regime filter, trailing stops and a drawdown brake. This is the risk engine.

The split was 40/60 until 4 October 2026. An optimisation of the rotation's share in steps of
5% (see [The rotation's share](#the-rotations-share-optimised)) gave 50% under a 50% drawdown
limit; the user then chose 70%, more return for more risk, as the competition ranks on
return first.

The two books' daily returns are barely correlated (0.18), so together they keep most of the
rotation's upside with a much smaller drawdown. The competition ranks on return and then
scores `0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar`.

The defensive book's rules:

| Rule | Detail |
|---|---|
| Regime | BTC above its 200-hour EMA: up to 75% invested in up to 8 positions. Otherwise up to 25% in up to 3, with PAXG first in line. |
| Trend filter | A coin is eligible while its 50-hour EMA is above its 200-hour EMA and the close is above the 200-hour EMA. |
| Ranking | Lowest volatility of hourly returns over the last 168 hours first (the low-volatility effect; see [Strategy research](#strategy-research)). Free slots go to the best-ranked eligible coins. |
| Entry timing | No new entry while RSI(14) is above 70, or within 24 hours of a stop-loss exit on that coin. |
| Sizing | Equal risk contribution: weights at which every position adds the same share of the book's variance, given how the coins move together over the last 336 hours. A convex problem, solved in `bot/optimize.py`. Scaled to the exposure limit and capped at 15% per coin. |
| Exits | The 50-hour EMA falls below the 200-hour EMA, or the close drops 8 ATR below the highest close since entry. A held coin is never sold just for ranking lower. |
| Drawdown brake | At 4% below its peak, the defensive book's trend positions are halved until it is back within 2%. The brake follows the defensive book's own value (tracked in the saved state), so swings in the rotation book do not trigger it. |
| PAXG core | 5% of equity stays in PAXG at all times. |
| Short sleeve | Optional, off by default: short the 3 most volatile coins (15% of the defensive book, inverse-ATR sizing) with a trailing stop 10 ATR above the lowest close since entry. Never shorts PAXG or a coin either book holds. |
| Activity rule | The competition requires trades on at least 8 days. If nothing has filled in the current 8-hour UTC block and less than 2 hours of it remain, the position furthest from its target is rebalanced. This guarantees at least 2 trades in every calendar day, whatever time zone is used. |
| Halted pairs | A pair Roostoo stops trading gets no orders. It is held as it is and never bought, and the defensive book is re-solved around it: equal risk contributions with its weight fixed, counting how it moves with the other coins. See round 11. |

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
- It still loses money in a crash: -56% in October 2021 to October 2022 (BTC -56%). The
  worst drawdown in six years was 58%.
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

### Convex optimisation of the defensive book (round 6)

The defensive book held 4 coins capped at 15% each, so the caps, not any optimiser, set its
weights. Round 6 let it hold up to 8 and compared optimisers, plus other splits between the
books. Composite score per fold:

| Design | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median | Folds better |
|---|---|---|---|---|---|---|---|---|
| 4 coins, inverse ATR (incumbent) | 9.78 | -1.99 | 1.46 | 5.92 | 3.72 | 1.30 | 2.59 | – |
| **8 coins, equal risk contribution** | 11.32 | -1.91 | 1.33 | 5.65 | 4.30 | 1.44 | **2.87** | **4/6** |
| 8 coins, minimum variance | 11.40 | -1.95 | 1.38 | 5.38 | 3.99 | 0.86 | 2.69 | 3/6 |
| 50% rotation / 50% book | 10.60 | -2.03 | 1.33 | 6.34 | 3.71 | 1.41 | 2.56 | 3/6 |
| 30% rotation / 70% book | 8.92 | -1.96 | 1.31 | 5.42 | 3.56 | 1.55 | 2.56 | 2/6 |

Equal risk contribution met every condition (higher median, better in 4 of 6 folds, worst
drawdown unchanged at 42%) and became the default. Minimum variance concentrated in the very
calmest coins and deepened the worst drawdown to 45%.

### Ideas from VECM-ARB

[VECM-ARB](https://github.com/samargoyal/VECM-ARB) trades cointegrated baskets of bank stocks:
Johansen tests and a vector error-correction model find the equilibrium spread, z-score
thresholds trade its reversion, a rolling Johansen test liquidates when the relationship
breaks, and a cvxpy minimum-variance hedge rebalances it. Its core, a long-short basket
trading a spread back to equilibrium, is a form of statistical arbitrage; the competition bans
"arbitrage", and whether that covers spread trading has not been confirmed. Round 10 tested
the whole design on crypto anyway. First, two of its ideas that fit a directional bot were
tested on the rotation book: hedging out the common factor before ranking (residual momentum,
the coin's return net of its beta to BTC), and its z-score entry threshold (skip a pick more
than 2 standard deviations above its one-week mean, the repo's own `entry_z = 2.0`).

### Rounds 7 and 8: VECM-ARB ideas and tail-risk control

Against the round-6 book, with the same rule (composite per fold, from October 2020):

| Design | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median | Folds better |
|---|---|---|---|---|---|---|---|---|
| **8-coin ERC book (kept)** | 11.32 | -1.91 | 1.33 | 5.65 | 4.30 | 1.44 | **2.87** | – |
| Control: 8 coins, inverse ATR | 11.44 | -1.92 | 1.26 | 5.55 | 4.35 | 0.96 | 2.81 | 2/6 |
| Residual momentum | 15.43 | -1.79 | 1.54 | 5.06 | 3.99 | 1.18 | 2.77 | 3/6 |
| Rotation CVaR capped at 5% of equity | 10.46 | -1.91 | 1.19 | 5.43 | 4.26 | 1.45 | 2.86 | 2/6 |
| Rotation CVaR capped at 3% of equity | 10.15 | -1.97 | 0.96 | 5.14 | 3.95 | 1.51 | 2.73 | 1/6 |
| Skip picks > 2 sd above their 1-week mean | 7.87 | -1.68 | 0.88 | 4.73 | 3.03 | 1.54 | 2.28 | 2/6 |
| Residual momentum and the z-score guard | 7.98 | -1.56 | 1.03 | 3.67 | 2.99 | 0.94 | 2.01 | 1/6 |

None met the rule. The control shows the optimiser earns its place: ERC beat inverse-ATR
sizing with the same 8 coins in 4 of 6 folds. Residual momentum was the near miss, better in
the three earliest folds (including the 2021–22 crash) and worse in the three latest. The
z-score guard skipped exactly the leaders the rotation book exists to catch. A 1-day CVaR cap
on the rotation book (the average loss on its worst 5% of days, from recent hourly returns)
mostly cut returns. All remain options in the code, off.

### Round 9: how robust are the rotation book's settings?

Each setting moved either side of its default, which was chosen early on 20 coins and two
years (composite per fold, from October 2020):

| Change | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median | Folds better | Worst drawdown |
|---|---|---|---|---|---|---|---|---|---|
| **Defaults (kept)** | 11.32 | -1.91 | 1.33 | 5.65 | 4.30 | 1.44 | **2.87** | – | 42% |
| Plus the 15% short sleeve | 8.84 | -2.04 | 1.19 | 5.55 | 5.25 | 1.61 | 3.43 | 2/6 | 41% |
| Top 3 coins | 15.19 | -1.35 | 1.15 | 5.36 | 4.10 | 2.23 | 3.16 | 3/6 | 38% |
| Trend filter 336h/1344h | 12.04 | -1.44 | 0.48 | 4.17 | 5.06 | 1.46 | 2.81 | 4/6 | 36% |
| Trend filter 72h/288h | 9.08 | -2.43 | 2.39 | 3.19 | 2.72 | 0.93 | 2.56 | 1/6 | 54% |
| Rebalance every 72h | 11.43 | -1.13 | 1.04 | 3.38 | 3.41 | 1.59 | 2.48 | 3/6 | 38% |
| Lookback 720h | 7.92 | -2.03 | 1.76 | 3.33 | 2.38 | 1.45 | 2.07 | 2/6 | 45% |
| Lookback 504h | 11.53 | -1.32 | 1.79 | 4.18 | 2.32 | 1.68 | 2.06 | 4/6 | 38% |
| Lookback 168h | 16.50 | -1.91 | 1.83 | 4.71 | 1.62 | 1.68 | 1.76 | 3/6 | 48% |
| Top 1 coin | 7.20 | -2.22 | 1.26 | 7.89 | 2.07 | 1.37 | 1.72 | 1/6 | 54% |

Nothing met the rule. Two lessons. The fold-count condition matters: the short sleeve has a
higher median only because the median of six numbers averages the middle two, while it was
worse in four of the six folds. And the 336-hour lookback sits on a peak rather than a plateau:
its neighbours score around 1.8–2.1. Every variant still beat holding BTC in five or six of
the six years, so the design holds up, but expect live results nearer the neighbours than
the defaults' 2.87.

### Round 10: VECM-ARB's three engines, tested on crypto

`research/h18_vecm.py` ports the whole of VECM-ARB:

1. **N-dimensional alpha engine.** At each re-fit, a Johansen trace test on the log prices of
   the 6 most traded coins gives the cointegration rank r, and a VECM at that rank gives the
   hedge ratios (β) and the speeds of adjustment (α). The spread β′ log p is traded on its
   z-score with VECM-ARB's own rules: enter at |z| ≥ 2, take profit at |z| ≤ 0.5, and leave
   early if the spread's slow mean drifts against the position.
2. **Cointegration breakdown protocol.** At every re-fit the Johansen test is re-run on the
   traded basket's trailing window; rank 0 liquidates the spread.
3. **Dynamic hedging.** If a leg cannot be traded when the spread is entered (halted, or not
   borrowable for a short), a minimum-variance hedge is re-solved on the other N−1 coins,
   keeping each one's side and a $0.5 long / $0.5 short book (cvxpy, as in VECM-ARB's phase 4).

VECM-ARB's tear sheet (Sharpe 1.02) estimates β on all of 2020–2026 and backtests on the same
data, so it is in-sample. Here each β is estimated only from data before it trades. Every leg
pays the taker fee and half its spread both ways, and shorts are 1x as on Roostoo. Three
designs were fixed before running: **A**, VECM-ARB's own settings on daily closes (a 500-day
window re-fitted every 30 days, z-score over 25 days); **B**, the same logic on hourly closes
(a 30-day window re-fitted weekly, z-score over 72 hours); **C**, B entering only when α and
β give the spread a half-life under a week.

The spread trader alone (return per fold, maximum drawdown in brackets):

| Design | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Cointegrated re-fits | Trades |
|---|---|---|---|---|---|---|---|---|
| A: daily, VECM-ARB settings | -4% (9%) | -18% (26%) | +0% (0%) | -9% (10%) | -7% (12%) | +2% (5%) | 34 of 78 | 66 |
| B: hourly | -2% (7%) | -28% (29%) | -38% (38%) | -9% (15%) | -16% (20%) | -15% (15%) | 106 of 318 | 642 |
| C: hourly, half-life under a week | -2% (6%) | -21% (24%) | -35% (35%) | -9% (15%) | -16% (20%) | -15% (15%) | 106 of 318 | 608 |
| A with no costs at all | -3% | -16% | +0% | -9% | -5% | +5% | | |
| B with no costs at all | +4% | -14% | -26% | +1% | -6% | -0% | | |

As a 20% sleeve beside the current bot, all three lowered the median composite (A 2.75,
better in 2 of 6 folds; B and C 2.58, better in none; the bot alone 2.87). What the test
shows:

- **Cointegration is there but does not last.** A third to almost half of the re-fits found
  rank 1 or more, far more than the 5% a 95% test finds by chance. But the relationships did
  not hold over the following days and weeks: the spreads drifted rather than reverted. Even
  with no fees at all every design lost money on average (A -4.6% a year, B -7.1%, C -5.8%);
  fees turn that into -6% to -18%.
- **The breakdown protocol works, but cannot save it.** With it switched off, design B loses
  -18.9% a year instead of -18.0%; it helped most in 2023–24 (-9% instead of -15%).
- **The N−1 re-hedge works.** With the largest short leg frozen at every entry, simply
  dropping that leg pushed the open position's volatility from 25% to 45% a year, while the
  re-hedge held it at 28%. There was just no profit to protect.

The spread trader is not used. VECM-ARB's re-hedging idea is, in the form that suits this
bot: when Roostoo halts a coin, the bot plans around it (round 11).

### Round 11: halted coins (VECM-ARB's dynamic risk engine, adapted)

Roostoo can stop trading a pair (`CanTrade` false in its exchange info). Until now the bot
refused to start if any of its 46 pairs was halted (under `run_bot.sh` it would have restarted
and failed indefinitely), and mid-run it would have kept sending orders the exchange refused.
Worse, a refused sell could starve the activity rule, which only steps in when nothing else
is planned. Now the bot re-reads the rules every hour (one extra call), and the planner leaves
halted pairs out before the activity rule looks for a trade.

VECM-ARB's answer to a frozen leg is to re-optimise the rest of the book around it. The bot's
version (`strategy.plan_around_halts`) holds a halted coin as it is, never buys it, gives its
slot to the next-best coin, and re-solves the defensive book around it: equal risk
contributions with the halted weight fixed, counting its covariance with the other coins
(`erc_weights_fixed` in `bot/optimize.py`, a strictly convex problem whose risk budget is found
by bisection), so a coin that moves with the halted one gets less.

`research/h19_halts.py` compared that with leaving the strategy untold, under harsh halts aimed
at holdings: every week a coin the bot had just bought stops trading for 3 days, 53 halts a
year. Two schedules (the week's first purchase, or its last) show how much is luck. The rule,
set after the first schedule and before the second: tell the strategy only if that is better
in at least 6 of the 12 runs. Composite per fold, from October 2020:

| Run | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median | Better than not told |
|---|---|---|---|---|---|---|---|---|
| No halts | 11.32 | -1.91 | 1.33 | 5.65 | 4.30 | 1.44 | 2.87 | |
| Schedule 1, not told | 11.76 | -2.05 | 1.21 | 5.48 | 4.00 | 1.02 | 2.60 | |
| Schedule 1, re-solved | 10.03 | -2.16 | 1.22 | 4.68 | 4.55 | 0.75 | 2.88 | 2/6 |
| Schedule 2, not told | 12.01 | -1.98 | 1.41 | 5.81 | 4.22 | 1.49 | 2.85 | |
| Schedule 2, re-solved | 11.58 | -1.87 | 1.43 | 4.96 | 4.48 | 1.60 | 3.04 | 4/6 |

Better in 6 of 12 with a higher median in both schedules: a tie, which met the rule, so it is
on. (A first run, before the planner was fixed to leave halted pairs out, gave 5 of 12; the
difference is noise either way.) Even halts hitting a holding every week barely moved the
results. What matters is that the bot keeps running and trading when a pair is halted.

### Round 12: microstructure, volume surfaces, Bayesian methods and Kalman filters

Ideas from open-source projects on each topic, tested on every coin the bot can trade (the 45
most traded crypto pairs each month, out of the 54 with spreads of 0.1% or less). The screen
was fixed before running: a signal earns a strategy test only if its daily rank correlation
(IC) with the next 24 or 168 hours' returns across the coins has the same sign in at least 5
of the 6 folds, with a mean of at least 0.02, and keeps the same sign in at least 4 folds once
the defensive book's low-volatility ranking is partialled out. Binance's hourly candles also
record quote volume, the number of trades and the volume bought by takers, which the
microstructure signals need (`research/fullbars.py` fetches them).

`research/h20_screen.py`, daily IC per fold (24 hours / 168 hours ahead):

| Signal | What it measures | 24h IC | 168h IC | Same sign | Passes |
|---|---|---|---|---|---|
| Corwin–Schultz spread | bid-ask spread estimated from hourly highs and lows | -0.074 | -0.107 | 6/6, 6/6 | yes |
| Amihud illiquidity | price move per dollar traded | -0.037 | -0.055 | 6/6, 6/6 | yes |
| Kalman trend strength | slope of a local linear trend filter on log price, over its standard deviation | +0.014 | +0.028 | 5/6, 5/6 | yes (168h) |
| Taker flow, 168h | share of the week's volume bought by takers (order-flow imbalance) | +0.006 | +0.027 | 5/6, 6/6 | yes (168h) |
| Taker flow, 24h | the same over a day | -0.003 | +0.018 | 5/6, 5/6 | no |
| Trade size | mean dollars per trade against the past month | -0.008 | +0.003 | 6/6, 4/6 | no (too weak) |
| Abnormal volume | volume against each coin's hour-of-week volume surface | -0.012 | +0.001 | 4/6, 4/6 | no |
| Bayesian momentum | 2-week return shrunk to the cross-sectional mean (empirical Bayes) | -0.020 | -0.009 | 6/6, 4/6 | no (-0.0199) |
| Low volatility (the book's ranking) | | +0.073 | +0.101 | 6/6, 6/6 | reference |
| 2-week return (the rotation's ranking) | | -0.020 | -0.012 | 6/6, 4/6 | reference |

Coins with wide estimated spreads or little trading did worse in every fold, even after
allowing for their volatility; steady trends (judged against their own noise, as a Kalman
filter does) and a week of net buying by takers both kept going, a little. A day of order
flow did not carry over, as several projects found at shorter horizons, and neither did
volume surprises against the hour-of-week volume surface. Shrinking momentum the Bayesian way
did not help: like the raw 2-week return it slightly predicts a reversal over the next day
(the rotation book earns its money from a few large trends, not from the average coin). A
first run of the screen was made before every coin's candles had finished downloading; on
complete data the week's taker flow passes and the Kalman trend is weaker (both shown here).

Market timing (`research/h22_regimes.py`), on BTC, whose trend switches the rotation book, and
on every coin timed by its own filter (1/N each, cash when off), 0.1% per switch:

| Filter | Median composite, BTC | Better than the EMA filter | Median composite, all coins | Better than EMA |
|---|---|---|---|---|
| EMA 168h / 672h (the bot's) | 1.19 | | 0.75 | |
| Bayesian online changepoint detection, drift > 0 | 0.68 | 1/6 | 0.33 | 2/6 |
| Always in | 1.72 | 2/6 | 0.47 | 0/6 |

Bayesian online changepoint detection (Adams and MacKay, with a normal-inverse-gamma model of
daily returns) switched four times as often as the EMA filter (54 times a year against 13)
and did worse. Deribit's implied volatility (DVOL, the 30-day point of BTC's volatility
surface): the variance risk premium's correlation with BTC's next week had the same sign in
only 4 of 5 years (mean -0.05), short of the bar.

### Round 13: the signals that passed, in the strategy

The signals that passed the screen were put into the bot (`bot/indicators.py`:
`KalmanTrend`, `SpreadEstimate`) and used in the two rankings, fixed before any fold run
(from the first, incomplete screen, so without taker flow, which round 15 tests):
**C1** ranks the defensive book by the sum of normal scores for low volatility, a narrow
estimated spread and Kalman trend strength (Amihud illiquidity repeats the spread's
information, so it was left out); **K1** picks the rotation book's coins by Kalman trend
strength instead of the 2-week return; and C1 and K1 together. Same rule as rounds 5 to 9.
Composite per fold, from October 2020:

| Design | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median | Folds better | Worst drawdown |
|---|---|---|---|---|---|---|---|---|---|
| **Current (kept)** | 11.32 | -1.91 | 1.33 | 5.65 | 4.30 | 1.44 | **2.87** | – | 42% |
| C1 composite book ranking | 10.51 | -2.06 | 1.26 | 5.59 | 4.55 | 1.10 | 2.91 | 1/6 | 45% |
| K1 rotation by Kalman trend | 8.68 | -0.35 | 0.69 | 3.15 | 2.44 | 2.47 | 2.46 | 2/6 | 30% |
| C1 and K1 | 9.14 | -0.45 | 0.57 | 2.94 | 2.14 | 2.13 | 2.14 | 2/6 | 31% |

None met the rule. C1's higher median is the median-of-six effect again: it was worse in five
of the six years. K1 is a real trade-off rather than an improvement: choosing the steadiest
trends instead of the biggest movers cut the worst drawdown from 42% to 30% and the 2021–22
crash from -39% to -9%, but gave up most of the bull years (2023–24: +76% against +229%), and
its median 14-day composite was half the current bot's. The competition ranks by return
first, so that trade is the wrong one here. Predicting the average coin's next day was not
what the strategy needed: the rotation book earns its money from a few very large moves, and
the defensive book uses its ranking only to choose new entries. Both rankings stay in the
code as options (`ranking: "composite"`, `rotation_ranking: "kalman"`), off.

### Round 14: neural networks and deep learning

`research/h21_learning.py` asks whether learned models rank coins better than low volatility
does. Every day at 00:00 UTC four models score every coin in the month's universe on its next
24 hours relative to the others, from 24 features (the H1 set and the microstructure signals
above), each turned into a normal score of its rank; the GRU also reads each coin's last 72
hourly bars (return, volume surprise, taker flow). All are retrained every quarter on earlier
days only, walk-forward from June 2020, with a day's gap so no target overlaps.
[derinteke/crypto-cross-sectional-forecasting](https://github.com/derinteke/crypto-cross-sectional-forecasting)
found that a GRU and a Transformer added nothing over gradient-boosted trees; the same
question on these coins:

| Model | Mean daily IC | Same sign | Daily top-8 portfolio, median composite |
|---|---|---|---|
| Ridge regression | 0.079 | 6/6 | 0.32 |
| Gradient-boosted trees | 0.078 | 6/6 | 0.14 |
| MLP neural network (64-32) | 0.058 | 6/6 | -0.04 |
| GRU recurrent network | 0.078 | 6/6 | 0.29 |
| Low volatility alone (the book's ranking) | 0.073 | 6/6 | 0.61 |

All four passed the screen, but they mostly rediscovered the low-volatility effect: the deep
network only matched the linear model, and no model's daily top 8 beat simply holding the 8
calmest coins. By the rule they still got a strategy test (`research/h23_ml_strategy.py`):
ridge (the best IC) and the GRU each ranked the defensive book's coins or the rotation book's,
using the walk-forward scores (made daily, where the bot's own ranking updates hourly):

| Design | Median composite | Folds better | Worst drawdown |
|---|---|---|---|
| **Current (kept)** | **2.87** | – | 42% |
| Book ranked by ridge | 2.73 | 0/6 | 44% |
| Book ranked by the GRU | 2.73 | 2/6 | 44% |
| Rotation ranked by ridge | 1.02 | 1/6 | 34% |
| Rotation ranked by the GRU | 2.03 | 2/6 | 34% |

None met the rule. As with the Kalman ranking, models that predict the average coin steer the
rotation book away from the few explosive moves it lives on. (The models exist in research
only; trading one live would mean putting its weights in the bot.)

### Round 15: order flow in the strategy

On complete data the week's taker flow passed the screen, so it got the same treatment
(`research/h24_order_flow.py`, with the scores computed once a day from Binance's taker
volumes, which the bot's candles do not carry):

| Design | Median composite | Folds better | Worst drawdown |
|---|---|---|---|
| **Current (kept)** | **2.87** | – | 42% |
| Book ranked by low volatility and taker flow | 2.69 | 1/6 | 44% |
| Book ranked by low volatility, spread, Kalman trend and taker flow | 2.71 | 2/6 | 44% |

Neither met the rule. Across rounds 12 to 15, eight signals, two regime filters and four
models were screened, and nine strategy designs built from the ones that passed were run
through the six folds. None beat the current bot, so it is unchanged: signals that predict the
average coin a little were not what this strategy needed.

### Round 16: tokenized stocks, tested on their shares' history

Roostoo's 21 tokenized stocks have only been listed since June–July 2026, too little history to
test anything on. Their shares have years of it, and the tokens track them closely: at the US
close, a daily return correlation of 0.990–1.000, an average gap of about 0.1% and a daily
tracking error of 0.07–0.35% (SPCX 0.79%) (`research/h25_stock_proxy.py`). So strategies were
tested on the shares, fixed before running: **S1** holds each stock while its 50-day average is
above its 200-day; **S2** rotates weekly into the 2 stocks with the best 3-month return while
the basket is above its 200-day average (the crypto rotation's logic).

On the shares, over 15 October-to-October years from 2011, the median composite was 2.53 for
holding all 21 equally, 2.25 for S1, 2.17 for S2 and 1.90 for QQQ. As a 20% sleeve beside the
crypto bot, S1 was better in 3 of the 6 crypto folds and S2 in 5 of 6 (median 4.04 against
2.87, worst drawdown 38% against 42%), which meets the rule.

But the 21 shares are the ones chosen for tokenization in 2026, after their big runs (NVDA,
PLTR, MSTR, MU, SNDK), so a strategy that buys the hottest of them is bound to look good in
hindsight, and S2's gain came mostly from the last two years. Run instead on the 40 largest US
companies at the start of 2011, a list made without knowing what came next, S2 had a median
composite of 0.50 against 2.14 for holding them all, and as a 20% sleeve it was better in only
1 of 6 crypto folds (median 2.77). Its edge on the tokenized list is the list itself, so no
stock sleeve was added.

### Round 17: a separate strategy for the shares

Since the crypto rotation fails on shares (a faithful copy lost 84% over 15 years on the 2011
top 40: over two weeks, large shares reverse where coins keep going), the shares got their own
search, from hypotheses to a final strategy. To keep hindsight out, every design was tested on
the S&P 500 as it was at each date (membership from
[fja05680/sp500](https://github.com/fja05680/sp500); each month the 50 members with the most
dollar volume; Yahoo prices for 76% of the 848 tickers that were members since 2010, the
missing ones mostly long-delisted). Gates fixed before running: beat simply holding those 50
(median composite, 9 of 15 years, drawdown at most 5 points worse), then improve the crypto bot
as a 20% sleeve on the usual rule.

| Round | Designs | Result |
|---|---|---|
| 1 (`research/h26_stock_strategies.py`) | 12-1 momentum, momentum with a market filter, low volatility, trend per stock, short-term reversal, 52-week-high momentum, volatility management, momentum with low volatility | None beat holding the 50 (median 2.18; the best, volatility management, 2.04). Holding them as a sleeve helped the bot in 2 of 6 folds. |
| 2 (`research/h27_stock_round2.py`) | The four lowest-drawdown designs straight to the sleeve test; turn of the month, index time-series momentum, volatility-scaled trend | Only momentum with low volatility passed (4 of 6, median 2.91 against 2.87), with one of the four a tie (-1.9145 against -1.9099). Its neighbours (10% or 30% sleeve, 6-month momentum, 3-month volatility) passed 1 of 4, so it was noise. |
| 3 (`research/h28_cash_overlay.py`) | Only the bot's idle cash (53% on average) in volatility-managed or trend-following shares | Worse: the worst drawdown rose from 41% to 47–51%, as the shares fell with crypto in 2021–22. |

Fourteen designs and a robustness check found no share strategy that improves the bot. Large
shares were hard to beat by holding them over 2011–2026, and they fall with crypto in a crash,
so they add little to an account whose returns come from crypto rallies. Searching on until
something passed would only find a lucky fit, so the search stopped here and the bot trades no
tokenized shares.

### Round 18: funding rates

Binance's perpetual futures charge a funding rate every 8 hours; when longs pay a lot, the
long side is crowded. The history is public, so it was screened with the H20 rule
(`research/h29_funding.py`) and passed: coins with high 7-day funding did worse over the next
week (IC -0.038, every fold, though near zero in 2024–26), and high BTC funding preceded weaker
weeks for BTC (correlation -0.071, every fold). Three designs were then fixed and run through
the folds (`research/h30_funding_strategy.py`):

| Design | Median composite | Folds better | Worst drawdown |
|---|---|---|---|
| **Current (kept)** | **2.87** | – | 42% |
| F1 book ranked by low volatility and low funding | 3.03 | 3/6 | 44% |
| F2 rotation skips the most crowded fifth of coins | 2.12 | 4/6 | 37% |
| F3 rotation out while BTC's funding is in its top fifth | 1.43 | 0/6 | 42% |

None met the rule. F2 shows why: the coins the rotation makes its money on are the crowded
ones, and skipping them cut 2023–24 from +229% to +98%. The hooks stay in the code, off.

### Round 19: shorts in bear markets, and volatility forecasts

When BTC's trend filter is off, the rotation's 40% sits in PAXG or cash (PAXG 25% of the time,
cash 22%). Two ways to use it, and two ways to size the rotation by forecast volatility, fixed
before running (`research/folds.py --round19`):

| Design | 20–21 | 21–22 | 22–23 | 23–24 | 24–25 | 25–26 | Median | Folds better | Worst drawdown |
|---|---|---|---|---|---|---|---|---|---|
| **Current (kept)** | 11.32 | -1.91 | 1.33 | 5.65 | 4.30 | 1.44 | **2.87** | – | 42% |
| A1 short the 2 weakest coins while the filter is off | 5.29 | -1.29 | 0.49 | 2.29 | 2.45 | 1.22 | 1.76 | 1/6 | 50% |
| A2 short the 2 most volatile coins while the filter is off | 8.09 | -0.41 | -0.18 | 3.10 | 2.52 | 1.08 | 1.80 | 1/6 | 46% |
| B1 rotation scaled by a HAR volatility forecast of BTC | 9.07 | -1.88 | 0.84 | 5.41 | 4.02 | 1.09 | 2.55 | 1/6 | 41% |
| B2 rotation scaled by an EWMA volatility forecast of BTC | 10.85 | -1.89 | 1.06 | 5.28 | 4.34 | 1.39 | 2.86 | 2/6 | 42% |

None met the rule. The shorts open after the trend filter has turned, which is after much of
the fall, and bear-market rallies then squeeze them: A2 halved the 2021–22 loss (-18% against
-39%) but turned 2022–23 from +39% into -16%. Scaling the rotation down when BTC is forecast to
be volatile (HAR: the mean of the last day's, week's and month's realised variance; EWMA:
RiskMetrics) mostly cut it out of the sharp rallies it earns from. All four stay as options,
off.

### Rounds 20 to 30: hypotheses aimed at the bot's weak spots, under a stricter rule

By round 19 about 50 designs had been tried, and a design with no real edge passes the old rule
(higher median, better in 4 of 6 folds) about one time in five. So from round 20
(`research/rounds.py`) a design is adopted only if it has a higher median composite, is better
in at least 5 of 6 folds, has a worst drawdown at most 2 points worse, and its neighbours (its
setting nudged both ways) pass the old rule. Each round was fixed before it ran, in the light
of the one before. Composite median (incumbent 2.87), folds better, worst drawdown (incumbent
42%):

| Round | Aimed at | Design | Median | Better | Worst DD |
|---|---|---|---|---|---|
| 20 | the rotation has no stop | trailing stop on each pick: 8 ATR / 5 ATR / 15% | 2.79 / 2.51 / 2.46 | 2 / 1 / 0 | 46 / 46 / 44% |
| 21 | what gets picked | must beat BTC, else BTC / else PAXG or cash | 2.87 / 2.87 | 2 / 1 | 42 / 43% |
| | | **ranked by 1-, 2- and 3-week momentum together** | 2.62 | **5** | **37%** |
| 22 | daily swaps | keep picks in the top 4 / keep picks 3 days | 3.27 / 2.86 | 4 / 4 | 37 / 38% |
| 23 | the account's drawdown | halve the rotation 15% below the peak / only above the 30-day average | 2.63 / 3.54 | 2 / 2 | 33 / 28% |
| 24 | a regime from all coins | breadth and BTC / breadth alone / breadth for the book | 2.32 / 1.97 / 2.52 | 2 / 1 / 3 | 39 / 55 / 42% |
| 25 | the filter's lag | also out below BTC's 200-hour average / re-enter after 24h | 2.54 / 2.86 | 1 / 3 | 37 / 42% |
| 26 | the near misses together | multi-horizon + top-4 buffer / + 3-day hold | 2.61 / 2.51 | 4 / 4 | 37 / 39% |
| 27 | conviction | one pick when it doubles the second | 3.22 | 3 | 48% |
| 28 | euphoria | halve the rotation when BTC is up 25% in 2 weeks | 2.78 | 1 | 42% |
| 29 | bounces in downtrends | picks must be in their own uptrend | 2.00 | 0 | 44% |
| 30 | idle capital in good times | the book 100% invested in up to 10 coins | 2.83 | 2 | 44% |

Nothing met the rule, so the bot is unchanged. One pattern is worth recording. The designs
that make the rotation less dependent on its single 2-week lookback (multi-horizon ranking,
letting winners stay) cut drawdowns, raised the 14-day composite and won most years, but all
lost 2024–25, one of the two years the rotation's settings were developed on (rounds 3 and 4).
Ranking by 1-, 2- and 3-week momentum together won all four folds that played no part in
designing the rotation, as well as 2025–26: over six years +12,518% against +8,228%, worst
drawdown 37% against 42%, median 14-day composite 3.38 against 2.53. It lost 2024–25 (3.25
against 4.30), which lowered its median (2.62 against 2.87), and of its neighbours one also won
5 of 6 folds while the other won 3. Under the rule it is not adopted; it is the one alternative
the evidence supports, one setting away:

```
{"strategy": {"rotation_ranking": "multi", "rotation_horizons": [168, 336, 504]}}
```

### Rounds 31 to 40: an untouched holdout, and the competition's own yardstick

To keep searching without simply finding lucky fits, rounds 31+ add a confirmation step on data
no research had touched: October 2018 to October 2020 (`research/holdout2018.py`; the coins
Roostoo lists today that traded then, 20 of the 54). A design counts as better only if it passes
the strict rule on the six folds (from round 31 compared fold by fold, paired: at least 5 of 6
folds better, which implies a positive median gain; the unpaired medians had let the one year
the rotation was tuned on veto designs better in the other five), its neighbours each win at
least 4 of 6 folds, and it beats the incumbent in both holdout years. No design's holdout result
is looked at before it passes on the six folds.

| Round | Design | Folds better | Result |
|---|---|---|---|
| 31 | sub-sleeves on 1-, 2- and 3-week momentum / half 2-week, half multi-horizon | 4 / **5** | the second's neighbour with 3- and 4-week horizons broke (3/6) |
| 32 | half 2-week, half 1+2-week ranking / sub-sleeves on 1- and 2-week momentum | **5** / 3 | the first's 240-hour neighbour broke (3/6) |
| 33 | hysteresis on the book's regime / on its trend exit / 6% rebalance threshold / on the rotation filter | 1 / 1 / 4 / 3 | cutting churn mostly cut useful trades |
| 34 | book risk-off PAXG only / 10% exposure / brake at 6% | 1 / 0 / 3 | the book's calm coins cushioned the 2021–22 crash |
| 35 | enter as soon as the filter turns on | 4 | more whipsaw, worst drawdown 48% |
| 36 | book skips rotation coins / correlation-aware second pick / picks need a 1-sd rise | 2 / 0 / 3 | the correlation term never outweighed momentum |
| 37 | rotation winners run to twice their weight | 3 | worst drawdown 46% |
| 38 | rotation filter on ETH / both regimes on ETH | 1 / 0 | BTC leads |
| 39 | picks need 2,000 hours of history | 3 | hardly binds |
| 40 | picks among the 20 most traded | 3 | |

`research/competition_rule.py` then judged all 37 designs of rounds 20–37 on the competition's
own yardstick, the median 14-day composite per fold (fixed after round 37): none was higher than
the incumbent's in 5 of 6 folds. The higher 14-day figures some designs show on average come
from one fold, 2020–21. No design reached the holdout. Forty rounds and about 85 designs later
the incumbent stands: designs that win most years break when their setting is nudged, which is
what noise around a good design looks like.

### Rounds 41 to 44: sentiment, Donchian channels, partial rebalancing, adaptive lookback

Same rule as rounds 31–40 (composite per fold, from October 2020; incumbent median 2.87, worst
drawdown 42%):

| Round | Design | Folds better | Median | Worst DD |
|---|---|---|---|---|
| 41 | halve the rotation in extreme greed (Fear & Greed ≥ 80) | 0/6 | – | 43% |
| | halve the rotation when stablecoin supply growth is in its yearly top fifth | 1/6 | – | 42% |
| 42 | Donchian rotation: 20-day breakouts, held until the 10-day low | 1/6 | 2.02 | **29%** |
| | rotation picks must be at a 20-day high | 1/6 | 0.41 | **22%** |
| | rotation picks exit below their 10-day low | 0/6 | 2.83 | 42% |
| | the book exits below its 10-day low instead of the ATR stop | 2/6 | 2.87 | 42% |
| 43 | plain rebalances move halfway to their target | 3/6 | 2.74 | 42% |
| 44 | 1-week lookback while BTC's volatility is above its 60-day median | 2/6 | 2.20 | 44% |

Both sentiment series passed a screen against BTC's next week (`research/h31_sentiment.py`:
Fear & Greed correlation -0.072, stablecoin growth -0.054, 5 of 6 folds), but the rotation earns
most in exactly those greedy weeks (`research/h32_sentiment_strategy.py`). The Donchian rules
nearly halved the worst drawdown by entering only after a 20-day high, and gave up most of the
return doing it.

### Rounds 45 and 46: TradingView's indicators

Round 45 (`research/h33_indicators.py`) screened about 65 of TradingView's built-in indicators
(most from the `ta` library, plus SuperTrend, Hull MA, Choppiness, Elder Ray, Chande Momentum,
Coppock, Balance of Power and linear-regression slope and R²), each on the 1-hour and the daily
chart, 124 series in all, with the H20 rules. 37 passed as 24-hour rankings, 23 as 168-hour
rankings and 3 as BTC timing signals. Most were the low-volatility effect again (ATR %, Keltner,
Bollinger and Donchian widths, Ulcer Index), or short-term reversal on the 1-hour chart; the one
genuinely different family was volume accumulation (Chaikin Money Flow and Accumulation/
Distribution change on the daily chart, IC +0.041 a week ahead).

Round 46 (`research/h34_indicator_strategies.py`) took, by a rule fixed before the screen, the
strongest of each family and used it four ways: in the book's ranking (U1), as the rotation's
ranking (U2), as a filter that keeps the rotation out of the worst fifth (U3), and, for the
timing signals, halving the rotation in their bearish fifth (U4). Fifteen designs; composite per
fold against the incumbent (11.32, -1.91, 1.33, 5.65, 4.30, 1.44):

| Design | Folds better | Worst DD |
|---|---|---|
| U1 book: low vol + Elder bull power / DI+ − DI− / Keltner width / Accumulation-Distribution | 3 / 2 / 3 / 0 | 43–45% |
| U2 rotation ranked by each | 1 / 1 / 1 / 2 | 31–40% |
| U3 rotation skips the worst fifth by Elder bull power / DI+ − DI− / Keltner width | 2 / 1 / 2 | 35–47% |
| **U3 rotation skips the worst fifth by Accumulation/Distribution change (daily)** | **5** | **38%** |
| U4 rotation halved when BTC is bearish on KAMA / SMA(200) / Choppiness | 1 / 0 / 2 | 38–42% |

The Accumulation/Distribution filter is the first design of the whole project to pass the
strict rule and its robustness check (neighbours 5 and 4 of 6). On the untouched holdout it won
2018–19 (2.12 against 1.98) but lost 2019–20 (1.50 against 2.87), so it is not adopted. With 124
series screened, about one design in fifteen would get this far by luck; the holdout is there
to tell, and it did. The option stays in the research code only (it needs Binance volume data
the bot's live loop already has, if a future, untouched test ever confirms it).

### Rounds 47 and 48: order flow and ICT concepts

Gamma exposure could not be tested: Deribit publishes options trades but not past open interest
by strike, so historical gamma levels cannot be rebuilt from free data. Order-book heatmaps
exist (Binance's futures book depth) only from 2023, which covers neither the early folds nor
the holdout. What could be tested, `research/h35_orderflow_ict.py` screened on the 1-hour and
daily charts with the H33 rules (24 series; the taker-buy history was extended back to 2018 for
the holdout):

- Order flow from taker-buy volume (delta = taker buys - taker sells): CVD slope, CVD-price
  divergence, absorption, delta z-score, buying climax.
- ICT concepts from OHLC: liquidity sweeps (a wick through the prior swing high or low that
  closes back inside), fair value gaps, market structure (the latest break of structure),
  change of character.

ICT's liquidity sweeps, fair value gaps and changes of character showed no predictive power at
all (ICs within ±0.02, inconsistent across folds); market structure on the 1-hour chart passed
only as a next-day reversal (IC -0.022). The slow order-flow series passed: three weeks of CVD
on the 1-hour chart (IC +0.038 a week ahead, every fold) and its daily-chart versions, the same
family as round 46's Accumulation/Distribution.

`research/h36_orderflow_strategies.py` used them as round 46 did (no timing signal passed).
Keeping the rotation out of the fifth of coins with the weakest three-week CVD won 5 of 6
folds (worst drawdown 41%), but its neighbour with 1.4x the window won 2, so it was not robust;
everything else won at most 3. "Don't buy what is being sold" has now come close twice (round
46 lost a holdout year, this one broke on a neighbour), which is suggestive and not enough.

### The rotation's share, optimised

Run alone, the rotation made +78,485% over the six years against +387% for the defensive book,
with drawdowns of 45–70% against 11–22%. So the user asked for its share to be optimised, in
steps of 5% (`research/rotation_weight.py`). Rules fixed before the new grid points ran: rank
each share in each fold by the median 14-day composite (the competition's yardstick), sum the
ranks, smooth over neighbouring shares, and confirm the winner on the 2018–2020 holdout.

| Rotation | 6-year return | Worst yearly drawdown | 2021–22 | Median yearly composite | 14-day rank sum |
|---|---|---|---|---|---|
| 30% | +4,268% | 35% | -32% | 2.84 | 45 |
| 35% | +5,687% | 39% | -35% | 2.75 | 34 |
| 40% | +8,228% | 42% | -39% | **2.87** | 25 |
| 45% | +9,875% | 46% | -43% | 2.75 | 30 |
| **50%** | **+12,696%** | **48%** | **-45%** | 2.76 | **47** |
| 55% | +15,807% | 52% | -49% | 2.69 | 35 |
| 60% | +18,756% | 54% | -51% | 2.65 | 41 |
| 70% | +31,945% | 58% | -56% | 2.67 | 54 |
| 80% | +45,032% | 63% | -61% | 2.62 | 57 |
| 95% | +69,960% | 68% | -66% | 2.56 | 59 |

The 14-day objective rises with the rotation's share, noisily, and picked 95%, which then failed
the holdout (2019–20: median 14-day composite -0.47 against -0.02 for 40%). The grid is a plain
risk-return line: each 5% more rotation adds return and 2–3 points of drawdown. So the share
was set by the usual form of that choice, the most return within a risk limit: the worst yearly
drawdown must stay under 50%. That is 50% (55% reaches 52%). It is also the share confirmed on
the holdout on both yardsticks: yearly composite 2.03 and 3.03 against 1.98 and 2.87, median
14-day composite -2.15 and -0.01 against -2.31 and -0.02. The price is a deeper worst drawdown
(48% against 42%) and a worse crash year (-45% against -39%).

The user then chose 60% and finally 70%: more return for more risk, as the competition ranks
on return before the composite. At 70%, over the six folds +31,945% (against +12,696% at 50%),
with a worst yearly drawdown of 58% and -56% in 2021–22; its median 14-day composite beat 40%'s
in 5 of the 6 folds. On the untouched holdout it beat 40% on both yardsticks in both years:
yearly composite 2.15 and 3.14 against 1.98 and 2.87, median 14-day composite -1.99 and +0.30
against -2.31 and -0.02 (returns +86% and +162% against +54% and +97%, drawdowns 32% and 40%
against 21% and 29%). 60% had not: it lost the 2019–20 14-day comparison.

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
order left open by a crashed run, and it rebuilds positions from the real wallet. It also
re-reads Roostoo's trading rules every cycle: a pair Roostoo has halted (`CanTrade` false) or
delisted gets no orders and is held as it is until it trades again. (It used to refuse to
start if any of its pairs was halted, which under `run_bot.sh` would have meant restarting
and failing indefinitely.) Trailing
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
| **Bot (45 coins, 70% rotation, 8-coin ERC book)** | **+1,248%** (36%) | **-56%** (58%) | **+56%** (46%) | **+580%** (38%) | **+265%** (43%) | **+38%** (37%) | **+31,945%** |
| Bot with 60% rotation | +973% (34%) | -51% (54%) | +52% (42%) | +431% (33%) | +228% (39%) | +36% (32%) | +18,756% |
| Bot with 50% rotation (the optimised share under a 50% drawdown limit) | +795% (30%) | -45% (48%) | +44% (38%) | +323% (29%) | +209% (35%) | +38% (28%) | +12,696% |
| Bot with 40% rotation (until 4 October 2026) | +679% (27%) | -39% (42%) | +39% (33%) | +229% (24%) | +181% (31%) | +36% (25%) | +8,228% |
| Bot with a 4-coin inverse-ATR book (before round 6) | +537% (25%) | -40% (42%) | +42% (30%) | +236% (23%) | +150% (31%) | +30% (22%) | +5,871% |
| Bot with 20 coins (the earlier list) | +786% (21%) | -28% (35%) | -1% (30%) | +39% (41%) | +89% (33%) | +43% (24%) | +2,258% |
| Defensive book alone | +117% (16%) | -10% (16%) | -11% (14%) | +2% (13%) | +30% (14%) | +16% (10%) | +167% |
| Hold BTC | +306% (55%) | -56% (74%) | +39% (27%) | +135% (32%) | +80% (31%) | -27% (54%) | +674% |

Maximum drawdown in brackets. The bot beat holding BTC in all six years (in 2022–23 only just:
+39.1% against +39.0%) with a smaller
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
| `journal/decisions.jsonl` | One line per hour: the signals for every pair, ranking scores, regime, brake state, pairs Roostoo is not trading, target weights and planned trades. |
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
left open, a Binance outage, and a pair halted at start-up and resumed later.
