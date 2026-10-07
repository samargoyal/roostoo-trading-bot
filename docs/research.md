# Strategy research

The research behind the bot, in the order it was done: every hypothesis, the rule fixed
before it was tested, and the result. The scripts are in [`research/`](../research/), each
named in its section; the summary is in the [README](../README.md#strategy-research).

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

## Time-series models and advanced portfolio construction

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

## Beating buy-and-hold

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

## Out-of-sample validation across six years

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

## Convex optimisation of the rotation book (round 5)

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

## Convex optimisation of the defensive book (round 6)

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

## Ideas from VECM-ARB

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

## Rounds 7 and 8: VECM-ARB ideas and tail-risk control

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

## Round 9: how robust are the rotation book's settings?

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

## Round 10: VECM-ARB's three engines, tested on crypto

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

## Round 11: halted coins (VECM-ARB's dynamic risk engine, adapted)

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

## Round 12: microstructure, volume surfaces, Bayesian methods and Kalman filters

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

## Round 13: the signals that passed, in the strategy

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

## Round 14: neural networks and deep learning

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

## Round 15: order flow in the strategy

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

## Round 16: tokenized stocks, tested on their shares' history

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

## Round 17: a separate strategy for the shares

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

Two follow-ups on trading hours (`research/h37_overnight.py`, daily open and close 2016–2026).
Most of a share's return comes while New York is closed: for the tokenized shares about +26% a
year from close to next open against +5% from open to close (2011's 40 largest companies: +9.5%
against +2.3%). Trading them only during US hours keeps the smaller part and pays two trades a
day, which loses 18–38% a year after fees. Holding only overnight with a predictor fares no
better: the best of three (overnight momentum, after Lou, Polk and Skouras) picked nights
earning 0.065% on average for 2011's 40 largest (0.18% for the tokenized shares), against
0.1–0.2% for the round trip, so it lost 9% a year after limit-order fees and beat simply holding
in 1 of 10 years.

Holding only the shares predicted to beat the cost (`research/h38_overnight_selective.py`: ridge
and gradient-boosted trees on 14 features, retrained yearly, 2018–2026) did no better. For
2011's 40 largest companies, the nights predicted above 0.1% then earned 0.028% (ridge) and
-0.005% (trees) on average: the predictions that cleared the cost were mostly noise, and every
version lost money after costs. On the tokenized shares the best made +6.7% a year against
+46.8% for holding them.

## Round 18: funding rates

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

## Round 19: shorts in bear markets, and volatility forecasts

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

## Rounds 20 to 30: hypotheses aimed at the bot's weak spots, under a stricter rule

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

## Rounds 31 to 40: an untouched holdout, and the competition's own yardstick

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

## Rounds 41 to 44: sentiment, Donchian channels, partial rebalancing, adaptive lookback

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

## Rounds 45 and 46: TradingView's indicators

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

## Rounds 47 and 48: order flow and ICT concepts

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

## The rotation's share, optimised

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

## Rounds 50 to 64: long-short

The user asked for a long-short strategy. Roostoo's shorts are 1x: a short locks USD collateral
equal to its size and pays 0.1% to open and to close, with no borrow fee, so long plus short
can never exceed the account, and a market-neutral book earns only half its long-short spread.
Every short tried before (H5, H13, round 19) failed: shorts opened after the trend had turned
and bear-market rallies squeezed them.

**A screen first** (`research/h50_long_short.py`): 24 variants, each a book of at most 100%
gross rebalanced daily, judged alone and blended with the bot.

| Idea | Result |
|---|---|
| Long the 3 strongest coins, short the 3 weakest (cross-sectional momentum) | The weakest bounce hardest: drawdowns up to 86% |
| The same on returns net of BTC's beta | The same problem |
| Short coins listed less than a year ago, in a downtrend (token unlocks) | -91% in 2020–21 |
| Short the 2 weakest coins while BTC's filter is off (round 19) | Lost money in all six years |
| Long the calmest coins, short the wildest | Lost in 4 of 6 years |
| BTC alone, long/short on its trend | -45% in 2022–23 |
| Altcoins against BTC on the trend of their ratio | Better in 1–3 of 6 years |
| **Every coin long in its uptrend, short in its downtrend, inverse-volatility weights** | **Positive in 5 of 6 years on its own** |

**Two planner fixes.** A fully invested book could not pay for an entry: the planner kept
proposing buys with no cash, which never filled, and blocked the activity rule, leaving days
without a trade. The planner now trims the holdings furthest above their targets to pay for
entries, and the activity rule falls back to a short (or to a small sale when fully invested).
The bot as it is backtests identically; the long-short books' first results had been flattered
by the stuck entries (a crash year of -40% became -54% once fixed).

**In the bot's own backtester**, each design in place of the 30% defensive book unless noted,
under the rule of rounds 31 on (paired gains in at least 5 of 6 folds, worst drawdown at most 2
points deeper, neighbours holding in 4 of 6, then both holdout years):

| Round | Hypothesis | Best design | Folds better |
|---|---|---|---|
| 50 | The per-coin trend book; regime-aligned; BTC alone; the rotation shorting a basket in bear markets | Trend book (240h/960h EMAs) | 4/6 |
| 51 | A neutral zone; sizing by trend strength | Either | 3/6 |
| 52 | Shorts only in confirmed bear markets (BTC below its 200-day average) | Hybrid book | 3/6 |
| 53 | Squeeze defences: shorts shrunk after wild months; 10-ATR trailing stops; turtle entries and exits; majors only | Stops, or the volatility scaling | 4/6 |
| **54** | **No short where perpetual funding is negative (crowded shorts)** | **Trend book with stops and the funding filter (R54b)** | **5/6, passed** |
| 55 | Overlay: the defensive book's idle cash shorts downtrends | With volatility scaling (R55c) | 5/6, passed |
| 56 | R54b plus volatility scaling; no crowded longs; R55c plus funding | R56a and R56c | 5/6, passed |
| 57 | Short only where funding is above the median; the rotation skipping crowded picks | — | at most 4/6 |
| 58 | A three-speed trend vote | The overlay with the vote | 5/6, but weaker than R56c |
| 59 | Half-size shorts; R54b regime-aligned | — | at most 4/6 |
| 60 | R54b's unused share in PAXG while gold rises | — | 4/6 |
| 61, 62 | The rotation's idle share running R54b's book while BTC's filter is off (crash year -36%), also only in confirmed bears | — | 3/6 |
| 63 | New shorts need funding of 0.005%; no new short below RSI 30 | Both | 5/6, passed, but not better than R54b |
| 64 | Trailing stops on R54b's longs; shorts on a faster trend than longs | Faster shorts | 5/6, passed, but not better than R54b; the long stops failed the holdout |

R54b, the best: every coin long while its 240-hour EMA is above its 960-hour EMA and short while
below, weighted by inverse volatility to 100% of the book; a short is covered once the price
rises 10 ATRs above its lowest close since entry (no new short in that coin for 24 hours); and
no coin is shorted while its perpetual funding rate has averaged below zero over the last 3
days, when shorts are crowded and a squeeze is likeliest.

| | 2020–21 | 2021–22 | 2022–23 | 2023–24 | 2024–25 | 2025–26 | 6 years | Worst drawdown |
|---|---|---|---|---|---|---|---|---|
| Bot as it is (defensive book) | +1,248% | -56% | +56% | +580% | +265% | +38% | +31,945% | 58% |
| R54b (long-short book) | +1,614% | -49% | +59% | +626% | +233% | +48% | +49,404% | 52% |

Its neighbours all held (funding over 1 or 7 days, stops at 7 or 14 ATRs, EMAs 168h/672h: 5/6;
EMAs 336h/1344h: 4/6), and it beat the bot in both holdout years (yearly composite 2.80 and 3.31
against 2.15 and 3.14; returns +124% and +187% against +86% and +162%), though funding history
starts in late 2019, so the 2018–19 year tests only the stops. It trades on 99–100% of days.
With fees doubled it still beat the bot in 5 of 6 years (+24,628% against +15,442%). The
rotation's share was checked for information: with R54b, 60/40 made +37,056% (worst drawdown
45%), 70/30 +49,404% (52%) and 80/20 +60,979% (59%); 70/30, the user's choice, was kept.
Its variants in rounds 55 to 64 landed within a few percent of it, a plateau rather than a
lucky setting, so the search stopped there and R54b went live on 5 October 2026. Live, the
funding rates come from Binance's USD-M futures API once a day (`LiveConfig.funding_url`);
a coin without them is not filtered, and if the API cannot be reached the filter is off for
the day.

## Swing and medium-frequency strategies

The user asked for swing and medium-frequency strategies, each with a real hypothesis. Every
one below was written down with its hypothesis before it ran, then judged in two stages. In
Stage A it trades alone, after the fee and half the Roostoo spread, and its net return and
yearly composite must be positive in at least 5 of the 6 folds. In Stage B it must improve the
bot, either in place of the long-short book (B1: 70% rotation, 30% this) or as a 20% slice
beside the bot (B2), in at least 5 of 6 folds and both holdout years.

- **`research/h60_swing_mft.py`: 23 swing strategies (days to weeks) and 28 medium-frequency
  ones (minutes to hours, on 5-minute candles from `research/klines5m.py`).**
  - Swing: one passed Stage A, buying dips in the market's leaders. One of its neighbours
    broke, and as 20% of the bot it improved only 2 of 6 years.
  - Medium-frequency: six had a gross edge in every fold at a maker's cost, and none survived
    the taker fee and spread. Hourly reversal in meme coins made +432% in 2020–21, then lost
    92–93% a year in 2024–26.
- **`research/h61_swing.py`: 30 more swing strategies.**
  - Nine passed Stage A: breakout and retest, volatility contraction near highs, relative
    strength on BTC's down days, Darvas boxes, all-time-high breakouts, negative funding in a
    bull market, the hottest meme coin, meme hype phases and the golden cross.
  - Their equal-weight ensemble rose in every year (+4% to +121%, worst drawdown 25%).
  - None improved the bot: B1 in at most 3 of 6 years, B2 in at most 4, and each in 0 or 1 of
    the 2 holdout years.
  - The rotation already holds the strongest trends, so a second trend book mostly buys the
    same coins later.

## Retail attention: Wikipedia page views (rounds 65 to 67)

Retail attention is the usual explanation for coins that move on social media. Tweets are
neither free nor available years back, but Wikipedia page views are (Kristoufek 2013).
`research/attention.py` collects them daily since July 2015 for 34 of the coins, with
"Cryptocurrency" standing for the whole market. Each day's scores use views up to the day
before.

**The study.** `research/h62_attention.py` tested 14 strategies:
- Two passed Stage A: attention flowing into rising coins, and broad participation.
- Neither improved the bot: B1 and B2 in 2 of 6 years, and 0 or 1 of the holdout years.

**Inside the live bot**, judged as rounds 50 to 64 were:

| Design | 6 years | Folds better | Result |
|---|---|---|---|
| Live bot (R54b) | +49,404% | | |
| R65a skip coins whose attention is fading | +54,574% | 3/6 | fail |
| R65b only coins with normal or rising attention | +64,190% | 4/6 | fail |
| R66a rank by return plus half the attention score | +35,421% | 1/6 | fail |
| R66b R65b on the coin's share of all crypto attention | +56,382% | 5/6 | a neighbour broke (1/6) |
| R66c leave a pick when its attention collapses | +50,418% | 1/6 | fail |
| R66d leave after a day of 5 times the usual views | +23,760% | 2/6 | fail |
| R66e invest only while crypto's attention is near normal | +24,161% | 3/6 | fail |
| R66f no shorts into rising attention | +49,715% | 4/6 | fail |

**Round 67** (`research/round67_attention.py`) ran R66b's other neighbour (a share floor of
-0.1) as its own design:
- It beat the bot in all 6 folds (+77,012% over 6 years, worst drawdown 50%), and its
  neighbours held (-0.2: 4/6; 0.0: 5/6).
- It lost the 2018–19 holdout year (+103% against +124%).
- It was not adopted. A design picked after seeing the folds has to win both holdout years.

## Securing profits (rounds 68 to 70)

The user asked for a way to secure profits that also raises the Sharpe ratio. The difficulty:
the rotation's returns come from a few very large winners.

**Fixed rules (round 68)**, each on top of R54b:

| Design | 6 years | Worst drawdown | Folds better |
|---|---|---|---|
| Live bot (R54b) | +49,404% | 52% | |
| R68b keep half of a pick once it is 40% above its entry | +17,503% | 52% | 0/6 |
| R68c a 15% trailing stop once a pick is up 25% | +11,557% | 52% | 0/6 |
| R68d the sleeve shrinks as the account falls from its 30-day high | +6,767% | 43% | 0/6 |

**A lock-in for the competition's own horizon** (`research/h68_lock_in.py`). Every 14-day
window was replayed with one rule: once the account is up 10% in the window, halve the
exposure.
- The median 14-day composite rose in all 6 folds (3.19 to 5.88) and in both holdout years.
- The mean window return fell from +5.7% to +3.7%.
- It trades expected return for a steadier score.
- Not adopted: the user did not want profits secured by a fixed target.

**Exits when a trend tires** (`research/h69_trend_exit.py`). The user's idea: leave a coin when
most indicators say its trend is over or turning choppy, rather than at a fixed target. First
came the evidence. Each day, twelve warnings were measured on the rotation's candidates (the
top 5 by 14-day return, 5,688 candidate-days in 2020–26) against their next 3 days:

| Warning | Next 72 hours when on | When off | Lower when on |
|---|---|---|---|
| Weak highs: within 3% of the 7-day high with RSI(14) below 60 | +0.16% | +2.63% | 6/6 |
| Volume divergence: up over 3 days on under 70% of the previous 3 days' volume | +0.44% | +2.61% | 6/6 |
| Momentum turned: 3-day return below zero | +1.69% | +2.76% | 6/6 |
| Sellers in charge: taker-buy share below half over 24 hours | +1.76% | +3.17% | 6/6 |
| Lagging BTC over 3 days | +1.78% | +2.76% | 6/6 |
| Below the 72-hour EMA; Kaufman efficiency under 0.25; ADX under 20 | +1.6% to +2.0% | +2.7% to +4.5% | 5/6 |
| Choppiness Index above 61.8 | +1.78% | +2.43% | 4/6 |
| Overextended, crowded longs, a volatility jump on a falling day | +4.0% to +7.0% | about +2.1% | 0–2/6 |

**What the evidence ruled out.**
- A vote did not predict: leaving at 6 or more warnings lowered the forward return in only 4
  of 6 folds.
- Neither did machine learning. Gradient-boosted trees on the continuous indicators, trained
  walk-forward, scored an AUC of about 0.53.
- Warned coins still rose on average, so selling them for cash would lose. A warning is worth
  acting on only if the next healthy candidate does better.
- Swapping warned picks for the next healthy candidate improved the 2 picks' next 24 hours in
  all 6 folds for the two strongest warnings. The votes and the trees managed at most 3.

**Round 69** put the swap into the bot. At the daily re-pick, a candidate showing either warning
is skipped and the next healthy one takes its slot:

| Design | 6 years | Worst drawdown | Folds better | 14-day windows better |
|---|---|---|---|---|
| R69a swap on either warning | +104,339% | 53% | 5/6 | 2/6 |
| R69b volume divergence only | +101,852% | 54% | 5/6, drawdown over the limit | 3/6 |
| R69c weak highs only | +59,650% | 50% | 4/6 | 2/6 |

R69a was not adopted:
- Its stricter neighbour held (5/6), but its looser one broke (3/6).
- Its median 14-day window, the competition's horizon, was worse in 4 of 6 years. The yearly
  gain came from a few large windows.

**Round 70** asked the same of the market, with nine warnings on BTC and on all coins:
- The more warnings were on, the worse the rotation's next day (0–1 warnings: +2.45%; 6 or
  more: +0.26%).
- Even the worst bucket was positive, so going to cash on the vote lost (2 or 3 of 6 folds
  better).
- One warning stood out: BTC's 3-day return below zero (the rotation's next day +0.33%
  against +1.52%, lower in all 6 folds). The next step tested it.

| Design | 6 years | Worst drawdown | Folds better | 14-day windows better |
|---|---|---|---|---|
| R70a the rotation at half size while BTC's 3-day return is negative | +32,686% | 44% | 4/6 | 1/6 |
| R70b out of the rotation then | +17,282% | 46% | 3/6 | 1/6 |

- Halving cut the worst drawdown by 8 points, but cost much of 2023–24's rally (+389% against
  +626%).
- Neither was adopted.

**What rounds 68 to 70 found.**
- The signs that a trend is tiring are real, but none pays inside this bot over a year. Its
  returns come from staying in the strongest trends. Every exit tried, fixed or
  indicator-driven, gave up more of the large winners than it saved.
- Every option of rounds 65 to 70 is off by default in `bot/research_rules.py`. With them off,
  the bot reproduces R54b's backtests exactly in all 8 years (folds and holdout).

**A risk-scaled lock-in for the competition window** (`research/h70_secure_profits.py`). The
competition scores one 14-day window, and its composite punishes a late drawdown more than it
rewards a late gain. So the last question was whether securing gains within the window pays,
without a fixed profit target. Each rule halves every position once, for the rest of the window.
Tested on every 14-day window of the live bot, 2020–26:

| Rule | Score better | Sharpe better | Mean 14-day return |
|---|---|---|---|
| None (the live bot) | | | +5.8% |
| h68: once up 10%, halve | 6/6 | 6/6 | +3.7% |
| **S1: once the gain exceeds one daily volatility times the square root of the days left** | **6/6** | **6/6** | **+4.2%** |
| S2: once up one volatility, halve on giving back half the peak gain | 5/6 | 5/6 | +4.0% |
| S3: while in profit, halve when BTC's 3-day return turns negative | 5/6 | 5/6 | +3.9% |

- S1's bar adapts to volatility and to the time left: early in the window it needs a large
  gain, near the end a small one, since a gain is then more than a normal loss over the
  remaining days could erase.
- Every rule cut the mean window return by more than the tenth allowed, so none passed
  outright. Securing gains always gives up part of the large runs.
- S1's benefit was robust. Settings from 0.75 to 3 volatilities raised the score in 5 or 6 of
  6 years and the Sharpe ratio in 5 or 6, and every setting raised both in the two holdout
  years.
- In the bot's own backtester (`--bot`: the rule as `StrategyConfig.secure_k` runs it, the
  window restarting every 14 days, 26 windows a year), k = 1 raised the median window score
  in 5 of 6 years and both holdout years, and the median window Sharpe ratio in all 8. The
  mean window return fell from +5.2% to +4.2%. Settings of 1.5 and 2 raised the score in only
  4 of 6 years.
- The user decided not to use it on the competition account. `secure_k` and `secure_mode`
  stay among the research options in `bot/research_rules.py`, off by default.

**Is there a right time to be short?** (`research/h71_timing.py`). The account's day-to-day
swings led the user to ask for longs and shorts timed faster. Six fast "market turning down"
signals were measured at each UTC day's close, 2020–26, against what followed:

| Signal | On | BTC's next 24 hours | BTC fell in | Market's next 24 hours |
|---|---|---|---|---|
| BTC's 3-day return below zero | 47% | +0.09% | 2/6 years | +0.06% |
| BTC down over 3% in a day | 10% | +0.47% | 1/6 | +1.12% |
| BTC's 24-hour EMA below its 72-hour EMA | 48% | +0.06% | 2/6 | +0.05% |
| BTC below its 72-hour EMA | 47% | +0.08% | 3/6 | +0.11% |
| BTC's 7-day EMA below its 28-day (the live filter off) | 47% | +0.03% | 2/6 | +0.01% |
| Under half the coins above their 240-hour EMA | 57% | +0.13% | 2/6 | +0.08% |

- None of them was followed by falling prices in most years. After sharp drops prices
  rebounded, so shorting on any of these signals would have lost money even before fees.
- Hedging the rotation's coins with a BTC short on these signals earned less than simply
  holding less of them, since BTC usually rose.
- Shorts pay only in lasting downtrends. The bot already acts on those at the slow speed: the
  rotation leaves when BTC's filter fails, and the long-short book shorts coins in downtrends.

**Trading the account's swings** (`research/h72_swings.py`). Two days in, the account had
swung between +1% and +3% three times, and the user asked what selling near each top and buying
back lower would have made. On the live bot's own hourly equity, 2018–26:
- What followed a move was unrelated to it: the correlation between the last and the next 6
  to 72 hours was between -0.13 and +0.14 in every year, and after a 2% rise in a day the next
  day still rose on average in most years.
- The rule (sell once up 2% from the 24-hour low, buy back after a 1.5% pullback, or after a
  further 3% rise so as not to miss a trend) made less than holding in 7 of 8 years: -36%
  against +234% in 2024–25, with about 700 trades a year. Selling half, or other thresholds,
  did the same.
- A separate swing sleeve did less harm: h61's nine swing strategies that passed Stage A,
  equally weighted, as 10% of the account cut the worst drawdown from 52% to 48% and the mean
  14-day return from +5.2% to +4.9%, and raised the median 14-day score in only 2 of 6 years.

**A swing sleeve, long and short** (`research/h73_swing_sleeve.py`, `research/h74_long_short_swings.py`).
The user asked for more experiments on that sleeve, and for short swings as well as long ones.
The bot was modelled as its sleeves rebalanced daily (70% the rotation alone, 30% the long-short
book alone), which tracks it closely (+1,638% against +1,615% in 2020–21, the same 52% worst
drawdown), and each variant was judged against that model:

| Swing sleeve (h61's nine long survivors) | Years better | Worst drawdown | 14-day score better | Mean 14-day return |
|---|---|---|---|---|
| None (the model of the live bot) | | 52% | | +5.2% |
| 10% from both books | 3/6 | 48% | 2/6 | +4.9% |
| 10% from the long-short book | 2/6 | 53% | 2/6 | +5.1% |
| 10% from the rotation | 3/6 | 46% | 1/6 | +4.7% |
| 5%, 20%, or inverse-volatility weights | 3/6 | 45–50% | 0–2/6 | +4.5–5.0% |
| The rotation's idle share while BTC's filter is off | 3/6 | 49% | 4/6 | +5.2% |

- None passed. Swings in the rotation's idle share came closest: the same mean 14-day return,
  a better 14-day score and Sharpe ratio in 4 of 6 years and both holdout years, but worse
  years in 2023–26 (+188% against +228% in 2024–25).
- Short swings: nine short setups, the mirrors of the long survivors (breakdowns and retests,
  contraction near lows, weakness on BTC's up days, Darvas breakdowns, new lows, crowded longs
  in a bear market, the death cross) and two of their own (failed breakouts, bear-market
  rallies). All nine failed Stage A, losing in most years even in 2021–22's crash: rallies
  inside downtrends squeezed them (failed breakouts lost 96% in 2020–21). So a long-short swing
  book is the long one, and only slow trend-following shorts (the live book's) have paid.
- Shorts only in downtrends (`research/h75_gated_short_swings.py`): the same nine, entering
  only while BTC's 7-day EMA was below its 28-day and covered when it turned back, cut their
  losses (failed breakouts went from 1/6 to 4/6 years), but only one passed Stage A
  (relative weakness on BTC's up days), and both its neighbours broke. Even then it lost 25% in
  2021–22's crash. Run with the rotation's idle 70%, the long and short swings together kept
  the mean 14-day return (+5.3% against +5.2%) and raised the 14-day Sharpe ratio in 5 of 6
  years, but were better in only 3 of 6 years and not on the 14-day score in the holdout.

**Keeping the capital invested** (`--modes`). The user asked not to cut the capital. So with the
same trigger (k = 1), the rotation's coins instead moved somewhere calmer, in the bot's backtester:

| Once secured | Score better | Sharpe better | Holdout: score, Sharpe | Mean 14-day return |
|---|---|---|---|---|
| Nothing (the live bot) | | | | +5.2% |
| Halve every position, the rest in cash | 5/6 | 6/6 | 2/2, 2/2 | +4.2% |
| The rotation into the long-short book | 3/6 | 4/6 | 1/2, 1/2 | +3.8% |
| The rotation into BTC | 2/6 | 3/6 | 1/2, 2/2 | +3.8% |
| The rotation spread over its top 5 coins | 4/6 | 4/6 | 1/2, 2/2 | +5.7% |
| The rotation into PAXG | 4/6 | 5/6 | 2/2, 2/2 | +3.1% |
| Halve every position, the rest in PAXG | 4/6 | 6/6 | 2/2, 2/2 | +3.9% |
| The rotation's coins sold into PAXG and barred, the next pick buying others | 3/6 | 2/6 | 2/2, 2/2 | +5.7% |
| The same with k = 2 | 2/6 | 3/6 | 1/2, 1/2 | +4.4% |

- Moving into other crypto did not protect the score: BTC, the long-short book and a wider
  rotation all fall with the market. Only holding less crypto did.
- Halving into PAXG keeps every dollar invested and still raised the Sharpe ratio in all 6
  years and both holdout years. Halving into cash did slightly better on every measure, since
  gold has its own swings and costs.
- The user's design, selling the coins that made the gain into PAXG and buying other coins at
  the next pick, kept the full risk and lowered the median Sharpe ratio in 4 of 6 years. Its
  mean window return rose in the folds (+5.7%) but fell in the holdout (+2.9% against +4.7%):
  the coins sold often kept rising, and the ones bought instead fell with the market.

## Entries and exits, re-tested on the live bot (rounds 71 to 73)

The user asked for better entry and exit points, for longs and shorts. Most such rules had been
tested in rounds 20–48 on the bot of that time; rounds 71 (`research/round71_entries_exits.py`,
the rotation) and 72 (`research/round72_book_entries_exits.py`, the long-short book's longs and
shorts) tested them again on R54b, under the rule of rounds 65–70.

| Rule | 6 years | Worst drawdown | Years better | Result |
|---|---|---|---|---|
| Live bot (R54b) | +49,404% | 52% | | |
| Rotation: trailing stop 8 ATRs / 15% | +26,744% / +11,557% | 55% / 53% | 2/6, 0/6 | fail |
| Rotation: exit below the 72-hour low | +9,780% | 55% | 1/6 | fail |
| **Rotation: keep a held pick while it ranks in the top 3** | **+99,308%** | **51%** | **5/6** | **top 4 held (4/6), top 5 broke (3/6)** |
| **Rotation: rank on 7-, 14- and 21-day returns together** | **+84,839%** | **50%** | **5/6** | **7+14 held (5/6), 14+21+30 broke (3/6)** |
| Rotation: buy only at a 72-hour high | +473% | 29% | 1/6 | fail |
| Rotation: own uptrend, not overextended, steady climb | +15,106% to +36,253% | 51–57% | 0–2/6 | fail |
| Rotation: re-entry delay, filter hysteresis | +37,618%, +54,836% | 52%, 50% | 2/6, 3/6 | fail |
| Book: neutral band, size by trend strength | +46,207%, +39,115% | 52%, 53% | 0/6 | fail |
| Book longs: trailing stops (round 64) | +44,878% | 52% | 2/6 | fail |
| Book shorts: turtle entries and exits, 6-ATR stop, bear markets only | +42,123% to +47,341% | 54–56% | 1–3/6 | fail |
| Book shorts: funding margin, not when oversold, faster trend (rounds 63–64) | +49,224% to +50,795% | 51–52% | 2–4/6 | fail |

- Stops and channel exits cut the rotation's winners, as in round 20; breakout entries kept
  it out of most of the trend.
- Two rules passed the folds and broke on one neighbour: holding a pick until it leaves the
  top 3 (fewer swaps), and ranking on several horizons (round 21's near miss again). Both
  make the rotation trade less, not more. Not adopted by the rule.
- The user judged the far neighbours' misses unimportant, so the holdout was then run for both:
  the rank buffer beat the bot in both years (yearly score 3.28 and 3.82 against 2.80 and 3.31,
  worst drawdown 21% and 38% against 31% and 41%, and the median 14-day score too), as did the
  multi-horizon ranking (3.23 and 3.46). Both together did no better than the buffer alone in
  2020–26 (+77,163%, 14-day score better in 3 of 6 years).

**Round 73: the multi-horizon ranking, adopted** (`research/round73_ranking.py`). The user
chose the multi-horizon ranking, as the more responsive of the two, and asked whether to
re-pick more often than daily and whether to weight the horizons:

| Multi-horizon ranking | 2020–26 | Worst drawdown | Years better | 14-day score better | Holdout |
|---|---|---|---|---|---|
| Re-picked daily, horizons equal | +84,839% | 50% | 5/6 | 3/6 | both years |
| Re-picked every 12, 8, 6 or 4 hours | +34,482% to +82,895% | 53–61% | 1–4/6 | 1–3/6 | mixed |
| **Daily, the 1-week rank counted twice (2/1/1)** | **+123,592%** | **51%** | **5/6** | **3/6** | **both years** |
| Daily, weighted 3/2/1 | +139,284% | 51% | 5/6 | 3/6 | both years |
| Daily, weighted 1/1/2 | +89,032% | 55% | 4/6 | 3/6 | both years |
| Daily, the two picks by inverse volatility or equal risk | +66,408% to +66,732% | 49–50% | 3–4/6 | 3/6 | both years |

- Re-picking within the day swapped coins on noise: the horizons barely move in hours.
- Counting the last week more helped, at 2/1/1 and 3/2/1 alike.
- Chosen by the user on 6 October 2026: the 2/1/1 ranking, re-picked daily. The same day the
  user decided to stay on R54b, so it is kept ready in `config/comp_multi.json`. It failed
  round 71's neighbour test and was chosen after the holdout had been looked at, so expect
  less than the backtest: it beat the old ranking on the median 14-day window in only 3 of 6
  years, and lost in 2024–25 (+82% against +233%), a year that rewarded the 2-week leaders.

## Convex optimisation on the live bot (round 74)

The user asked for convex optimisation on the bot as live after round 73
(`research/round74_convex.py`). The long-short book weights its positions by inverse
volatility, ignoring how they move together; the new `ls_weighting` option instead solves for
the weights over the positions' side-adjusted hourly returns (a short counts as minus the
coin), re-solved daily, with the same gross.

| Design | 6 years | Worst drawdown | Years better | 14-day score better |
|---|---|---|---|---|
| Live bot (multi-horizon ranking, inverse-volatility book) | +123,592% | 51% | | |
| R74a the book by equal risk contribution (30 days of returns) | +89,005% | 57% | 2/6 | 2/6 |
| R74b the book at minimum variance, 10% cap | +93,420% | 56% | 2/6 | 1/6 |
| R74c the rotation's two picks by equal risk contribution | +84,638% | 51% | 1/6 | 3/6 |
| R74d the rotation at 60% / 65% / 75% / 80% (the split) | +80,947% to +178,921% | 45–58% | 3/6 | 2–4/6 |

- Both optimised books did worse and drew down deeper. The covariance of the last month did
  not hold when it mattered: in sell-offs crypto's correlations jump towards one, and the
  optimiser had leaned on the "diversifying" positions that then fell together.
- The two picks again did better equally weighted: an optimiser leans on the calmer leader.
- The split is a trade of return for drawdown with no free lunch: more rotation, more of both.
- None adopted; `ls_weighting` stays a research option, off by default.

## The research queue (rounds 75 on)

`RESEARCH_QUEUE.md` (from the user) lists circuit breakers, session filters, a validation audit
and new signals, each to be tested on both the live bot (`config/comp.json`) and the previous one
(`config/comp_r54b.json`) and judged on the competition rule C1–C5 (`research/queue.py`).

**Part A, operational circuit breakers** (`bot/breakers.py`, `tests/test_breakers.py`). Ten
checks against bad data, a broken connection or runaway orders: price divergence from Binance,
stale quotes, missing candles, wide spreads, rejected orders, slippage, order sanity, a fee
budget, holdings that changed between cycles, and commit-controlled `trading_halt` and
`reduce_only` flags. Exits and the activity trade always stay allowed (except under
`trading_halt`), and each trip is written to `breakers.csv`. Tested by fault injection against
the simulated exchange; off until `breakers.enabled` is set for an account, after a live cycle on
the testing account.

**Measurements first** (`research/h76_measurements.py`):
- Weekends: the weekend's return against the following Monday and Tuesday correlated from
  -0.45 to +0.21 by fold, no consistent reversal.
- Shock bars (a -3 standard deviation hour on 3 times the usual volume for that hour of the
  week) were followed by gains, not further falls: next 24 hours +0.3% to +3.1% in 5 of 6 folds
  with BTC's filter on. Selling them (B4) is the wrong way round and was dropped; buying the
  ones that close off their low (E5) goes to part E.
- The 10 most traded coins' mean 48-hour correlation reached 0.85 on only 2–7% of days, so a
  correlation cap (B8) would almost never bind; not built.

**Round 75, risk circuit breakers** (`research/round75_risk_breakers.py`). Median 14-day
composite better than each bot's (folds of 6), and the yearly return given up:

| Breaker | Live bot | Given up | R54b | Given up |
|---|---|---|---|---|
| B1 daily loss 3%, everything at half at 5% | 0/6 | -340 pts | 2/6 | -158 pts |
| B1b daily loss 3% only | 2/6 | -33 pts | 3/6 | -12 pts |
| B2 drawdown ladder 4/7/10% | 1/6 | -654 pts | 0/6 | -366 pts |
| B3 a position's 24-hour loss capped at 2.5% of equity | 1/6 | -418 pts | 0/6 | -264 pts |
| B5 short squeeze guard | 1/6 | +26 pts | 1/6 | +2 pts |
| B6 BTC shock | 3/6 | -339 pts | 3/6 | -154 pts |
| B7 volatility regime | 4/6 | -242 pts | 3/6 | -133 pts |

- None passed C1 on either bot. The drawdown ladder cut the worst yearly drawdown from 51% to
  20%, but by keeping the bot small through the rebounds that make its year.
- Losses on this book cluster and then reverse: every breaker that sells into a fall sells near
  the low, as the shock-bar measurement showed hour by hour.

**Round 76, session filters** (`research/round76_sessions.py`).
- C1, the hour of the daily re-pick: no hour beat 00:00 in more than 4 of 6 folds on either
  bot, and no block of 4 hours in a row reached 5 of 6, so the hour stays. The live bot's mean
  14-day composite peaked at 07:00 (22.4 against 8.8), but 07:00 beat 00:00 in only 2 of 6
  folds: one outlier year, the noise the block guard exists to ignore.
- C6, the funding hours: 08:00 and 16:00 beat 00:00 in 2–3 of 6 folds, so C6 was not run.

| Filter | Live bot: 14-day better | Given up | R54b: 14-day better | Given up |
|---|---|---|---|---|
| C2 entries only in the US session | 2/6 | -12 pts | 0/6 | -128 pts |
| C2 entries only in Europe's and the US's sessions | 2/6 | -1 pt | 0/6 | -153 pts |
| C3a no weekend entries | 2/6 | -237 pts | 1/6 | -145 pts |
| C3b everything at 0.7 at weekends | 1/6 | -185 pts | 0/6 | -116 pts |

None passed: delaying entries to a session costs the trend its first hours, and the weekends
were no worse than weekdays.

**Round 77, new signals** (`research/round77_signals.py`; the meta-labels from
`research/h77_meta_labels.py`). Median 14-day composite better than each bot's (folds of 6):

| Signal | Multi-horizon bot | Given up | R54b | Given up |
|---|---|---|---|---|
| E1 BTC short against half the rotation's beta | 4/6 | -570 pts | 2/6 | -303 pts |
| E2 half size while BTC tracks QQQ and QQQ is below its 50-day average | 3/6 | +6 pts | 3/6 | +3 pts |
| E3 entries only after a losing paper trade | 1/6 | -376 pts | 2/6 | -61 pts |
| E3 falsification: only after a winning one | 1/6 | | 1/6 | -226 pts |
| E4 rotation only in low-entropy trends | 1/6 | -352 pts | 2/6 | -140 pts |
| E4 falsification: high entropy | 2/6 | -253 pts | 2/6 | -109 pts |
| E5 capitulation buys, BTC's filter on | 3/6 | +14 pts | **5/6** | -5 pts |
| E5 capitulation buys, filter off | 3/6 | +4 pts | 4/6 | +1 pt |
| E6 meta-labelled sizing | 3/6 | | 3/6 | -73 pts |
| E7 no longs while funding is 1.5 sd above the coin's norm | 2/6 | | 3/6 | -165 pts |
| E8 longs grow only while volume accelerates | 1/6 | -10 pts | 2/6 | -29 pts |

- None was confirmed. The closest of the whole queue was E5 on R54b: 2% buys after high-volume
  shock hours that closed off their low, while BTC's filter was on, passed C1 to C4 (its
  neighbours, 1% and 3%, held) and failed C5, better in one holdout year of two. It changed
  the yearly return by a few points either way: too small to matter.
- E1 cut the worst drawdown to 25–28% at the cost of most of the return, like round 75's ladder.
- E6's model had no skill to size with: walk-forward AUC 0.48–0.55.

**D2, the deflated Sharpe ratio** (`research/d2_deflated_sharpe.py`). N = 437 designs and
neighbours tried in rounds 20–77; the spread of their Sharpe ratios from 40 re-run at random.

| Fold | Multi-horizon: Sharpe | DSR | R54b: Sharpe | DSR | Luck's best of 437 |
|---|---|---|---|---|---|
| 2020–21 | 3.56 | 1.00 | 3.09 | 0.99 | 0.94 |
| 2021–22 | -1.20 | 0.01 | -1.54 | 0.00 | 1.35 |
| 2022–23 | 1.27 | 0.62 | 0.98 | 0.50 | 0.97 |
| 2023–24 | 3.18 | 0.98 | 2.88 | 0.94 | 1.40 |
| 2024–25 | 1.11 | 0.68 | 1.88 | 0.90 | 0.67 |
| 2025–26 | 1.20 | 0.54 | 0.89 | 0.42 | 1.09 |

In the bull years both bots' Sharpe ratios clear what the best of 437 unskilled tries would
reach by luck (deflated Sharpe 0.9 or more); in the flat and falling years they do not
(0.0–0.6). The edge is a trend-following one that pays in trending markets, not an
all-weather one, which the six folds' returns already suggested.

**D1, the permutation test** (`research/d1_permutation.py`). Each fold's prices, warm-up
included, shuffled in 24-hour blocks in the same order for every coin (drift, volatility, fat
tails and cross-coin correlation kept, trends destroyed), 100 times, against the real history;
coins without a near-complete history were left out of both. Median 14-day composite:

| Fold (coins) | Multi-horizon: real / shuffled median / p | R54b: real / shuffled median / p |
|---|---|---|
| 2018–19 (8) | 3.94 / -2.30 / 0.04 | 3.83 / -2.22 / 0.04 |
| 2019–20 (11) | 5.01 / -1.57 / 0.03 | 3.36 / -1.69 / 0.04 |
| 2020–21 (13) | 23.29 / 1.59 / 0.02 | 31.24 / 1.49 / 0.01 |
| 2021–22 (26) | -1.00 / -2.79 / 0.23 | -0.90 / -2.56 / 0.24 |
| 2022–23 (26) | -3.41 / -2.11 / 0.76 | -2.62 / -2.06 / 0.59 |
| 2023–24 (30) | 5.09 / -0.81 / 0.05 | 5.79 / -0.37 / 0.04 |
| 2024–25 (36) | 1.56 / -0.99 / 0.16 | 2.40 / -0.79 / 0.12 |
| 2025–26 (46) | 2.11 / -1.59 / 0.13 | 1.14 / -1.38 / 0.20 |

- Both bots beat the shuffled histories in 7 of 8 folds; combined over the folds (Fisher's
  method) p is about 0.001 for each. The rules earn from the order of prices, the trends, not
  only from the drift and volatility a shuffled market keeps.
- Fold by fold the edge is clear only in trending years (2018–21, 2023–24). In 2024–26 it beat
  the shuffles but not significantly, so the recent edge is weaker than the six-year backtest
  suggests, as D2 found. The response the queue sets for a weak result is less concentration,
  not more signals; it is the user's choice of risk.

**Round 78, the last session ideas** (`research/round78_session_momentum.py`).

| Design | Multi-horizon: 14-day better | Mean yearly return | R54b: 14-day better | Mean yearly return |
|---|---|---|---|---|
| C3c the ranking without weekend hours | 2/6 | +188 pts | 2/6 | +385 pts |
| C4 US-session strength over 7 days | 2/6 | -214 pts | 2/6 | -163 pts |
| C4 the same over 14 days | 3/6 | -127 pts | 3/6 | -22 pts |
| C4 falsification, Asian strength, 7 days | 2/6 | -448 pts | 2/6 | -96 pts |
| C4 falsification, Asian strength, 14 days | 1/6 | -178 pts | 3/6 | +44 pts |

None passed. C3c's higher mean return came almost all from 2020–21 (+4,630% and +3,684%) with
deeper drawdowns (55–57%) and worse 14-day windows; C4 and its falsification failed alike, so
there is no session effect in either direction.

**Round 79, re-picking at each session open** (`research/round79_session_repicks.py`, the
user's question): the rotation re-picked at 00:00, 08:00 and 13:00 UTC, as Asia, Europe and
the US open, instead of once a day.

| Bot | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day better |
|---|---|---|---|---|---|---|---|---|
| Multi-horizon | +3,239% | -44% | +96% | +946% | +82% | +79% | 51% | |
| ... re-picked at each session open | +2,016% | -55% | +123% | +820% | +97% | +76% | 61% | 3/6 |
| R54b | +1,614% | -49% | +59% | +626% | +233% | +48% | 52% | |
| ... re-picked at each session open | +1,285% | -56% | +111% | +490% | +205% | +42% | 60% | 2/6 |

It failed on both: three re-picks a day swapped coins on intraday noise, cost fees, and deepened
the worst drawdown by 8–10 points, as round 73's fixed-interval re-picks had.

**Round 80, a faster ranking** (`research/round80_faster_ranking.py`, the user asked for a
faster bot). The ranking leaning on more recent returns (each version replaces the ranking of
either bot, so the returns are the same on both, judged against each baseline):

| Ranking | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day better: multi / R54b |
|---|---|---|---|---|---|---|---|---|
| Multi-horizon 7/14/21 days, 2/1/1 | +3,239% | -44% | +96% | +946% | +82% | +79% | 51% | |
| F1 7/14/21 days, 4/1/1 | +2,273% | -44% | +107% | +908% | +47% | +75% | 53% | 2/6, 4/6 |
| F2 3/7/14 days, 2/1/1 | +5,246% | -47% | +119% | +585% | +119% | -13% | 56% | 2/6, 3/6 |
| F3 3 and 7 days | +4,567% | -42% | +125% | +519% | +83% | -13% | 58% | 1/6, 2/6 |

None passed. The faster rankings won big in the strongest bull year and lost in the last one
(-13% against +79%), with deeper drawdowns and fewer winning 14-day windows (53% against 56%):
short returns pick up the coins of the moment, which reverse as often as they run.

**Round 81, the live bot faster** (`research/round81_faster_r54b.py`): each of R54b's three
speeds turned up, against R54b.

| Design | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day better |
|---|---|---|---|---|---|---|---|---|
| R54b | +1,614% | -49% | +59% | +626% | +233% | +48% | 52% | |
| S1 the rotation on 10-day returns | +2,443% | -52% | +160% | +751% | +213% | +7% | 53% | 2/6 |
| S2 BTC's filter on 5/20-day EMAs | +1,292% | -56% | +48% | +425% | +79% | +27% | 58% | 1/6 |
| S3 the book on 7/28-day trends | +1,625% | -51% | +59% | +609% | +240% | +43% | 52% | 1/6 |
| S4 all three | +2,125% | -61% | +146% | +520% | +134% | -6% | 61% | 1/6 |

None passed. As in round 80, faster settings won more in some strong years and lost in the
choppy last one (S1: +7% against +48%), with fewer winning 14-day windows: the faster filter
whipsawed in and out (S2's worst drawdown 58%), and the faster book changed nothing.

**Round 82, the two bots combined** (`research/round82_combined.py`, the user's question).

| Design | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day better: multi / R54b |
|---|---|---|---|---|---|---|---|---|
| R54b | +1,614% | -49% | +59% | +626% | +233% | +48% | 52% | |
| Multi-horizon (2/1/1) | +3,239% | -44% | +96% | +946% | +82% | +79% | 51% | |
| K1 half the rotation on each bot's picks | +2,281% | -46% | +77% | +780% | +147% | +64% | 51% | 2/6, 2/6 |
| K2 one ranking summing both (2/2/1) | +2,727% | -36% | +90% | +792% | +94% | +78% | 52% | 4/6, 3/6 |

Neither passed C1 against either bot. Both land between the two, as a blend would: K2 made more
than R54b in 5 of 6 years and had the mildest crash year of any version (-36%), but its median
14-day window beat R54b's in only 3 of 6 years.

The user chose K2 for the competition account on 6 October 2026 (`config/comp.json`; R54b kept
in `config/comp_r54b.json`). Its holdout, run then: +136% and +191% against R54b's +124% and
+187%, with a better yearly composite (3.00 and 3.49 against 2.80 and 3.31) and median 14-day
composite (6.10 and 5.33 against 5.21 and 3.10) in both years. Before the switch, the bot with
every research option off reproduced R54b's backtests exactly in all 8 years, and the new
config file reproduced K2's.

**Round 83, correlation and convex optimisation on K2** (`research/round83_correlation.py`;
`mean_variance` and `hrp_weights` in `bot/optimize.py`). Earlier optimisers ignored which coin
was strongest; these do not.

| Design | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day better |
|---|---|---|---|---|---|---|---|---|
| K2 | +2,727% | -36% | +90% | +792% | +94% | +78% | 52% | |
| O1 the second pick less correlated with the first | +2,727% | -36% | +90% | +792% | +99% | +78% | 52% | 1/6 |
| O2 mean-variance over the top 4, momentum as expected return | +3,350% | -36% | +98% | +1,008% | +91% | +47% | 51% | 2/6 |
| O2b the same over the top 2 | +2,704% | -38% | +110% | +988% | +77% | +60% | 53% | 2/6 |
| O3 hierarchical risk parity for the long-short book | +3,226% | -42% | +125% | +749% | +134% | +66% | 50% | 2/6 |

None passed C1. O1 changed almost nothing: K2's top two are rarely the most correlated pair.
O2 and O3 added return in some years (O2 +136 points a year on average, O3 +86 with the
lowest worst drawdown, 50%), but the median 14-day window was better in only 2 of 6 folds:
over two weeks the optimised weights did no better than equal ones. Built and tested
(`tests/test_optimize.py`); both stay research options. The hierarchical risk parity splits
along the cluster tree rather than the halves of the ordered list, which separated
near-identical coins in a first version caught by its test.

**Round 84, a wider rotation** (`research/round84_wider_rotation.py`, the user asked for more
coins or shorts in the long-only rotation), on K2:

| Design | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day windows won | 14-day better |
|---|---|---|---|---|---|---|---|---|---|
| K2 (2 coins, long only) | +2,727% | -36% | +90% | +792% | +94% | +78% | 52% | 56% | |
| M1 the top 3 | +3,117% | -30% | +85% | +633% | +144% | +61% | 46% | 57% | 3/6 |
| S1 bear-market shorts on the 2 weakest | +1,708% | -60% | +42% | +324% | +11% | +58% | 72% | 52% | 1/6 |
| S2 a bear-market short basket | +2,186% | -19% | +66% | +568% | +68% | +143% | 58% | 56% | 4/6 |
| S3 an always-on short leg, 30% | +917% | -23% | +90% | +420% | +80% | +38% | 43% | 58% | 2/6 |

None passed. Three coins (M1) cut the worst drawdown to 46% and the crash year to -30% for
about the same return, but beat K2's median 14-day window in only 3 of 6 folds. Shorting the
weakest coins in bear markets (S1) was squeezed again (-60% in 2021–22); the basket of
downtrends (S2) helped the crash years but deepened the worst drawdown; and the always-on short
leg (S3) cut risk by giving up most of the bull years.

**Round 85, shorting in the 3-coin rotation** (`research/round85_shorts_three_coins.py`). The
user kept K2 with 3 coins (round 84's M1, uncommitted) and asked for shorts in it:

| Design (3 coins and) | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day better |
|---|---|---|---|---|---|---|---|---|
| nothing (the baseline) | +3,117% | -30% | +85% | +633% | +144% | +61% | 46% | |
| X1 a 20% short leg, weakest by return | +1,500% | -22% | +83% | +398% | +134% | +33% | 35% | 1/6 |
| X2 the same, weakest by K2's score | +1,406% | -25% | +86% | +394% | +127% | +38% | 36% | 1/6 |
| X3 the same, coins in their own downtrend only | +1,815% | -29% | +88% | +471% | +155% | +26% | 36% | 2/6 |
| X4 a bear-market short basket | +2,380% | -9% | +58% | +455% | +92% | +110% | 55% | 4/6 |
| X5 X3 with the rotation at 80% | +2,252% | -35% | +101% | +554% | +168% | +28% | 42% | 2/6 |
| X6 X3 with the top 4 | +1,307% | -25% | +37% | +325% | +171% | +43% | 36% | 1/6 |

None passed. The always-on short legs cut the worst drawdown by about 10 points and gave up a
third to a half of the bull years; the bear-market basket turned 2021–22 into -9% and made
+110% in 2025–26, but deepened the worst drawdown to 55%.

The 3-coin bot's split between the rotation and the book, for information:

| Rotation | 50% | 60% | 65% | 70% | 75% | 80% | 90% |
|---|---|---|---|---|---|---|---|
| Six years | +47,012% | +77,584% | +91,553% | +119,017% | +145,396% | +178,805% | +254,945% |
| Worst drawdown | 38% | 42% | 45% | 46% | 49% | 51% | 55% |
| 14-day windows won | 58% | 58% | 58% | 57% | 56% | 56% | 54% |

With three coins the rotation is less concentrated, so at 80% it has the drawdown K2 has at
70% with two (51% against 52%) and more return (+178,805% against +105,989%).

The user chose 3 coins at 75% for the competition account on 6 October 2026
(`config/comp.json`; K2 kept in `config/comp_k2.json`). Its holdout, run then, was mixed:
2018–19 +91% (yearly composite 2.62, median 14-day 5.22) against K2's +136% (3.00, 6.10) and
R54b's +124% (2.80, 5.21); 2019–20 +219% (4.41, 6.44) against +191% (3.49, 5.33) and +187%
(3.31, 3.10). The bot with every research option off reproduced R54b's and K2's backtests
exactly before the switch, and the new config file reproduced its own.

**The queue, in sum.** Nothing from parts B, C or E passed C1–C5 on either bot (the nearest, E5's
capitulation buys, failed the holdout and moved returns by a few points). The validation audit
found a real trend-following edge, strongest in trending markets and weaker in the last two
years. The operational breakers (part A) are ready, off until a testing-account cycle.

## Scalping, long and short (H86)

The user asked for scalping strategies, long and short. `research/h86_scalping.py` tested seven
on 5-minute bars of 16 coins, 2020–26, each written down before running: VWAP snap-backs,
US-open range breakouts, volume-confirmed momentum bursts, RSI(2) pullbacks in the 4-hour
trend, altcoins following a sharp BTC bar, fading liquidation cascades and blow-off tops, and
Bollinger squeeze breakouts. Longs and shorts were judged apart: the mean net return per trade
had to be positive in at least 5 of the 6 folds.

All 14 failed. The gross edge per trade was between -6 and +4 basis points (blow-off shorts,
221 trades, lost 36 before costs);
a market round trip costs about 27 (0.1% a side plus half the spread). Even with limit orders
for longs at 0.05% a side and no spread, a best case, every long lost 7 to 12 basis points a
trade; shorts are market-only on Roostoo. The one positive fold (cascade fades in 2024–25,
+20 basis points over a few dozen trades) was alone among the six. Moves inside an hour are
too small against Roostoo's fees to trade, which agrees with H60's medium-frequency results.

## Opening-range breakouts (H87)

The user asked for an ORB test. `research/h87_orb.py` ran the full grid on 5-minute bars of 16
coins, 2020–26, written down before running: the Asia (00:00 UTC), London (08:00) and US
(13:30) opens held to 8 hours after, and the UTC day held to the next 00:00; ranges of 15, 30
or 60 minutes (60, 120 or 240 for the day); three exits (a 2R target, held to the session's
end, a stop at the range's middle); and four filters (none, the 1-day/4-day trend, twice the
usual volume, and acting only at hour closes as the bot does). That is 252 designs, longs and
shorts apart.

None of the 252 passed: none made money per trade after costs in even 4 of the 6 folds. The
session opens had no edge before costs (the best of any of them, +5.5 basis points a trade,
against about 27 in costs). The UTC-day breakout for longs, with the trend filter, had the
only real gross edge, +23 basis points a trade on a 240-minute range, but that is the trend
already, mostly from 2020–21 (+91 net that year; -42 in both 2021–22 and 2025–26), and after
costs it lost 4 a trade. Shorts lost in every design.

## Williams %R (H88, round 86)

The user asked for Williams %R. `research/h88_williams_r.py` tested five designs on hourly bars
of 49 coins, 2020–26, longs and shorts apart, each with lookbacks of 10, 14 and 21: buying
pullbacks (%R below -80) in the bot's trend, breakouts through -20, plain reversals (below -95,
above -5), a 14-day %R near its high with an hourly pullback, and the 14-day %R alone (in when
it crosses above -20, out below -50). Only the last, on longs, had an edge: +220 basis points a
trade after costs, positive in 5 of 6 folds with 14- and 21-day lookbacks but 4 of 6 with 10, so
it failed. The faster designs earned 0 to 15 basis points a trade before costs of about 25.

That slow signal is a breakout, close to what the rotation already does, so round 86
(`research/round86_williams_r.py`) put it in the rotation, on the live bot (K2, 3 coins at 75%)
and on R54b: new picks only near their 14-day high (%R at or above -20), held picks leaving
below -50, and both. All three failed C1 and C2 on both bots. The filter kept the rotation out
of coins that were rising but not at their highs, and halved the bull years (2022–23: +10%
against +92%); the exit sold picks in ordinary pullbacks. The options (`rotation_wr_hours`,
`rotation_wr_entry`, `rotation_wr_exit`) stay in `bot/config.py`, off.

## RSI(3), EMA and VWAP dip-buying (H89)

The user sent a published QQQ strategy: buy when RSI(3) closes below 22, the 100-day EMA is
above its level 5 days ago and the close is 1% or more below the 10-day VWAP; sell when RSI(3)
closes above 55. `research/h89_rsi3_vwap.py` ran it on the bot's 49 coins on daily, 4-hour and
hourly bars, with the mirror for shorts, and neighbours (RSI 18 and 26, gaps of 0.5% and 2%).

On daily bars the longs made +149 basis points a trade after costs and won 63% of trades, alike
in all four neighbours, but lost in both bear years (2021–22 -133, 2022–23 -65), so 4 of 6
folds: a fail, though the nearest of anything since round 85. The shorts lost 2.4% a trade
(bull-market pops kept rising). On 4-hour and hourly bars the edge before costs (+24 to +45
basis points for longs) was mostly eaten by them; no version passed. Our QQQ file has closes
only, so the original was not re-run on QQQ.

## Order flow (H90)

The user sent a list of 62 order-flow concepts. Binance's bars carry each hour's taker-buy
volume and trade count, so `research/h90_order_flow.py` built 17 signals from those that can be
measured: bar delta, CVD and its divergences, absorption at 7-day extremes, exhaustion, Asia's
session delta, large average trade size, price-volume divergence, initiative and responsive
moves around the 7-day value area, the developing POC, the week's anchored VWAP and its bands,
liquidity sweeps, failed auctions, trapped traders and structure shifts. About 20 concepts need
the order book or tick-by-tick bid and ask prices, which we do not have (footprints, DOM, level
2, icebergs, spoofing, liquidity pulling and stacking, imbalances). Each signal traded on hourly
bars of 49 coins, held 24 hours with a 3-ATR stop (neighbours: 12 and 48 hours), longs and shorts
apart; then votes (2 or 3 agreeing within 6 hours) and all 136 pairs.

No single signal passed. Most earned less than the 25 basis points a trade that costs take;
exhaustion longs (buying a 3x-volume, 2.5-sd drop that closes in the bar's top half) made +127 a
trade in 5 of 6 folds, but not with every hold, and lost in both holdout years. The votes
failed. Three of the 272 pair-sides passed with all three holds, about as many as chance gives,
and on the holdout years (2018–20, fewer coins) only one held up: initiative above the 7-day
value area together with a break of structure in a downtrend, for longs (+48 and +187 basis
points a trade at 24 hours, +5 and +71 at 12, -27 and +276 at 48). It averages +24 basis points
a trade on about 220 trades a year across 49 coins, too small and too rare to matter beside the
rotation; it is not in the bot.

## The book from stochastic calculus (round 87)

The user asked to make the long-short book stronger with stochastic calculus. Treating each coin
as a diffusion, dS/S = mu dt + sigma dW with jumps, `research/round87_book_sde.py` tested six
changes, judged on C1–C5 on the book alone and on the live bot (options in `bot/config.py`, off):
Merton sizing (weights mu / sigma^2, the drift read from the EMA gap), sizing by the t-statistic
of a Kalman filter's slope, trading only coins with a variance ratio of at least 1, no shorts
where jumps are over 30% of the variance (bipower variation), scaling the book by BTC's
volatility forecast, and the first, third and fourth together.

All six failed C1 on both bots. On the book alone (the baseline: +484%, +32%, +7%, +84%, +27%, +50%;
worst drawdown 43%): Kalman sizing made the most (+692% in 2020–21, +71% in 2025–26) but less in 2021–24 and
a worse drawdown (49% against 43%); Merton sizing concentrated the book in the strongest trends
and lost in 2022–23; the variance-ratio filter kept the book mostly in cash (drawdown 12%, but
+60% in 2020–21 against +484%); the jump filter changed little; volatility management cut the
drawdown to 36% and the return with it. A diffusion with a drift read from the trend is what the
book already assumes; the refinements traded return for risk or the other way, never both.

## Forecasting the range (H91, round 88)

The user asked for a range forecast. `research/h91_range_forecast.py` forecast each coin's daily
volatility with a HAR model of the log Parkinson range (yesterday's, last week's and last month's
values, pooled over 49 coins and refitted before each fold, walk-forward). It explained 29–46% of
the next day's log range out of sample, against -5 to 23% for yesterday's range and 15–32% for
the 30-day mean: a real forecast. Its 20% bands held less well in recent years (the top touched
on 9–12% of days in 2020–23 but 18–20% in 2024–26).

Trading the band failed: fading an hourly close beyond it lost 28–41 basis points a trade after
costs, and following it lost 4 to 69, longs and shorts, at bands touched on 10, 20 and 30% of
days. Round 88 (`research/round88_book_range.py`) weighted the book by the HAR forecast instead
of the last week's volatility (option `ls_sizing` "har", off): on the book alone it was better in
4 of 6 folds and moved returns by a few points; on the live bot 2 of 6. A better volatility
forecast barely changes inverse-volatility weights, which only compare coins with each other.

## Machine learning (H92, H93)

The user asked for machine learning. `research/h92_ml_ranking.py` built 24 features a coin a day
(returns over 1 to 30 days and against BTC, volatility, range, volume, taker-buy share, Williams
%R, the book's EMA gap, the trend's t-statistic, the variance ratio, skewness, the best day of
the month, BTC's state) for the bot's 49 coins, and trained a ridge regression, LightGBM and a
neural network (an MLP, three seeds) to rank coins on the next 7 days' return, walk-forward (each
fold predicted by a model trained only on earlier days, with a 7-day gap).

The models ranked far better than K2 on average: a daily rank correlation with the next 7 days
of +0.05 to +0.12 in every fold, against about -0.02 for K2. But a rotation on their top 3 made
less than K2's in 5 or 6 of the 6 folds (LightGBM +707% against +4,895% in 2020–21). The rotation
earns from the few coins that run away, which momentum finds; the models learned the average
ordering, mostly short-term reversal and low volatility.

`research/h93_ml_book.py` used them where an average edge pays, in a long-short book, in the same
daily simulation as the live book: long the top fifth, short the bottom fifth (LightGBM, ridge),
and the live book with an ML veto. None beat the book in 5 of 6 folds: the ML books lost 51–56%
in 2023–24, when the coins they shorted rallied; the veto cut the worst drawdown from 47% to 24%
but the return too (+124% against +407% in 2020–21). A classifier trained to find each week's
top tenth beat K2's rotation in 2 of 6 folds. Reinforcement learning was not tried: six years
give about 300 independent weekly decisions, too few to train a policy that does not memorise.

## Machine learning for the book alone (H94, round 89)

The user asked for ML, deep learning and reinforcement learning on the book alone.
`research/h94_ml_book.py` learned the book's own decisions, walk-forward as H92, in H93's daily
simulation: each coin's direction by LightGBM (T1) and by a neural network (T2), meta-labels
keeping only the trend positions a model expects to pay (T3) or sized by its confidence (T4),
and a learned controller (Q1, one-step fitted Q, a contextual bandit) choosing each day between
the whole book, its longs, its shorts or nothing from the market's state. T1, T2 and T4 lost
(worst drawdowns 70–86%); T3 cut the worst drawdown from 47% to 30% but beat the book's return
in 2 of 6 folds. Q1 beat it in 4 of 6 on return and 5 of 6 on the median 14-day return, with a
40% worst drawdown, and its neighbours (`research/h94_q1_robust.py`: reward horizons of 5 and 10
days, trees of 3 and 15 leaves) in 4–5 of 6; all lost 2020–21, the year with the least training.

What it learned is that the book's shorts lose while most coins are in uptrends, its longs earn
nothing while most are in downtrends, and shorts lose after BTC has fallen 15% in 30 days.
Round 89 (`research/round89_book_breadth.py`) made those plain rules (options
`ls_breadth_align`, `short_btc_crash`, off): longs only while at least half the coins trend up,
shorts only while fewer do (G1), and no shorts after a BTC crash (G2). On the book alone both
beat it in 5 of 6 folds with a shallower drawdown but won fewer 14-day windows (C2); on the live
bot G2 passed C1–C4. Both lost to the book in both holdout years, on both bots (book alone, median
14-day composite 5.30 and 1.43 against 6.00 and 3.90 for G1), so the pattern was 2020–26's and
not a rule. The book stays as it is. Reinforcement learning proper (a multi-step policy) was not
tried, for H93's reason.

## Correlation on the book alone (round 90)

Rounds 74 and 83 had weighted the book by correlations only inside the live bot. Round 90
(`research/round90_book_correlation.py`) ran them on the book alone, with a new use of
correlation as a danger signal (option `ls_corr_cut`, off): equal risk contribution (P1),
minimum variance (P2), hierarchical risk parity (P3), the book halved while the 10 most traded
coins' mean 72-hour correlation is above 0.7 (K1), or its shorts dropped instead (K2c).

P1–P3 and K2c failed C1 on both bots; hierarchical risk parity added the most return (+36 points
a year alone, +81 in the live bot) but won the median 14-day window in 3 of 6 folds. K1 was the
best result for the book so far: better in 5 of 6 folds on the 14-day yardstick, a worst
drawdown of 36% against 43%, more 14-day windows won, and its neighbours (0.6, 0.8) held. It
failed the holdout: 2018–19 +75% (14-day composite 4.42) against +104% (6.00), 2019–20 +67%
(3.76) against +70% (3.90), though with drawdowns of 20% and 18% against 29% and 32%. Halving the
book when correlations spike reliably cuts risk, but in the holdout it gave up more return than
it saved; in the live bot it was better in 3 of 6 folds.

## More return: the split (round 91)

The user asked for more return. `research/round91_more_return.py` mapped the split on the live
bot (K2, 3 coins), over the six folds and the 2018–20 holdout:

| Rotation | 6-year total | Worst drawdown | Crash year 2021–22 | Mean median 14-day composite | Holdout 2018–19 / 2019–20 |
|---|---|---|---|---|---|
| 75% (live) | +145,396% | 49% | -33% | 8.72 | +91% / +219% |
| 80% | +178,805% | 51% | -38% | 8.95 | +88% / +232% |
| 85% | +217,895% | 54% | -40% | 9.73 | +87% / +245% |
| 90% | +254,945% | 55% | -44% | 9.29 | +84% / +250% |
| 95% | +299,393% | 57% | -47% | 9.76 | +87% / +263% |
| 75%, hierarchical risk parity book | +168,205% | 49% | -39% | 9.63 | +97% / +226% |
| 85%, hierarchical risk parity book | +199,090% | 54% | -45% | 10.00 | +89% / +232% |

Each 5% more in the rotation adds about 2–3 points of worst drawdown; the 14-day composite rises
to 85% and then flattens. 100% is not supported (the book's share divides by it). The user chose
to keep 75%.

## A book that makes more (rounds 92 and 93)

The user asked to keep working on the book, return first. Round 92
(`research/round92_book_return.py`): only the 10 strongest trends (N1), longs on a faster trend
than shorts (N2), trends younger than 14 days at double weight (N3), and N1 with N2 (N4). Round 93
(`research/round93_book_dual.py`): dual momentum, a long also beating BTC over 30 days and a short
lagging it, the rest in cash (D1) or filling the book (D2), and fresh trends at three times their
weight (F1). On the book alone (baseline +484%, +32%, +7%, +84%, +27%, +50%; worst drawdown 43%):

| Design | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Worst drawdown | 14-day better |
|---|---|---|---|---|---|---|---|---|
| N1 10 strongest trends | +772% | +3% | -11% | +20% | +17% | +44% | 45% | 1/6 |
| N2 faster longs | +497% | +26% | +9% | +82% | +31% | +45% | 37% | 3/6 |
| N3 fresh trends 2x | +487% | +52% | +13% | +111% | +17% | +53% | 45% | 2/6 |
| N4 N1 + N2 | +766% | +7% | +0% | +39% | +2% | +42% | 36% | 1/6 |
| D1 dual momentum, cash | +112% | -14% | +3% | +15% | -1% | +8% | 46% | 0/6 |
| D2 dual momentum, filled | +55% | -31% | -23% | +22% | +1% | -6% | 65% | 0/6 |
| F1 fresh trends 3x | +459% | +55% | +21% | +136% | +18% | +50% | 46% | 3/6 |

None passed C1 on either bot. Concentration paid only in the 2020–21 bull run. Dual momentum
failed worst: coins beating BTC are few, and the book's edge is breadth across trends, not
picking leaders (the rotation's job). Fresh trends were the one consistent effect, better in
2021–24 at both 2x and 3x, but worse in 2024–25 and the 14-day windows did not follow. Options
`ls_top_n`, `ls_fresh_days`, `ls_fresh_boost`, `ls_rel_hours`, `ls_rel_fill`, all off.

## Trend quality for the book (rounds 94 to 96)

Round 94 (`research/round94_book_chop.py`) went after the book's choppy years: hysteresis (a coin
keeps its side until its EMA gap is 1% past zero the other way) and weights times Kaufman's
efficiency ratio (net move over path length) of each coin's hourly returns, re-scaled to the
book's gross and capped at 3x. Hysteresis lost return. The efficiency ratio added the most of
any book design so far, but failed C1 (the median 14-day window better in 2 of 6 folds).

The user asked for return first, so round 95 (`research/round95_efficiency.py`) and round 96
(`research/round96_book_quality.py`) judged on return: six-year totals, worst drawdowns and the
2018–20 holdout. On the book alone (baseline +2,794%, worst drawdown 43%, holdout +104% / +70%):

| Design | Six years | Worst drawdown | Return better | Holdout 2018–19 / 2019–20 |
|---|---|---|---|---|
| Efficiency ratio, 7 days | +4,463% | 38% | 3/6 | +78% / +72% |
| Efficiency ratio, 14 days | +5,560% | 41% | 4/6 | +84% / +91% |
| Efficiency ratio, 30 days | +6,666% | 31% | 3/6 | +158% / +99% |
| Efficiency 14 days + fresh trends 3x | +12,605% | 38% | 5/6 | +43% / +131% |
| R^2 of a line through log price, 14 days | +4,224% | 44% | 4/6 | +132% / +158% |
| R^2, 7 days | +3,210% | 37% | 3/6 | +109% / +106% |
| R^2, 30 days | +2,112% | 49% | 2/6 | +94% / +88% |
| Soft regime tilt (against-trend side at half) | +3,555% | 39% | 4/6 | +78% / +71% |

Weighting the book towards clean trends added six-year return at every efficiency-ratio window
with a drawdown no deeper, and the 30-day version won both holdout years by a wide margin; but
year by year it lost some folds by a few points, and the 14-day windows did not improve. Inside
the live bot (the book at 25%) the 14-day version made +158,256% against +145,396% with the same
worst drawdown and both holdout years better. "Efficiency + HRP" equalled round 90's HRP book:
the HRP weights override the efficiency tilt. Options `ls_hysteresis`, `ls_er_hours`,
`ls_r2_hours`, `ls_regime_tilt`, off.

Round 97 (`research/round97_efficiency_stress.py`) stress-tested the 30-day efficiency ratio on
the book alone. Every variant kept more six-year return than the book and no deeper a worst
drawdown: 21 days +5,037% (37%), 45 days +3,368% (41%), the weight cap at 2x +5,772% (35%) and 5x
+7,004% (31%), and at double fees +5,793% (34%) against the book's +2,452% (44%) at double fees.
The 30-day variants won both holdout years (+142% to +166% against +101–104%, and +93% to +102%
against +68–70%); the 21- and 45-day ones one each. The median 14-day window was better in 3–4
of 6 folds and the share of positive windows 58% against 57%. On return, risk, costs and the
holdout this is the strongest book design so far; it fails only C1, the 14-day yardstick.

The user deployed the 30-day efficiency-ratio book on 7 October 2026 (`config/comp.json`, with
`ls_er_hours` 720; the previous settings kept in `config/comp_k2_3.json`). In the full backtester
the live bot with it made +163,209% over six years against +145,396%, worst yearly drawdown 50%
against 49%; holdout +102% and +214% against +91% and +219%. On the competition's 14-day
yardstick it was not better: the median 14-day composite was higher in 2–3 of 6 folds and lower
in both holdout years (4.26 and 4.86 against 5.22 and 6.44). A fresh run of the live settings,
outside the result cache, reproduced the cached backtests exactly with every option off and with
the efficiency ratio on.

## The split with the efficiency book (round 99)

The user asked to lower the rotation's share. `research/round99_lower_split.py` ran the live bot
with the efficiency-weighted book and the rotation at 50% to 75%:

| Rotation | Six years | Worst drawdown | Crash year 2021–22 | 14-day windows won | Mean median 14-day composite | Holdout 2018–19 / 2019–20 |
|---|---|---|---|---|---|---|
| 75% | +163,209% | 50% | -37% | 56% | 9.34 | +102% / +214% |
| 70% | +131,086% | 48% | -36% | 56% | 9.17 | +110% / +211% |
| 65% | +115,077% | 45% | -28% | 58% | 9.49 | +109% / +193% |
| 60% | +98,263% | 42% | -25% | 58% | 10.01 | +112% / +189% |
| 55% | +102,471% | 39% | -17% | 59% | 10.48 | +121% / +177% |
| 50% | +73,356% | 37% | -20% | 59% | 10.73 | +120% / +169% |

With the efficiency book the 14-day composite rose as the rotation's share fell, unlike with
the old book (round 91), where it rose with the share. The user chose 55% on 7 October 2026
(`config/comp.json`; 75% kept in `config/comp_er_75.json`).

Round 98 (`research/round98_efficiency_more.py`) took the efficiency ratio further on the book
alone: keeping only the cleaner half of the coins lost (+264% over six years); squaring the
tilt (X2) made +11,632% with a 34% worst drawdown, both holdout years better (+181%, +117%) and
its neighbours (powers 1.5 and 3) alike; adding fresh trends at double weight (X3) +10,182%,
31%, holdout +121% and +121%. Both beat the inverse-volatility book on the 14-day yardstick in 4
or 5 of 6 folds. Inside the bot at 75% they made +179,359% and +167,604% against the efficiency
book's +163,209%.

Round 100 (`research/round100_book_convex.py`) put convex optimisation on the efficiency book,
keeping the efficiency ratio as an input: risk contributions in proportion to it (C1, Spinu's
risk budgeting, `risk_budget_weights` in `bot/optimize.py`) and mean-variance with its normal
scores as expected returns (C2). Both were judged against the efficiency book (the comparison
labelled "before" in the output is that book too: the harness then read `config/comp.json`,
which the deployment had changed; it now reads fixed files). On the book alone C1 made +888%
over six years and C2 +5,516% against +6,666%, with deeper drawdowns (38% and 40% against 31%)
and both holdout years worse. In the bot at 75%, C2 at risk aversion 0.5 made +202,219% against
+163,209% and the 14-day windows better in 4 of 6 folds, but the design itself 2 of 6 and one
holdout year of two. The plain efficiency tilt stays.

Round 101 (`research/round101_book_at_55.py`) ran round 98's two stronger books at the live
split, against the live bot (55% rotation, the efficiency book at 45%). The squared tilt (X2)
made +114,066% over six years against +102,471%, both holdout years better on return and the
14-day composite (+125% / +216% against +121% / +177%; its neighbours alike), but the median
14-day window better in only 3 of 6 folds, a worst drawdown of 41% against 39% and a crash year
of -25% against -17%. Efficiency with fresh trends (X3) made +108,576%, better in 4 of 6 folds on
the 14-day yardstick, one holdout year of two. Neither passed C1, so the live book stays; X2 is
the next candidate if more return is wanted for a deeper crash year.

## An aggressive third sleeve (H95, H96)

The user proposed three sleeves: the rotation 30%, an aggressive sleeve 30% and the book 40%.
`research/h95_aggressive_sleeve.py` and `research/h96_more_aggressive.py` tested eleven
candidates for the aggressive 30%, each mixed with the full backtester's 30/40 rotation and
efficiency book (daily rebalanced), against the live bot (55% rotation, 45% efficiency book).
Rule: the median 14-day composite better in at least 5 of 6 folds and both holdout years, a
worst drawdown at most 2 points deeper.

| 30/30/40 with | Six years | Worst drawdown | Mean median 14-day composite | 14-day better | Holdout |
|---|---|---|---|---|---|
| (live 55/45) | +102,974% | 39% | 10.48 | | +121% / +177% |
| A1 H61's swing ensemble | +20,014% | 28% | 9.42 | 1/6 | +87% / +118% |
| A2 14-day Williams %R breakouts | +36,258% | 37% | 6.49 | 0/6 | +113% / +232% |
| A3 the RSI(3) daily dip-buy | +11,094% | 35% | 8.08 | 1/6 | +86% / +119% |
| A4 fast momentum (top 2, 3 and 7 days) | +105,455% | 47% | 11.23 | 1/6 | +139% / +190% |
| A5 A1–A4 together | +32,763% | 34% | 10.41 | 1/6 | +107% / +163% |
| A6 all-in momentum (top 1, 3 and 7 days) | +62,645% | 49% | 8.00 | 0/6 | +100% / +271% |
| A7 all-in K2 (top 1) | +140,058% | 44% | 8.67 | 1/6 | +117% / +238% |
| A8 meme momentum (top 2 memes) | +49,632% | 33% | 8.52 | 1/6 | +84% / +118% |
| A9 the 5 strongest trends | +19,226% | 33% | 8.15 | 2/6 | +149% / +133% |
| A10 the efficiency book cubed | +45,468% | 33% | 11.91 | 4/6 | +147% / +153% |
| A11 breakouts in the most volatile third | +37,292% | 38% | 5.49 | 0/6 | +104% / +237% |

None passed. Taking 25 points from the rotation cost more than any sleeve earned: the control
(30% rotation, 70% book, no sleeve) made +36,530%. A7, the rotation concentrated in one coin,
was the only one to make more than the live bot (+140,058%), at a deeper drawdown (44%) and a
worse 14-day yardstick in 5 of 6 folds. A10, in effect a larger efficiency book, came nearest
on the rule (4 of 6, both holdout years, a shallower drawdown) for well under half the return.
On their own the aggressive sleeves drew down 62–85% (A3 lost money over the six years).

## A dollar-neutral book (round 103)

The user asked about a dollar-neutral book. `research/round103_dollar_neutral.py` (option
`ls_neutral`, off): the book's longs and shorts each scaled to half its gross (DN1), or long the
stronger half of the coins by trend and short the weaker half (DN2), no short where funding is
negative. On the efficiency book alone DN1 made +109% over six years and DN2 -25%, against
+6,666%, with deeper drawdowns (46% and 36% against 31%); in the live bot +23,564% and +14,041%
against +102,471%, both with a worse crash year (-36% and -34% against -17%). Neutral, the book
gave up both of its jobs: riding the market in bull years and shorting it in bear years. Its
edge is the direction of each coin's trend, not one coin against another; the weak half of
crypto bounces too often to short as a hedge.

## Winners against losers, and a regime-driven split (rounds 104 to 106)

The user asked for a book long the winners and short the losers, re-chosen dynamically, and
for the 55/45 split to follow a fast regime. Round 104 (`research/round104_winners_losers.py`)
replaced the book with the best and worst coins on the rotation's ranking, re-ranked every hour:
long 5 / short 5 dollar neutral (and 3 or 8 a side), shorts only in bear markets, or only in
downtrends. All lost everything over the six years on the book alone. A diagnosis run found the
cause: re-ranking hourly, the book traded about five times its value a day (498%, against 25%
for the efficiency book), and the fees ruined it. Round 105 (`research/round105_dynamic_winners.py`)
made the count dynamic (winners and losers by how far each coin's standardised score stands from
the rest, entering at +-1 sd and leaving at 0) with that hysteresis, hourly or daily, by sign
and size of the score, or as dual momentum. Turnover fell to 73–92% a day, still three times the
book's, and every design still lost on the book alone or made a fraction of it (the best, every
coin by its score re-ranked daily, +237% over six years against +6,666%, worst drawdown 60%);
in the live bot the best, dual momentum, made +58,087% against +102,471% with a worse crash
year. Crypto's losers bounce too often to short, and rankings move too fast to trade cheaply.

Round 106 (`research/round106_dynamic_split.py`, option `split_regime`) set the rotation's share
every hour at 75% in a bull market, 55% neutral and 35% in a bear one, read from BTC against its
50- and 200-hour EMAs, from the share of coins up over 72 hours, or both, with neighbours 70/55/40
and 80/55/30. Every version did worse than the fixed split, by every measure: six years +13,473%
to +56,085% against +102,471%, worst drawdown 44–52% against 39%, the crash year -39% to -47%
against -17%, 14-day windows better in 0 or 1 of 6 folds. The fast regime flipped often and each
flip re-picked the rotation and moved a fifth of the account; the slow BTC filter the rotation
already has does the regime's job better.

On 7 October 2026, with the account at -4.2% and the market falling, the user expected the fall
to continue and switched the competition account to the efficiency book alone (rotation off;
`config/comp.json`, the 55/45 settings kept in `config/comp_er_55.json`). In the backtests the
book alone made +26% in the 2021–22 crash year against -17% for 55/45, and +6,709% over six years
against +102,471%: a defensive choice that gives up most of a rally. Of the two-week windows in
the 55/45 backtests that began 3–6% down after three days, 32% finished positive, with a median
of +1.2% over the remaining eleven days.

Round 107 (`research/round107_crash_detector.py`) tested detectors that switch the 55/45 bot to
the book alone and back. Handing the rotation's share to the book whenever the rotation's own
BTC filter is off (B1, round 61's option `ls_absorb_rotation`) made +97,846% over six years
against +102,471% fixed, a better crash year (-9% against -17%) and latest year (+92% against
+57%), better 14-day windows in 4 of 6 folds and both holdout years better on return (+145% and
+187% against +121% and +177%), but a deeper worst drawdown (47% against 39%, in 2024–25, when
the filter flickered). Confirmation by BTC's 200-day average (B2) was in between; the fast
detectors (BTC against its 50- and 200-hour EMAs, breadth, both) whipsawed and made a fifth to a
quarter as much.

## Validated on every crash (H98)

The user asked to validate long-short designs on the crashes themselves.
`research/h98_crash_validation.py` found every spell from October 2018 to October 2026 in which
BTC fell 25% or more below its running high (eight, from -25% to -77%, peak to trough) and
measured each design's chained equity over each, beside its yearly returns.

| Design | 2018 | 2019–20 | Jan 2021 | 2021 | 2021–22 | 2024 | 2025 | 2025–26 | Mean | Six years |
|---|---|---|---|---|---|---|---|---|---|---|
| BTC | -52% | -63% | -25% | -53% | -77% | -26% | -28% | -53% | | |
| Book alone (live) | +53% | +119% | +14% | +22% | +11% | -13% | +28% | +28% | +32.8% | +6,709% |
| Book, shorts on 168h/672h | +46% | +128% | +14% | +23% | +5% | -15% | +29% | +27% | +32.1% | +4,748% |
| Book, shorts on 120h/480h | +51% | +136% | +16% | +20% | +6% | -12% | +23% | +30% | +33.8% | +3,765% |
| Book on 120h/480h | +40% | +168% | +16% | +28% | -2% | -16% | +23% | +29% | +35.7% | +3,597% |
| Book on 72h/288h | +45% | +184% | +6% | +23% | -4% | -5% | +10% | +7% | +33.0% | +1,521% |
| 55/45 | +24% | +104% | +31% | +44% | -28% | -30% | +6% | +4% | +19.3% | +102,974% |
| 55/45, book in bears (B1) | +53% | +158% | +31% | +55% | -19% | -33% | +10% | +33% | +36.2% | +98,491% |
| B1 with shorts on 168h/672h | +47% | +175% | +30% | +55% | -19% | -34% | +10% | +28% | +36.8% | +99,558% |

Faster trends did not make the book better in crashes on average and cost it return in the
other years. Handing the rotation's share to the book while the BTC filter is off (B1) did best
over the crashes on average and kept nearly all of 55/45's six-year return; its weak crashes
were the 2021–22 bear market and the 2024 correction, where the book alone lost less.

On 7 October 2026 the user chose B1 for the competition account (`config/comp.json`: 55/45 with
`ls_absorb_rotation` 1.0; the book alone kept in `config/comp_book.json`).

`research/h99_attribution.py` split B1's return between its sleeves (each day's share times each
sleeve's own return, from backtests of each alone, summed by fold; the approximation's
compounded total is within a few points of B1 in most years). Over 2020–26 the rotation gave
+566 points and the book +271, about two-thirds and one third; the rotation earned most in the
bull years (90% of 2023–24's gain) and lost in the crash year (-23 points, when the book made
+38), and the two were close to even in 2025–26 (+40 and +38) and the 2018–19 holdout year, when
the book made most of it.

## An altcoin index for the switch (round 109)

The user asked whether the market's bleeding could be tracked through a crypto index. The
Nasdaq Crypto Index and the CME CF and Bloomberg Galaxy indices have no free history (and the
Nasdaq index is about three quarters BTC, so it would switch with B1's BTC filter). Round 109
(`research/round109_alt_index.py`, option `regime_index`) built an equal-weight index of the
bot's coins other than BTC and PAXG and moved B1's switch onto it ("alts"), or off when either
index is off ("either"), over every BTC crash since 2018 and every year:

| Switch | Mean over the 8 crashes | Crash year 2021–22 | Six years | Worst drawdown | Mean median 14-day composite |
|---|---|---|---|---|---|
| none (55/45) | +19.3% | -17% | +102,974% | 39% | 10.48 |
| BTC (B1, live) | +36.2% | -9% | +98,491% | 47% | 14.89 |
| altcoin index | +31.3% | -5% | +36,814% | 43% | 10.52 |
| either | +46.2% | +12% | +58,099% | 41% | 10.93 |

The altcoin index alone switched more often and cost the bull years. "Either" protected best in
crashes (positive in the crash year, +46% on average over the crashes) for about 40% less return
over six years and a lower 14-day yardstick. B1's BTC switch stays: the best 14-day yardstick of
all and nearly all of 55/45's return.
