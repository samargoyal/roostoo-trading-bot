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

Long-only trend following on hourly bars over the 20 most traded crypto pairs on Roostoo,
plus PAXG (gold-backed) as a defensive asset. The goal is a steady, positive return with
shallow drawdowns, because the competition ranks on return and then scores
`0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar`.

| Rule | Detail |
|---|---|
| Regime | BTC above its 200-hour EMA: up to 75% invested in up to 4 positions. Otherwise up to 25% in up to 3, with PAXG first in line. |
| Trend filter | A coin is eligible while its 50-hour EMA is above its 200-hour EMA and the close is above the 200-hour EMA. |
| Ranking | Lowest volatility of hourly returns over the last 168 hours first (the low-volatility effect; see [Strategy research](#strategy-research)). Free slots go to the best-ranked eligible coins. |
| Entry timing | No new entry while RSI(14) is above 70, or within 24 hours of a stop-loss exit on that coin. |
| Sizing | Weights proportional to `price / ATR(14)`, so each position carries similar risk, scaled to the exposure limit and capped at 15% per coin. |
| Exits | The 50-hour EMA falls below the 200-hour EMA, or the close drops 8 ATR below the highest close since entry. A held coin is never sold just for ranking lower. |
| Drawdown brake | At 4% below peak equity, trend positions are halved until the drawdown is back under 2%. |
| PAXG core | 5% of equity stays in PAXG at all times. |
| Activity rule | The competition requires trades on at least 8 days. If nothing has filled in the current 8-hour UTC block and less than 2 hours of it remain, the position furthest from its target is rebalanced. This guarantees at least 2 trades in every calendar day, whatever time zone is used. |

Trades are only placed when a holding is more than 4% of equity away from its target
(entries and exits always go through), which keeps fees down.

## Which assets it trades

The bot trades the 20 crypto pairs with the highest USD trading volume over the last 30
days, among Roostoo pairs with a bid-ask spread of at most 0.1% and at least 1000 hours of
price history, plus PAXG. As of 2 October 2026 that is:

> BTC, ETH, ZEC, SOL, XRP, NEAR, BNB, SUI, DOGE, UNI, ENA, AVAX, WLD, LINK, ADA, ARB, TAO,
> PUMP, LTC, TRX, and PAXG

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

- Over any 14-day window (the length of the competition) the median backtest return is
  close to zero: 46% of windows were positive in the falling year and 61% in the rising
  one. The worst 14-day loss was 6.6%.
- It gives up upside in strong bull markets. Average exposure is only about 22–25%, so from
  October 2024 to October 2025 it made 37% while simply holding BTC made 80% (though with
  a 31% drawdown against the bot's 13%).
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

1. Fetches the last 1000 closed hourly candles per pair from Binance and rebuilds the
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

$100,000 starting cash. "Basket" holds the same 20 coins plus PAXG in equal weights.

| Oct 2025 – Oct 2026 (falling market) | Taker | Maker | Hold BTC | Hold basket |
|---|---|---|---|---|
| Total return | 12.9% | 19.9% | -26.8% | -34.5% |
| Max drawdown | 12.3% | 10.3% | 53.7% | 64.9% |
| Sharpe | 0.98 | 1.34 | -0.46 | -0.42 |
| Sortino | 1.64 | 2.46 | -0.66 | -0.58 |
| Calmar | 1.05 | 1.92 | -0.50 | -0.53 |
| Composite score | 1.26 | 1.96 | -0.55 | -0.52 |
| 14-day windows: median return / worst | -0.1% / -4.2% | 0.0% / -4.1% | | |

| Oct 2024 – Oct 2025 (rising market) | Taker | Maker | Hold BTC | Hold basket |
|---|---|---|---|---|
| Total return | 37.2% | 47.6% | 79.5% | 38.2% |
| Max drawdown | 12.7% | 11.5% | 30.9% | 61.2% |
| Sharpe | 2.16 | 2.55 | 1.58 | 0.81 |
| Sortino | 4.09 | 4.90 | 2.54 | 1.19 |
| Calmar | 2.92 | 4.16 | 2.57 | 0.62 |
| Composite score | 3.16 | 3.97 | 2.26 | 0.90 |
| 14-day windows: median return / worst | +0.5% / -6.6% | +0.8% / -6.1% | | |

The bot traded about 5 times a day, with at least 3 trades on every day, and was on
average 22–25% invested. With the earlier momentum ranking the same backtests returned
1.8% and 43.2%, with drawdowns of 15.0% and 14.5%.

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
