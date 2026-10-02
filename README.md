# Roostoo trading bot

An autonomous trend-following bot for the [Roostoo](https://github.com/roostoo/Roostoo-API-Documents)
mock exchange, built by Team124 for the HK vs AU vs IN Quant Trading Hackathon. It trades
through the Roostoo REST API with no manual intervention: every order comes from the
strategy, and every order, decision and hourly equity point is recorded.

- [Strategy](#strategy)
- [How it works](#how-it-works)
- [Backtest results](#backtest-results)
- [Setup and running](#setup-and-running)
- [Configuration](#configuration)
- [Records for judging](#records-for-judging)
- [Tests](#tests)

## Strategy

Long-only trend following on hourly bars over 12 liquid pairs: BTC, ETH, SOL, BNB, XRP,
DOGE, SUI, AVAX, LINK, NEAR, ZEC and PAXG (gold-backed). The goal is a steady, positive
return with shallow drawdowns, because the competition ranks on return and then scores
`0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar`.

| Rule | Detail |
|---|---|
| Regime | BTC above its 200-hour EMA: up to 75% invested in up to 4 positions. Otherwise up to 25% in up to 2, with PAXG first in line. |
| Trend filter | A coin is eligible while its 50-hour EMA is above its 200-hour EMA and the close is above the 200-hour EMA. |
| Ranking | `0.5 x 72h return / 72h volatility + 0.5 x 168h return / 168h volatility`. Free slots go to the best-ranked eligible coins. |
| Entry timing | No new entry while RSI(14) is above 70, or within 24 hours of a stop-loss exit on that coin. |
| Sizing | Weights proportional to `price / ATR(14)`, so each position carries similar risk, scaled to the exposure limit and capped at 15% per coin. |
| Exits | The 50-hour EMA falls below the 200-hour EMA, or the close drops 8 ATR below the highest close since entry. A held coin is never sold just for ranking lower. |
| Drawdown brake | At 4% below peak equity, trend positions are halved until the drawdown is back under 2%. |
| PAXG core | 5% of equity stays in PAXG at all times. |
| Activity rule | The competition requires trades on at least 8 days. If nothing has filled in the current 8-hour UTC block and less than 2 hours of it remain, the position furthest from its target is rebalanced. This guarantees at least 2 trades in every calendar day, whatever time zone is used. |

Trades are only placed when a holding is more than 4% of equity away from its target
(entries and exits always go through), which keeps fees down.

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
| + 8 ATR stop, 72h/168h momentum, 4% rebalance threshold (current) | **+19.8%, 15.5%** | **+27.0%, 14.1%** |

Figures assume every order is a market order (0.1% fee plus slippage). Removing the drawdown
brake raised return slightly but pushed max drawdown to 24–27%, so the brake stays.

### Known weaknesses

- Profits come from a minority of strong trends. Over any 14-day window (the length of the
  competition) the median backtest return is close to zero: about half of the windows are
  positive, and the worst lost 5.6%.
- It gives up upside in strong bull markets. Average exposure is only about 20%, so from
  October 2024 to October 2025 it made 27.0% while simply holding BTC made 79.5% (though
  with a 31% drawdown against the bot's 14%).
- The 12 coins are highly correlated, so several positions can behave like one.
- Backtests use Binance candles. Roostoo's prices are streamed from Binance and were
  within a fraction of a percent of them when checked, but fills on Roostoo are not
  guaranteed to match.

## How it works

```
bot/
  config.py        every tunable number, with JSON overrides
  roostoo.py       API client: signing, clock offset, rate limit, retries, request log
  market_data.py   Binance hourly candles (live signals, warm-up, backtest cache)
  indicators.py    EMA, ATR, RSI, rolling volatility, updated one bar at a time
  strategy.py      target weights from signals and current holdings
  planner.py       trades from targets: rebalance threshold and activity rule
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
costs fall between the two.

Oct 2025 – Oct 2026, $100,000 starting cash:

| | Taker | Maker | BTC buy and hold | 12-coin buy and hold |
|---|---|---|---|---|
| Total return | 19.8% | 31.1% | -26.8% | 127.4% |
| Max drawdown | 15.5% | 13.9% | 53.7% | 57.5% |
| Sharpe | 1.29 | 1.82 | -0.46 | 1.36 |
| Sortino | 2.24 | 3.30 | -0.66 | 2.15 |
| Calmar | 1.28 | 2.24 | -0.50 | 2.22 |
| Composite score | 1.67 | 2.54 | -0.55 | 1.93 |
| Trades per day | 4.3 | 4.3 | | |
| Average exposure | 20% | 21% | | |

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
