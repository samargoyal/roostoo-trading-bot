# Roostoo trading bot

An autonomous trend-following bot for the [Roostoo](https://github.com/roostoo/Roostoo-API-Documents)
mock exchange, built by Team124 for the HK vs AU vs IN Quant Trading Hackathon. It trades
through the Roostoo REST API with no manual intervention: every order comes from the
strategy, and every order, decision and hourly equity point is recorded.

- [Strategy](#strategy)
- [Which assets it trades](#which-assets-it-trades)
- [Strategy research](#strategy-research)
- [How it works](#how-it-works)
- [Strategy versions](#strategy-versions)
- [Backtest results](#backtest-results)
- [Setup and running](#setup-and-running)
- [Configuration](#configuration)
- [Records for judging](#records-for-judging)
- [Tests](#tests)

## Strategy

Two books share the account, trading the 45 most traded crypto pairs on Roostoo plus PAXG
(gold-backed):

- **Momentum rotation, 70% of equity.** Holds the 2 coins that rank best on their 1-, 2- and
  3-week returns together, weighted 2/2/1 (each needs a positive 2-week return), re-chosen
  daily at 00:00 UTC, while BTC's 168-hour EMA is above its 672-hour EMA, and leaves at once
  when it is not. This is the return engine. Until 6 October 2026 it ranked on the 2-week
  return alone (R54b, kept in [`config/comp_r54b.json`](config/comp_r54b.json)); the ranking
  sums that one's score and the multi-horizon bot's (round 82).
- **Long-short trend book, 30% of equity** (the competition account since 5 October 2026,
  through [`config/comp.json`](config/comp.json)). Every coin long while its 240-hour EMA is
  above its 960-hour EMA and short while below, weighted by inverse volatility, with a trailing
  stop on each short and no short where shorts are crowded (negative perpetual funding). See
  [Rounds 50 to 64](docs/research.md#rounds-50-to-64-long-short).
- **Defensive trend book**, the code's default book, which held the 30% until 5 October 2026:
  low-volatility coins in uptrends, with a market regime filter, trailing stops and a drawdown
  brake.

The split was 40/60 until 4 October 2026. An optimisation of the rotation's share in steps of
5% (see [The rotation's share](docs/research.md#the-rotations-share-optimised)) gave 50% under a 50% drawdown
limit; the user then chose 70%, more return for more risk, as the competition ranks on
return first.

The two books' daily returns are barely correlated (0.18), so together they keep most of the
rotation's upside with a much smaller drawdown. The competition ranks on return and then
scores `0.4 x Sortino + 0.3 x Sharpe + 0.3 x Calmar`.

Over the six yearly folds, the rotation with the long-short book made +49,404% against
+31,945% with the defensive book, with a worst yearly drawdown of 52% against 58%, and a
better competition score in 5 of 6 years and in both untouched holdout years. Ranking the
rotation on several horizons as it is live now made +105,989%, more than R54b in 5 of 6
years, with the mildest crash year of any version (-36% against -49%) and a better yearly and
14-day score in both holdout years; but its median 14-day window beat R54b's in only 3 of the
6 folds, and it lagged in 2024–25 (+94% against +233%).

The long-short book's rules:

| Rule | Detail |
|---|---|
| Direction | Long while the coin's 240-hour EMA is above its 960-hour EMA, short while below. PAXG is left out. |
| Sizing | Each coin's share of the book is its inverse volatility (168 hours) over the sum for all coins, so calmer coins weigh more, and a coin not held leaves its share in cash. Shorts are 1x on Roostoo, so longs plus shorts never exceed the book's 30%. |
| Short stop | A short is covered once the close rises 10 ATR above its lowest close since entry; no new short in that coin for 24 hours. |
| Crowding filter | No short in a coin whose Binance perpetual funding rate averaged below zero over the 72 hours to 00:00 UTC: shorts paying longs means crowded shorts, the set-up for a squeeze. Fetched once a day from Binance's USD-M futures API; a coin without a perpetual or without data is not filtered. |
| Paying for entries | When entries need more cash than is free, the holdings furthest above their targets are trimmed to pay for them. |
| Activity rule | As below; with no long to adjust, the short furthest from its target is. |

The defensive book's rules (the default book; it held the 30% until 5 October 2026):

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
> POL, CAKE, SEI, ZEN, TUT, VIRTUAL, PENDLE, CRV, FLOKI, EIGEN, WIF, PLUME, and PAXG

(FORM was replaced by FLOKI on 4 October 2026, when FORM's spread rose above 0.1% and the
rule picked FLOKI instead.)

The list was 20 coins until a six-year test (see [Out-of-sample validation](docs/research.md#out-of-sample-validation-across-six-years))
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

Every rule above was chosen by tests fixed in advance, not by tuning until a backtest looked
good. The full log of 73 research rounds is in [docs/research.md](docs/research.md), with the
scripts in [`research/`](research/). In short:

- **Method.** Six one-year folds (October 2020 to October 2026) run through the bot's own
  backtester, and an untouched holdout (October 2018 to October 2020) used only to confirm a
  design that has passed. From round 31 a change is adopted only if it beats the bot on the
  competition's composite score in at least 5 of the 6 folds, with a worst drawdown at most 2
  points deeper, if its neighbouring settings also hold, and then only if it beats the bot in
  both holdout years.
- **What worked.** Low volatility predicts which coin does better next (the defensive book's
  ranking); two-week momentum finds the coins that lead a rally (the rotation); BTC's trend
  says when to be in the market (the rotation's filter); and per-coin trend following, long
  and short, with the shorts defended against squeezes, beat the defensive book (rounds 50–64).
- **What did not.** Statistical arbitrage (VECM), machine learning and neural networks, order
  flow and ICT concepts, TradingView indicators, sentiment and retail attention, tokenized
  stocks, 81 swing and medium-frequency strategies, and most short selling: shorts opened late,
  into crowded squeezes, failed. Nor did any rule for securing profits, fixed or driven by
  indicators (rounds 68–70): the returns come from a few large winners, and every exit cut
  them by more than it saved.

## How it works

```
bot/
  config.py        every tunable number, with JSON overrides
  roostoo.py       API client: signing, clock offset, rate limit, retries, request log
  market_data.py   Binance hourly candles (live signals, warm-up, backtest cache)
  universe.py      which pairs to trade: the most traded crypto pairs on Roostoo
  indicators.py    EMA, ATR, RSI, rolling volatility, updated one bar at a time
  strategy.py      the strategy: target weights from signals and current holdings
  research_rules.py  research options, all off by default, kept out of strategy.py
  reasons.py       the reason recorded with every target and order
  planner.py       trades from targets: rebalance threshold, activity rule, sells
                   anything outside the universe
  execution.py     orders on Roostoo: rounding, limit-then-market, fills
  live.py          the hourly loop, state recovery, dry-run mode
  journal.py       logs, append-only trade and equity records, saved state
  backtest.py      replays the same strategy and planner on Binance candles
  metrics.py       return, drawdown, Sharpe, Sortino, Calmar, composite score
config/comp.json   the competition account's settings, loaded automatically
scripts/run_bot.sh restarts the bot whenever it exits (inside tmux, or as a service:
                   scripts/install_service.sh)
research/          the research scripts (docs/research.md)
tests/             unit tests, plus end-to-end runs against a simulated exchange
```

Every hour, a minute after the candle closes, the live bot:

1. Fetches the last 2500 closed hourly candles per pair from Binance and rebuilds the
   indicators. Roostoo has no history endpoint, but its prices are streamed from Binance.
   If Binance is unreachable, it uses its cached candles plus hourly bars built from
   Roostoo ticker samples taken every 5 minutes.
2. Once a day, with the long-short book, fetches each coin's perpetual funding rates from
   Binance's futures API for the crowding filter (if that API cannot be reached, the filter is
   off for the day).
3. Reads the Roostoo wallet, open shorts and ticker, and values the portfolio at the last price.
4. Asks the strategy for target weights and the planner for trades.
5. Sells and covers first, then buys and shorts with the cash actually available. Each
   order rests as a limit order at the best bid or ask (0.05% maker fee); anything unfilled
   after 5 minutes is cancelled and sent at market (0.1%). Stop-loss exits and shorts go
   straight to market.
6. Reconciles the strategy state with the new wallet, records everything, and saves
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

## Strategy versions

What ran on the competition account, and from which commit. The bot writes its commit and
config file to `bot.log` at every start, and `runs/comp/restarts.log` records each restart.

| Live from (UTC) | Commit | Strategy |
|---|---|---|
| 4 October 2026, 12:59 | `189ae70` | 70% momentum rotation and 30% defensive trend book, long only |
| The restart after 18:43, 4 October 2026 | `ccace84` | 70% momentum rotation and 30% long-short trend book (round 54's R54b) |
| Not deployed | `5bc0678` | `config/comp.json` switched to the multi-horizon ranking, but the server was not updated and the account kept trading R54b; the next commit put R54b back in `config/comp.json` and the ranking in `config/comp_multi.json` |
| The restart after this push, 6 October 2026 | the commit that adds this row | R54b with the rotation ranked on 1-, 2- and 3-week returns weighted 2/2/1 (round 82's K2); R54b's settings are kept in `config/comp_r54b.json` |

Later commits that do not change the strategy (refactoring, documentation, research) are not
listed; the backtests check that they trade exactly as before.

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
| **Bot as live (rotation ranked on 1, 2 and 3 weeks, 2/2/1; long-short trend book)** | **+2,727%** (34%) | **-36%** (45%) | **+90%** (52%) | **+792%** (42%) | **+94%** (47%) | **+78%** (34%) | **+105,989%** |
| R54b, the rotation ranked on 2 weeks (until 6 October 2026, `config/comp_r54b.json`) | +1,614% (38%) | -49% (52%) | +59% (46%) | +626% (37%) | +233% (42%) | +48% (34%) | +49,404% |
| The rotation ranked on 1, 2 and 3 weeks, 2/1/1 (`config/comp_multi.json`, not live) | +3,239% (33%) | -44% (51%) | +96% (50%) | +946% (43%) | +82% (48%) | +79% (33%) | +123,592% |
| Bot with the defensive book (until 5 October 2026) | +1,248% (36%) | -56% (58%) | +56% (46%) | +580% (38%) | +265% (43%) | +38% (37%) | +31,945% |
| Bot with 60% rotation | +973% (34%) | -51% (54%) | +52% (42%) | +431% (33%) | +228% (39%) | +36% (32%) | +18,756% |
| Bot with 50% rotation (the optimised share under a 50% drawdown limit) | +795% (30%) | -45% (48%) | +44% (38%) | +323% (29%) | +209% (35%) | +38% (28%) | +12,696% |
| Bot with 40% rotation (until 4 October 2026) | +679% (27%) | -39% (42%) | +39% (33%) | +229% (24%) | +181% (31%) | +36% (25%) | +8,228% |
| Bot with a 4-coin inverse-ATR book (before round 6) | +537% (25%) | -40% (42%) | +42% (30%) | +236% (23%) | +150% (31%) | +30% (22%) | +5,871% |
| Bot with 20 coins (the earlier list) | +786% (21%) | -28% (35%) | -1% (30%) | +39% (41%) | +89% (33%) | +43% (24%) | +2,258% |
| Defensive book alone | +117% (16%) | -10% (16%) | -11% (14%) | +2% (13%) | +30% (14%) | +16% (10%) | +167% |
| Hold BTC | +306% (55%) | -56% (74%) | +39% (27%) | +135% (32%) | +80% (31%) | -27% (54%) | +674% |

Maximum drawdown in brackets. With the long-short book the bot beat holding BTC in all six
years, with a better competition score than with the defensive book in five of them. With the
defensive book it also beat holding BTC in all six years (in 2022–23 only just:
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

A tmux session does not survive a reboot of the server. To run the bot as a systemd service
instead, which also starts it after a reboot, stop the tmux copy first (`Ctrl-c` in its
window), then:

```
sudo scripts/install_service.sh comp
```

The same `git pull && pkill ...` deploys new code; `systemctl status roostoo-bot-comp` shows
the service.

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

Without `--config`, the live bot uses `config/<account>.json` if it exists, so each account's
settings are committed and a restart picks them up. `config/comp.json` runs the competition
account: the long-short book, and the rotation ranked on 1-, 2- and 3-week returns weighted
2/2/1. Two alternatives are kept: `config/comp_r54b.json`, the settings before 6 October 2026
(the rotation ranked on 2-week returns), and `config/comp_multi.json` (weighted 2/1/1). To
switch, copy one over `config/comp.json`, commit, push and restart, so the change is in the
history.

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
| `logs/bot.log` | What the bot did and why, starting with the commit and config it runs. Rotated. |
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
left open, a Binance outage, and a pair halted at start-up and resumed later. They run on
every push (`.github/workflows/tests.yml`), on Python 3.9 and 3.12.
