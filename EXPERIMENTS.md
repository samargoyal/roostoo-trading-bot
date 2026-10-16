# Roostoo bot: experiment log

Everything we tested for the HK vs AU vs IN Quant Trading Hackathon, in one place: what the market
turned out to be, what worked, what did not, how each live version was chosen, and how to check the
account's records against the code. The full notes, round by round, are in
[docs/research.md](docs/research.md) (150 numbered hypotheses, about 2,700 lines); the scripts are in
[research/](research/). Last updated 16 October 2026.

**Live now:** cash (`config/comp_flat.json`), after the account was closed at **+2.6%** on 13 October.
**Best strategy found:** 70% long the pair with the best 25-day return, 30% short the worst, re-picked
daily (`config/comp_ls7030.json`): +5,850% over October 2023 to October 2026, **p = 0.008**.

---

## 1. The task

A $100,000 mock account on Roostoo, whose prices are streamed from Binance, traded for 14 days (4-17
October 2026) by a fully autonomous bot on one AWS instance. Taker fee 0.1%, maker fee 0.05%, short
positions 0.1% to open and to close; at most 30 API calls a minute; no leverage, high-frequency
trading, market making or arbitrage; trades on at least 8 of the 14 days. Judged on compliance (the
trade logs must match the declared strategy, every change committed), return, then a composite score
of 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar, then a review of code and strategy.

---

## 2. What the market turned out to be

| Finding | Evidence |
|---|---|
| **Fees decide everything under a day.** No signal held for less than about a day pays a 15-25 bp round trip | 5-minute bars (H60, H86, H125), 1-minute bars of 64 pairs (H126, H133), 1-second bars (H127, H128): the typical one-minute move is 2-6 bp and every design lost 14-34 bp a trade |
| **A month of momentum is the robust edge for a concentrated holding** | 684 all-in designs (H144, H145): all 32 that passed the significance tests ranked on 21-35 day returns; 1-10 day returns, reversal, RSI and BTC-relative strength did not |
| That edge lives in the top tail, not the average | The 28-day return's rank correlation with the next three days across all pairs is -0.005 (t -0.7): it does not order coins on average, yet holding the single strongest pair is significant (p = 0.001 against random picks) |
| **Trend-following shorts make money in crashes** | The long-short book with shorts on 24/72-hour EMAs gained in all 8 BTC falls of 25% or more since 2018 (average +35.9%, H139) |
| Flipping the whole account between long and short loses | 300 long-or-short designs lost 36-100% over three years (H146): the side signal turns at the wrong times and every flip pays fees |
| A high win rate and large wins rarely come together | 3,576 designs (H130-H136): frequent signals reach a 60% win rate only with wins smaller than losses; both together only on BTC crash days (about six a year) |
| Machine learning and foundation models add little | Ridge, GBM, MLP, GRU: rank IC 0.06-0.08, mostly rediscovering low volatility (H21); Google's TimesFM 3.0, zero-shot on 53,613 daily forecasts: IC +0.008, t 1.4 (H143) |
| Roostoo's prices follow Binance's 1-3 seconds late | A 4-minute probe of both feeds side by side. Trading that delay is arbitrage, which the rules forbid; the scalper refuses to enter unless Roostoo already shows Binance's price |
| Periods differ sharply | Strategies tuned on bull years lose in bear years; the 24/72-hour book that was live from 9 to 12 October had p = 0.25 over 2023-26 and lost 14% in the latest year (H137) |
| Every backtest here is optimistic | The coin list is today's (survivorship); about 3,600 designs were tried; the deflated Sharpe ratios below allow for the second, nothing allows fully for the first |

---

## 3. Method

- **One code path.** The backtester runs the bot's own `Strategy` and planner on hourly Binance
  candles, with the coin list chosen point in time (the most traded pairs each month), taker fees and
  each pair's spread. The live bot and the backtest differ only in where prices and fills come from.
- **Out-of-sample discipline.** Designs were fixed before running and judged on six yearly folds
  (October 2020 to October 2026), with October 2018 to October 2020 held back as an untouched
  holdout; a design had to win on most folds and on the holdout, and its neighbours had to hold too.
- **The competition's own yardstick.** The composite score over every 14-day window, and the odds
  of a given gain or loss over the days left (H129).
- **Randomness checks** (H137, H144-H150): a 7-day block bootstrap of the mean daily return, the
  deflated Sharpe ratio for the number of designs tried, and a permutation test against 1,000 runs
  picking random coins under the same rules.
- **Fast search, then confirmation.** A vectorised daily simulator (thousands of designs on 15 CPU
  cores) screens; finalists are re-run in the production backtester before anything goes live.
- **Engineering.** 273 unit tests, a dry run through the live code path before every deployment,
  every live version a single committed config file, and a compliance check (section 8).

---

## 4. Live history

Each row is one commit of `config/comp.json` followed by a restart on the server. The account's
figures are as the account showed them at the time.

| From | Commit | Version | Account |
|---|---|---|---|
| 4 Oct | `189ae70` | 70% momentum rotation + 30% long-only trend book | started at $100,000 |
| 4 Oct | `ccace84` | 70% rotation + 30% long-short trend book (R54b) | +1.5% on 5 Oct |
| 6 Oct | `f549cb8` | rotation ranked on 1-3 week momentum, 3 coins, 75/25 (K2) | |
| 7 Oct | `d03a152` → `ec0b431` | book weighted by efficiency ratio; splits of 55/45 and book alone | -4.2% on 7 Oct |
| 7-8 Oct | `ebe18e5` → `08a629f` | the book alone on fast 24/72-hour EMAs, with bands, stops and blocks | about -5% |
| 9 Oct | `c7d408b`, `10e0236` | flattened, then the same book from cash | |
| 12 Oct | `0875ba5` | B1: 60% rotation + 40% book (chosen for its odds of a 6.6% week) | -6% |
| 12 Oct | `e186193`, `b4e4360` | + checks every minute; the crash-proof book below BTC's 200-day average | -9% |
| 13 Oct | `366b6c3` | 70/30 long/short rotation | -12.5% when deployed |
| 13 Oct | `e24a937` | cash | **+2.6%** when closed |

The rotation's longs fell with the altcoins on 7-8 and 11-13 October while BTC hardly moved, so its
BTC filter never turned. The 70/30 design bought S at 0.056; S rose 27% within hours and the account
was closed near that peak, at +2.6%. Holding on would have given it back: S fell 22% two hours later.

**The lesson we take from it.** About 25 versions in ten days was too many: each switch reacted to a
few days of results, paid to rebalance, and abandoned an edge that only shows over weeks. The
research below was sound; the deployment discipline was not.

---

## 5. What worked

| Idea | Evidence |
|---|---|
| Momentum rotation into the strongest coins, filtered by BTC's 168/672-hour trend | the backbone of every strong design; holdout-confirmed |
| Ranking on 1-, 2- and 3-week returns together (K2) | six folds +105,989% against +49,404%; holdout better in both years |
| Long-short trend book, coins weighted by inverse volatility × 30-day efficiency ratio | book alone +6,666% against +2,794%, drawdown 31% against 43% (rounds 94-97) |
| The book takes the rotation's share while BTC's filter is off (B1) | best of eight designs over every BTC crash since 2018 |
| Shorts on 24/72-hour EMAs | profit in all 8 crashes since 2018 (H139) |
| Switching to the crash-proof book below BTC's 200-day average | development Sharpe 2.60 against 2.35; p = 0.032 and 0.004 out of sample; positive every year (H141) |
| **70% long the best 25-day pair, 30% short the worst, daily** | **+5,850% over 2023-26, positive every year, p = 0.008; median 14-day composite 1.95** (H149, H150) |
| Filtering trades with a classifier (meta-labeling) | picks beat random candidates in both out-of-sample periods (p ≤ 0.003), though its profit was significant in only one (H135) |

---

## 6. What did not work

**Faster trading.** Scalping at any speed (5 minutes to 1 second, market or resting limit orders,
the hottest coins only, more or fewer trades a day); re-picking more often than daily (hourly
re-picks turned +1,217% into -98%); stops on every wick within the hour (H121); minute-by-minute
stop checks (a wash, H138).

**Signals.** Short-term reversal and RSI; opening-range breakouts (252 designs); Williams %R;
order-flow and ICT patterns; funding-rate rankings; Wikipedia attention; Fear & Greed; Donchian
channels; capitulation buys in general (passed only on crash days, did not survive the correction
for 1,008 designs tried).

**Risk add-ons.** Stops at the entry price (cut B1's three years from +2,086% to +172%);
take-profits and profit locks; rotation stops of 4-15%; volatility targeting; drawdown brakes;
hourly regime switches.

**Models.** ML rankings (ridge, GBM, MLP, GRU); TimesFM 3.0; convex and mean-variance optimisers;
VECM pairs trading (lost even without costs).

**Shorting the losers outright.** As a sleeve, all-in, or as the bear-market half of a flip: losers
bounce violently, and 1x shorts on Roostoo pay 0.1% each way.

---

## 7. How the best strategy is built

`config/comp_ls7030.json`, in the bot's ordinary code path:

1. **Universe:** every crypto pair Roostoo lists (65), with Binance hourly history.
2. **Rank** each pair on its 600-hour (25-day) return at 00:00 UTC.
3. **Hold** 70% of the account in the top pair and short the bottom pair with 30% (a pair whose
   perpetual funding shows crowded shorts is skipped); no BTC filter, no book.
4. **Re-pick** daily; between re-picks only the activity rule's small orders trade.
5. **Costs:** taker fees and half the spread in the backtest; limit orders at the touch first live.

---

## 8. Checking the account against the code

`python -m bot.compliance --account comp` (with `--exchange` on the server, which has the keys)
reads the bot's own records and writes `runs/comp/compliance.md`:

1. every start of the bot ran a committed config, and the settings it logged are those in the commit;
2. every order came from the cycle decision before it (same pair, same target weight, the right
   direction), apart from the activity rule's small orders;
3. every order's reason is one the strategy or planner gives;
4. the days with at least one filled order (the rule asks for 8 of 14);
5. every order in Roostoo's own history was recorded by the bot (none placed by hand);
6. the account's return, drawdown and competition score.

---

## 9. Key files

| File | Purpose |
|---|---|
| `bot/strategy.py`, `bot/research_rules.py` | The strategy and every research option (off unless configured) |
| `bot/live.py`, `bot/execution.py`, `bot/planner.py` | The live loop, order placement, trade planning |
| `bot/backtest.py`, `research/folds.py` | The production backtester and the fold harness |
| `bot/compliance.py` | The compliance check (section 8) |
| `config/comp.json` | What the competition account runs; `config/comp_*.json` every earlier and alternative version |
| `research/h1*.py` | The scripts behind each hypothesis |
| `docs/research.md` | The full notes |
