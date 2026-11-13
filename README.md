# Amadeus: a BTC market assistant

Amadeus tells you **how much BTC is likely to move in the next 4 hours**, where to put stops, and how big a position that implies. She also explains **why it moved**, reads the news, and warns you before US data releases.

Why she leads with volatility and not with long/short: on your data, her 4-hour direction calls were about a coin flip (AUC 0.52), and they lost money after fees. The *size* of the next move, on the other hand, is genuinely predictable. Her volatility model explains about 50% of the variation in the next-4h range, vs 21% for "just use recent volatility", and that holds in every year tested. So that's what she's built around. She still shows the direction model's lean, labelled honestly, and paper-trades it in case a future retrain finds an edge.

What she looks at for direction (and partly for volatility):
- **BTC itself:** momentum over 1h to 3 days, position in its 24h range, distance from its 7-day average, volatility regime, volume surge, taker buy/sell imbalance
- **Crypto breadth:** ETH and SOL relative to BTC
- **Positioning:** perp premium over spot, perp taker flow, funding rate and how it's changing
- **Traditional markets:** S&P and Nasdaq futures, the dollar index, gold, VIX, the 10-year yield, and the BTC-stocks correlation
- **Calendar:** time of day, weekends
- **News:** headlines from CoinDesk, Cointelegraph, The Block, Decrypt, BBC, Al Jazeera, CNBC, the Fed and Google News searches for war, sanctions, embargoes, oil, Fed and tariffs, plus the calendar of US high-impact data releases (CPI, jobs, FOMC). See "News" below.

## Setup (Windows, once)

Open PowerShell in this folder (Shift + right-click the folder → "Open in Terminal"):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 1. Train and read the report

```powershell
python train.py
```

This downloads about 2 years of hourly data. The first run takes a few minutes; later runs are quicker because the data is cached. It then tests the model **walk-forward**: each month is predicted by a model trained only on the months before it. It prints the hit rate, the after-fee result for strong signals, and a verdict. The report is saved to `training_report.txt`.

### 2. Try a call in the terminal

```powershell
python check.py
```

### 3. Create the Telegram bot

1. In Telegram, message **@BotFather** → `/newbot` → choose a name and a username.
2. Copy the token it gives you.
3. Copy `.env.example` to `.env` and paste the token after `TELEGRAM_TOKEN=`.
4. Run `python bot.py`, open your bot in Telegram and send `/start`. She replies with your chat id.
5. Paste that number after `ALLOWED_CHAT_ID=` in `.env`. Restart the bot with Ctrl+C, then `python bot.py`.
   From then on she only answers you.

After that, double-click `run_bot.bat` to start her.

## Using her

| You send | She does |
|---|---|
| "long or short?", "how volatile is it?" or `/signal` / `/vol` | volatility forecast, stops for either side, position size, the direction lean (honestly labelled), news |
| `/stats` | shows the paper-trading record so far (including her silent hourly trades) and how her volatility forecasts compared with the real moves |
| `/open` | lists paper trades still running |
| `/wallet` (or "how much money do you have left?") | her paper wallet: started at $1,000, all-in at 40x, what's left. `/wallet reset` starts over |
| `/status` (or "are you alive?") | shows whether she's running and her background jobs are working: uptime, last successful scan, trade checker and news run, when each runs next, any errors, model age, data freshness. No reply at all means the bot isn't running. She also messages you if the hourly scan fails 3 times in a row, and again when it recovers |
| "why did BTC dump today?" / "why the spike in the last 6h?" / `/why 24h` | finds the biggest move in that window and what lined up with it (see below) |
| `/news` (or "what's the news?") | shows the headlines that matter for BTC, rated, plus upcoming US data |
| `/alerts`, or "ping me if…" / "alert me when…" / "notify me" | explains what she pings you about by herself, what she doesn't (direction setups, while there's no edge) and why, and the current on/off settings |
| `/alerts on` / `/alerts off` | turns volatility and US-data pings on or off (volatility pings are at most one per 3h). The hourly paper trade and forecast log keep running either way |
| `/whales` (or "any whales?") | whale watch on Binance: what she's watching right now, last hour's liquidations, and the thresholds. `/whales on` / `/whales off` mutes or unmutes the pings. She pings on **$100M+ net market buying or selling in 5 minutes** (about 4 a day), **$5M+ of longs or shorts liquidated in 5 minutes**, and **order walls** standing 10+ minutes within 1% of price on **Binance, Coinbase, OKX, Kraken, Bitfinex and Bitstamp** ($5M+ in one $50 price band) and the **OKX perp** ($30M+, because futures order books are far deeper), at most 3 new walls an hour, with a follow-up saying whether the wall got hit or was pulled (a spoof). Change the sizes with `WHALE_BURST_USD`, `WHALE_LIQ_USD`, `WHALE_WALL_USD`, `WHALE_WALL_PERP_USD` in `.env`. None of these are trade signals: bursts and liquidations already happened (often they *are* the move), walls are often fake, and Binance never shows *who* placed an order |
| ↳ liquidation prices | `/hl` sends a second message: **where the top 20's BTC positions get liquidated**, nearest first, with each trader's rank and size, and how much sits within 10% of price. These are their real liquidation prices as Hyperliquid reports them, not an estimate. They move whenever a trader adds margin or resizes, and big accounts usually cut before getting there, so it shows where they'd be forced out, not where price is going. |
| ↳ order books (or "where are the liquidity clusters?") | `/whales` also shows the **order books right now**: buy orders vs sell orders resting within 1% of price, added up over the six spot exchanges (OKX futures shown separately); the **three biggest liquidity clusters on each side** ($100 price bands, with how many exchanges they sit on); and what became of the walls she reported in the last 7 days (hit vs pulled). A two-line version appears under every `/signal`. It's context only: it never changes her call, a heavier buy side hasn't been tested as a signal, and resting orders can be cancelled instantly. A reading is saved every minute (`whale_depth`) so this can be tested later. |
| `/heatmap` (or "show me the liquidation map") | a **picture of where BTC positions on Hyperliquid get liquidated**: price up the side, and for each $500 of price a bar for how many dollars of positions would be force-closed there (shorts above the price, longs below). She adds it up from the real liquidation prices of the **1,000 biggest Hyperliquid accounts**, reading 100 of them a minute, so the map is at most about 10 minutes old (after a restart, wait 2 minutes for the first picture and 10 for the full one). The caption gives the totals within 5% and 20% of price and the biggest clusters. Know its limits: it is **one exchange** (a small slice of all BTC futures; Binance, OKX and Bybit keep positions private, so nothing can be said about them without guessing), big accounts mostly use low leverage so much of the money sits further than 20% away (the picture says how much), and these prices move whenever someone adds margin. Whether price gets pulled towards these clusters is untested; she saves the map hourly so it can be tested later. Context only: no pings, never part of her call |
| `/hl` (or "what are the top traders doing?") | the **Hyperliquid top 20**: the 20 accounts with the best 30-day profit on Hyperliquid, what each holds in BTC with their entry price and unrealised profit, the overall split (e.g. "13 short · 3 long · 4 flat") with the shorts' and longs' average entry, and the three wallets that copy them. `/hl on` / `/hl off` mutes the pings. A "Stocks & commodities" line adds up their biggest stock and commodity futures positions. She pings when one of them **opens, flips or closes a $5M+ BTC position** ($10M+ on any other coin or stock/commodity future), or adds or cuts a big piece of one, with their entry, leverage, liquidation price and a link to the account. See below |
| `/breaking on` / `/breaking off` | turns pings for major breaking news on or off |
| `/model` | shows the backtest summary |
| `/retrain` | refreshes the model with the latest data (do it weekly) |

Every hour, about 90 seconds after the hour, she checks the market and pings you when:
- **volatility steps up** into ELEVATED or HIGH, and again when it calms back down
- **US data is less than 2 hours away** (CPI, jobs, FOMC): one heads-up per release
- **a strong direction setup appears**, but *only* if the latest `train.py` found a real edge after fees. Right now it hasn't, so these stay silent.

### Volatility forecast
For the next 4 hours she forecasts BTC's **high-to-low range**. For example: "expect ~1.1% (80% chance 0.6–2.0%)". You also get:
- a **regime** label (CALM / NORMAL / ELEVATED / HIGH), measured against the past year
- the chance of a 2%+ and a 3%+ swing
- what's driving it: time of day, recent activity, volume, weekends, the VIX, funding extremes and so on

It's checked walk-forward, like the direction model, and the training report shows predicted vs actual ranges. In testing, "18% chance of a 2%+ swing" came true 19% of the time.

**Using it:** stops and targets scale with the forecast. In a quiet market they're tight; in a wild one they're wide. Position size follows from that: with `RISK_PCT=1` (the default), the stop is placed so that hitting it costs 1% of your account. So **when volatility is high, the same risk means a smaller position**. This is how professionals size trades, and it's the most useful thing the bot can do for you.

### News
Every 10 minutes Amadeus reads the feeds and rates each new headline: impact 0–3, and bearish or bullish for BTC.
- **Ratings:** without an API key she uses keyword rules. They're free, but crude: a routine update on an ongoing war can look like breaking news. If you add `ANTHROPIC_API_KEY` to `.env`, Claude rates the headlines instead and understands context ("Fed rules out a rate hike", "nothing new in this story"). The key is billed per use from prepaid Console credits, separately from any Claude subscription (it doesn't touch your Claude app limits). With Haiku 5.5 rating only new headlines, expect a few cents a day; set a monthly spend limit in the Console to be safe. After adding the key, run `pip install -r requirements.txt` once (it adds the `anthropic` package) and restart. `/status` shows whether the Claude scorer is working or has fallen back to keywords (for example when credits run out).
- **Breaking pings:** she pings you on major (impact 3) new stories, such as war breaking out, an oil embargo, a surprise Fed move or an exchange collapse. With keyword rules, at least 2 different outlets have to report the story first.
- **Effect on calls:** news **can only make a call more cautious**. A strong call becomes "lean", with no ping, when the news is clearly against it or when US CPI, jobs or FOMC data lands inside the 4-hour window. When major news broke in the last 2 hours, she also warns you that moves may be jumpy.
- **Why news never upgrades a call:** there's no free archive of past headlines to backtest on, so news isn't part of the trained model. By the time a headline reaches RSS, the price has often already moved. News is a risk filter, not a crystal ball.
- **Checking whether the filter helps:** every paper trade records the news backdrop at the time, and `/stats` shows results for trades where the news agreed vs. was against.

### "Why did it move?"
Ask in plain words: "why did btc crash?", "what happened this week?", "why the pump in the last 6h?". Amadeus works out the time window (24h by default, up to a week) and whether you mean the drop or the rise. Then she:
1. **Finds the move.** She locates the biggest drop or rise in that window, from peak to trough, with its exact start and end times and the volume compared with normal.
2. **Checks what else moved during exactly those minutes:**
   - ETH and SOL: if they moved too, it was the whole crypto market; if not, it was something BTC-specific
   - S&P and Nasdaq futures, the dollar, gold, oil and yields: did all risky assets move, or just crypto?
   - Open interest and aggressive buy/sell volume: falling open interest during a crash means leveraged longs were liquidated
   - Headlines from 3h before to 3h after the move, plus scheduled US data releases
3. **Gives her read:** the most likely cause, what made the move bigger, and whether it was crypto-only.
   With an `ANTHROPIC_API_KEY`, Claude writes the explanation from that evidence. Without one, rules do. Either way, if nothing lines up in time she says the trigger is unclear rather than inventing one.

Keep in mind that this shows **what happened at the same time**, which is strong evidence but not proof. Headlines published *after* a move are often reporters explaining it after the fact, and she marks those as such. Open interest history only goes back 30 days.

You can also run it from the terminal: `python explain.py "why did btc dump in the last 24h"`.

Times in messages are shown in `DISPLAY_TZ` (Tokyo by default; change it in `.env`).

### How the exit levels work
Each level is set from the forecast size of a 4-hour move (σ, from the volatility model: forecast range ÷ 1.6):
- **Take-profit zone:** 0.75σ to 1.25σ in your favour
- **Stop-loss:** 1σ against you
- **Time stop:** close after 4 hours if neither level is hit

In the paper log, a trade counts as a win at the near end of the TP zone. If the target and the stop are both touched in the same 5 minutes, it counts as a stop. Both rules are deliberately pessimistic.

### Her paper wallet (leverage)
Amadeus has a paper wallet that starts with **$1,000** (`WALLET_START` in `.env`). She goes **all-in at 40x** (`FIXED_LEVERAGE`) on the BTC perpetual future, one trade at a time: the whole balance is the margin (isolated), so the position is 40× the balance. A lean that comes up while she's already in a trade still goes in the unleveraged log, but the wallet sits it out. Nothing is actually traded. `/wallet` (or "how much money do you have left?") shows the balance, and `/stats` starts with it. Every trade the wallet was in gets a close message showing the $ result and the new balance. What 40x means:
- **Wins and losses ×40.** A 0.5% move against her costs 20% of the wallet.
- **Fees ×40.** 0.10% round trip on 40× the money is **4% of the wallet on every trade, win or lose**. With no edge, this alone drains the wallet: about 25 break-even trades cut it to roughly a third.
- **Funding ×40.** Perp holders pay or receive funding every 8 hours (00:00, 08:00, 16:00 UTC). The log uses Binance's actual rates: a long pays a positive rate and a short receives it.
- **Liquidation.** At 40x, a **2.1%** move against her (`1/40 − 0.4%` maintenance margin) and the exchange closes the position: the whole wallet is gone. Her stops are usually closer than that, but in a busy market (forecast range above ~3.4%) the stop is *further* than the liquidation price, so she gets liquidated before her stop can save her.
- **Broke.** Once balance × 40 is under $100 (Binance's minimum order), she can't trade. `/wallet reset` starts a new $1,000 wallet. Old trades stay in the log.

Stops are assumed to fill exactly at the stop price. In a real crash they can slip, and 40x multiplies that slippage too.

### Hyperliquid top 20 and the three copy wallets
Hyperliquid is a crypto exchange that runs on its own public blockchain, so unlike Binance, every account's positions are public. Once a day Amadeus downloads its leaderboard and picks the **20 accounts with the best 30-day profit**, skipping ones that didn't earn it with a directional bet: accounts that are losing all-time (one lucky month on a losing record), accounts with more "profit" than trading volume (a deposit or transfer), accounts under $1M, and market makers (over 100× their account traded in 30 days, like Auros: their position comes from quoting both sides, not a view). Someone who starts losing drops down the 30-day ranking and falls off the list at the next refresh, and she tells you who went out and who came in. Every minute she reads their positions.
- **Pings:** when one of them opens, flips or closes a $5M+ BTC position, or $10M+ on any other coin or stock/commodity future (`HL_PING_USD`, `HL_PING_ALT_USD`). Stock and commodity futures (gold, the S&P 500, Google...) are on a separate Hyperliquid exchange called **xyz**, also watched (`HL_DEXES`; on 2026-10-29 the top 20 held $403M there and about $1M on the other 9 such exchanges combined). Those pings say "GOLD (xyz)" and carry a warning: these futures trade 24/7 but the real market doesn't, so off-hours prices are thin, and a big short may be a hedge on shares held elsewhere. They're context only, with no paper trading. Also when they add or cut a big piece: at least a quarter more (or half less) than at the last ping, so a $200M holder needs a $50M add to count. Opens show their average entry, the current price and their liquidation price. Closes and cuts show their **exit price**: the average of their actual fills since her last ping about that position (a big position usually closes over many fills). They also show how far price moved in their favour and the profit Hyperliquid booked, before fees. If the fills can't be fetched, she shows the entry and the last unrealised profit she saw instead. Positions they already had when she first sees them aren't pinged. In a replay of their recent trades (2026-10-29), they held their BTC shorts almost unchanged and did most of their trading in altcoins and stock perps. That replay had zero BTC pings, even at $1M, and about 2 a day on other coins at $10M. Expect BTC pings to be rare: when one comes, it's a real change of mind.
- **In the long-or-short reply:** a "Hyperliquid top 20" section with their BTC split and whether they agree with her own lean. It never changes her call: nobody has tested whether last month's winners keep winning.
- **Three paper wallets test whether copying them works**, each starting with $1,000 and racing her own wallet. Each rule was fixed before any results came in, so none of them can be tuned until it looks good:
  - **Mirror** (🪞): holds what they hold *right now*, in proportion. For each of the 20 it takes BTC position ÷ their Hyperliquid futures account (flat = 0), averages that, and holds the same exposure. On 2026-10-29 that came out at **1.29x short**. It rebalances when their average moves 25%+ away from what it holds, so fees don't eat it (0.05% a side on what's traded), and pays or receives perp funding every 8 hours. It holds as long as they do, with no stops, so it's the closest thing to real copy trading. She pings when it switches between long, short and flat.
  - **Copy**: every hour, if 60%+ of the top 20 holding BTC are on the same side (at least 3 of them), it goes all-in on that side for 4h. One vote per trader, because by dollars two or three huge accounts would decide everything.
  - **Follow**: when one of them opens or adds a big BTC position (the same pings as above), it goes all-in their way for 4h, entering at the price when she sees it. Expect few of these.
  - Copy and follow use the **same stops, targets, 4h time stop, 40x leverage, fees and funding** as her own hourly trade, so the only difference is who picked the direction. They don't hold for weeks like the traders: at 40x, any 2.1% move against a weeks-long copy would wipe it out. Holding for weeks is the mirror's job, at their own leverage.
- `/wallet` shows all four wallets and `/stats` adds the per-trade record. `/wallet reset` resets all four, so the race restarts level. Judge the copy and follow wallets after **50+ trades**, and the mirror after a month or more. Compare the mirror with what BTC itself did over the same time, since being short in a falling month is the easy part.
- **History for a proper test later:** every day she saves the top 1,000 of the leaderboard (`hl_snapshots`), and every hour the tracked 20's BTC positions and account sizes (`hl_positions`). After a month or two, these answer "do last month's winners keep winning next month?" using only what was known at the time. They also allow ranking traders on their **BTC** results instead of total profit.
- Why it may not work: a top-20 month is partly luck, high leverage and survivorship (47,000+ accounts, someone always has a great month); some are market makers or hedgers whose BTC position isn't a bet on direction; and you only see a trade after it happened.

## Read this before trusting her

On 2025–2026 data, her direction model scored an **AUC of 0.519** (0.5 is a coin flip). Even her strongest 10% of calls lost money on average before fees. Adding funding, flow and stock-market data didn't help. Don't trade her direction lean. The volatility side is the part that works.

- When the backtest finds no edge after fees, the direction is shown as "no proven edge" and never pinged.
- Judge her by `/stats` after **50+ finished paper trades**, not by a few lucky calls.
- Maker (limit) orders cost about 0.04% round trip instead of 0.10%, which changes the maths a lot. Set `FEE_ROUND_TRIP=0.0004` in `.env` to see the effect.
- Leverage multiplies the losses as much as the wins. This is a learning tool, not financial advice.

## Keeping her running
She only works while the PC is awake and online.

- **Crashes:** `run_bot.bat` restarts her 10 seconds after she stops. To stop her for good, press Ctrl+C and answer **Y** to "Terminate batch job (Y/N)?". Each restart sends the "is online" message, so you'll see it happen.
- **Reboots:** add a Task Scheduler entry that runs `run_bot.bat` "At log on".
- **Sleep:** set Settings → System → Power → "When plugged in, put my device to sleep after" to **Never**, and set Windows Update active hours.
- **Knowing when she's down:** she can't message you when she's dead, so use a dead-man's switch. Make a free check at [healthchecks.io](https://healthchecks.io) (period 1 minute, grace 5 minutes; add its Telegram or email alert), then put its ping URL in `.env` as `HEARTBEAT_URL=https://hc-ping.com/...`. She opens it every minute; when the pings stop, it alerts you. It also catches a dead PC, power cut or internet drop. `/status` shows the last heartbeat.
- **What a restart loses:** open trades, wallets and stats are in `paper_log.db` and survive. Hourly scans inside the gap are skipped, and Hyperliquid moves made while she was down aren't pinged (she silently records the positions she finds on startup).
- **Only one copy at a time.** Two copies with the same bot token fight over Telegram and paper-trade everything twice.

Moving her to a server later is just a matter of copying this folder (with `paper_log.db` and `.env`) over and running `python bot.py` there under something that restarts it.

## Files
`config.py` (settings) · `data.py` (Binance + Yahoo downloads) · `features.py` (signals) · `model.py` (training + backtest) · `vol_model.py` (volatility forecast) · `signal_engine.py` (current call + message) · `paper_log.py` (SQLite paper trades) · `news.py` (headlines + calendar) · `explain.py` (why it moved) · `bot.py` (Telegram) · `train.py` · `check.py`
