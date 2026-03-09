# 🤖 AutoEvolve

<img width="1481" height="672" alt="image" src="https://github.com/user-attachments/assets/421a83f9-ae57-4ed1-b2a1-74e102fa3ba9" />

**AutoEvolve** is a self-improving trading bot system built on top of [FreqTrade](https://www.freqtrade.io/). It uses a large language model (LLM) to continuously analyse trading performance, rewrite its own strategy code, deploy it, and roll back if results worsen — all without human intervention.

> ⚠️ **This is experimental software. Trade only what you can afford to lose. Always test in dry-run mode first.**

---

## How It Works

AutoEvolve runs alongside FreqTrade and watches its performance in a continuous loop:

```
Watch → Trigger → Analyse (LLM) → Rewrite code → Deploy → Monitor → Keep or Rollback
                      ↑_____________________|
```

1. **Watch** — Every 60 seconds, reads the FreqTrade SQLite database and computes metrics (win rate, profit factor, Sharpe, drawdown, consecutive losses, profit drawdown from peak)
2. **Trigger** — When a trigger condition fires (too many losses, profit eroding, idle strategy, etc.), an evolution cycle begins
3. **Analyse** — The LLM receives the current strategy code, all performance metrics, recent trade history, and the full evolution log
4. **Rewrite** — The LLM produces an improved strategy in two phases: **ANALYSE** (diagnose what's wrong, plain English) then **CODE** (implement the fix)
5. **Deploy** — The new strategy is validated, sanitised, and deployed to FreqTrade via hot-reload (no restart needed)
6. **Monitor** — The next N trades are observed. If performance drops below thresholds, the previous generation is automatically restored
7. **Fix errors** — If FreqTrade reports a strategy runtime error, AutoEvolve calls the LLM to fix the code and redeploys the same generation (no generation number burned)

---

## Features

### 🔁 Two-Phase LLM Evolution
Each evolution cycle calls the LLM **twice**:
- **Step 1 — ANALYSE:** Given current metrics, trade history, and evolution history, the LLM diagnoses weaknesses and selects a concept to try
- **Step 2 — CODE:** Given Step 1's diagnosis and the current strategy code, the LLM rewrites only what needs changing

This separation keeps analysis clear and code generation focused.

### 🎯 Smart Triggers
| Trigger | Description |
|---|---|
| **Consecutive losses** | Fire after N losing trades in a row (counted from most recent trade backwards) |
| **Profit drawdown** | Fire when open unrealized losses erase X% of closed PnL — tracks full equity (closed PnL + live open PnL) |
| **Idle** | Fire when no new trade has opened in X minutes. Suppressed automatically when all trade slots are full |
| **Cooldown** | Minimum gap between evolutions to prevent thrashing |
| **Trade gate** | Don't trigger until enough closed trades exist for meaningful stats |

#### Profit Drawdown Trigger — Full Equity Formula
```
total_pnl = closed_pnl + open_pnl (live unrealized, from /api/v1/status)
threshold = closed_pnl × (1 - drawdown_pct / 100)
TRIGGER when: total_pnl < threshold
```
Example (`drawdown_pct=50`): closed=+10, open=-6 → total=+4 < floor=+5 → **TRIGGER**
Example (`drawdown_pct=50`): closed=+10, open=-4 → total=+6 > floor=+5 → no trigger

The trigger panel shows a progress display during the waiting phase:
```
waiting 5/10 closed trades · closed +16.0000 · open -6.0000
```
Negative PnL is highlighted in amber so you can see the pressure building before the trigger is even active.

#### Idle Trigger — Slot Awareness
The idle trigger is automatically suppressed when all trade slots are full (`max_open_trades` reached). When the strategy simply cannot open new trades because slots are taken, idleness is expected and should not trigger an evolution.

### 🛡️ Safety & Rollback
- Automatically monitors N trades after each deploy
- Rolls back to previous generation if win rate or drawdown crosses thresholds
- Monitoring timeout: exits monitoring if no trades happen within the window
- Maximum fix attempts before giving up and requiring manual intervention

### 🩺 Strategy Error Auto-Fix
- Scans the FreqTrade log in real time for runtime errors (even when the API is healthy)
- Detects errors occurring at candle time (e.g. `TypeError in populate_indicators`)
- Calls LLM to fix the code and redeploys **the same generation** (generation number not incremented)
- Log scanning uses a byte offset from FT start — old errors from previous sessions never trigger false fixes
- Gives up after configurable max attempts

### 📊 Live Dashboard
- Real-time metrics: win rate, profit factor, Sharpe, max drawdown, PnL
- **Trigger status panel** — all conditions with live countdowns and colour-coded readiness
- **Profit drawdown gauge** — waiting progress + full USDT breakdown (closed / open / total / floor)
- **Trades table** with two sub-tabs:
  - **Open** — live positions with unrealized profit fetched from `/api/v1/status` (updates every poll)
  - **Closed** — historical trades sorted newest-first with pair, direction, leverage, profit %, profit USDT, duration, entry tag, exit reason
- **Evolution history** — all generations with metrics and changelogs
- **Full strategy code viewer**
- **FT startup log viewer**
- One-click controls: Force Evolution, Restart FT, Hot-Reload Config, Rollback to any generation

### 🔌 LLM Provider Flexibility
Works with any OpenAI-compatible endpoint — just define a named block in `config.yaml`:

| Provider | Notes |
|---|---|
| **Anthropic** | Claude models (direct API, not OpenAI-compat) |
| **OpenAI** | GPT-4o and others |
| **OpenRouter** | Access many models via one API, with price/throughput/latency sorting and cost caps |
| **Chutes.ai** | Free tier with DeepSeek-R1 |
| **Ollama** | Local models (no API cost) |
| **Custom** | Any OpenAI-compatible endpoint — just name it and define `model`, `base_url`, `api_key` |

### 📝 LLM Conversation Logging
All LLM requests and responses are written to `logs/llm_conversations.log` (rotating, max 10 MB × 5 files). Useful for debugging the quality of prompts and understanding why the LLM made a particular change.

### 🧬 Trade History in LLM Context
The last `trade_history_count` trades (default: 20) are sent to the LLM with every evolution. Trades are all-inclusive — both closed and currently open — sorted oldest to newest by open date, so the LLM sees the most recent market behaviour including live positions. Each trade includes: pair, direction, leverage, profit %, profit USDT, duration, entry tag, exit reason, open date, close date.

---

## Installation

### Requirements
- Python 3.11+
- FreqTrade installed and configured
- An LLM provider API key (or local Ollama)

### Setup

```bash
# Clone / copy AutoEvolve into a folder next to or inside your FreqTrade directory
cd autoevolve

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configure

```bash
# Copy the example config and edit it
cp config.example.yaml config.yaml
```

Key settings to configure in `config.yaml`:

```yaml
freqtrade:
  config_file:    "./freqtrade_config.json"   # path to your FT config
  strategies_dir: "~/freqtrade/user_data/strategies"
  db_path:        "~/freqtrade/user_data/tradesv3.sqlite"
  api:
    host:     "127.0.0.1"
    port:     8080
    username: "freqtrade"
    password: "CHANGE_ME"       # must match FT api_server password

llm:
  provider: "chutes"            # or anthropic, openai, openrouter, ollama, or any custom name
  chutes:
    model:   "deepseek-ai/DeepSeek-R1"
    api_key: "your-key-here"
    base_url: "https://llm.chutes.ai/v1"
```

### Run

```bash
python main.py
```

The dashboard opens at **http://localhost:8501**

> FreqTrade does **not** need to be running before AutoEvolve starts — if the FreqTrade toggle is set to `running`, AutoEvolve will launch it automatically on startup and after every restart.

---

## Dashboard Guide

### Status Pills (top bar)
- `AE · RUNNING` — AutoEvolve loop is active
- `FT · ONLINE` — FreqTrade API is reachable
- `API · OK` — Authentication successful

### Desired State (sidebar)
Two persistent toggles that **survive restarts**:
- **EVOLVE** — pause/resume the evolution loop
- **FREQTRADE** — start/stop FreqTrade (auto-started on AutoEvolve launch if set to running)

### Metric Cards
Real-time stats computed from the FreqTrade database — updated every 60 seconds regardless of system state. Includes: win rate, profit factor, Sharpe ratio, Sortino ratio, max drawdown, total PnL, expectancy, avg win/loss, long/short win rate split, per-pair breakdown, entry tag stats, exit reason breakdown, duration stats, funding fee total, stop-loss hit count.

### Trigger Status Panel

| Row | Meaning |
|---|---|
| **Status** | `WATCHING` / `EVOLVING…` / `FIXING ERROR` / `MONITORING NEW GEN` / `PAUSED` |
| **Cooldown** | Time remaining before next evolution is allowed |
| **Trade gate** | Closed trades needed before triggers activate |
| **Loss trigger** | Consecutive losses counted from most recent trade backwards |
| **Idle trigger** | Time since last trade was *opened*. Shows slot fill status when suppressed |
| **Profit drawdown** | Waiting progress (N/min closed trades) or active: DD% / threshold with USDT breakdown |
| **Monitoring** | Trades observed after last deploy (rollback window) |

### Trades Tab

**Open sub-tab** — live positions fetched from the FreqTrade REST API. Shows unrealized profit in real time.

**Closed sub-tab** — all closed trades from the database, newest first.

### Rollback
Select any previous generation from the dropdown and click **Rollback to Selected** — the strategy is restored and FreqTrade hot-reloaded instantly.

---

## Configuration Reference

### FreqTrade
```yaml
freqtrade:
  venv:           "~/freqtrade/.venv"
  config_file:    "./freqtrade_config.json"
  strategies_dir: "~/freqtrade/user_data/strategies"
  db_path:        "~/freqtrade/user_data/tradesv3.sqlite"
  strategy_name:  "AutoEvolveStrategy"
  initial_strategy_file: ""   # optional: seed gen 1 from your own .py file
  reset_db_after_evolve: true # wipe trade DB after each successful evolution
  extra_flags: "--dry-run"
  logfile: "./logs/freqtrade.log"
  ft_error_max_fix_attempts: 5
  api:
    host: "127.0.0.1"
    port: 8080
    username: "freqtrade"
    password: "CHANGE_ME"
```

### Evolution
```yaml
evolution:
  exploration_mode: "adaptive"     # conservative | adaptive | aggressive
  concepts_to_try:
    - "smart_money_concepts"        # Order blocks, FVG, liquidity sweeps, BOS/CHoCH
    - "support_resistance_pivots"
    - "multi_timeframe_filter"
    - "volume_analysis"
    - "dynamic_stoploss_atr"
    - "dynamic_leverage"
    - "entry_confluence"
    - "partial_exit_scaling"
    - "regime_filter"
    - "candlestick_patterns"
```

### Triggers
```yaml
trigger:
  consecutive_losses:          3    # evolve after N consecutive losses
  min_trades_before_trigger:  10    # minimum closed trades before any trigger activates
  cooldown_minutes:           60    # minimum gap between evolutions
  idle_trigger_minutes:        5    # evolve if no new trade opened in X min (0 = off)
                                    # suppressed automatically when all slots are full
  profit_drawdown_pct:        20    # evolve when open losses erase X% of closed PnL (0 = off)
                                    # formula: total_pnl = closed_pnl + open_pnl
                                    #          threshold = closed_pnl × (1 - pct/100)
  profit_drawdown_min_trades:  5    # minimum closed trades before drawdown trigger activates
```

### Rollback
```yaml
rollback:
  evaluate_after_n_trades:    30   # trades to observe after deploy
  rollback_if_winrate_below:  0.35
  rollback_if_drawdown_above: 0.15
  monitoring_timeout_minutes: 60   # exit monitoring if no trades for this long
```

### LLM
```yaml
llm:
  provider: "anthropic"         # any named block below — or your own custom name
  timeout:  600                 # seconds (DeepSeek-R1 can be slow)
  max_prompt_tokens: 50000      # trim prompt if over this size
  include_trade_history: true
  trade_history_count:   20     # last N trades (open + closed) sent to LLM
  extra_allowed_imports: []     # third-party libs beyond the built-in FreqTrade set
                                # e.g. ["ta", "sklearn.preprocessing"]

  anthropic:
    model:       "claude-opus-4-6"
    api_key_env: "ANTHROPIC_API_KEY"
    max_tokens:  16384

  openai:
    model:       "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
    max_tokens:  16384
    temperature: 0.6

  openrouter:
    model:       "anthropic/claude-opus-4-6"
    api_key_env: "OPENROUTER_API_KEY"
    base_url:    "https://openrouter.ai/api/v1"
    max_tokens:  16384
    temperature: 0.6
    allow_fallbacks:      true
    max_price_prompt:     0.10   # USD per 1M prompt tokens
    max_price_completion: 0.40   # USD per 1M completion tokens
    sort:                 "price" # price | throughput | latency

  chutes:
    model:       "deepseek-ai/DeepSeek-R1"
    api_key_env: "CHUTES_API_KEY"
    base_url:    "https://llm.chutes.ai/v1"
    max_tokens:  16384
    temperature: 0.6

  ollama:
    model:    "codellama:34b"
    base_url: "http://localhost:11434"
    max_tokens: 16384
    temperature: 0.6

  # Custom provider — any OpenAI-compatible endpoint:
  # myprovider:
  #   model:    "my-model-name"
  #   base_url: "https://api.myprovider.com/v1"
  #   api_key:  "sk-..."
  #   max_tokens: 8192
```

---

## Project Structure

```
autoevolve/
├── main.py                  # Entry point
├── config.yaml              # All configuration (hot-reloadable)
├── freqtrade_config.json    # FreqTrade config (managed here)
├── autoevolve/
│   ├── orchestrator.py      # Main loop, trigger logic, evolution coordination
│   ├── improver.py          # Two-phase LLM calls, prompt construction, code sanitisation
│   ├── deployer.py          # Strategy deployment, FT process management, REST API helpers
│   ├── harvester.py         # SQLite reader, full metrics computation, trade history
│   ├── journal.py           # Evolution history, generation tracking
│   ├── server.py            # FastAPI dashboard backend + WebSocket
│   ├── ft_monitor.py        # FreqTrade health monitoring background thread
│   ├── strategy_template.py # Baseline strategy for generation 1
│   ├── utils.py             # Config, state file, logging helpers
│   └── templates/
│       └── index.html       # Dashboard UI (single-file, no build step)
├── checkpoints/             # Saved strategy generations (auto-managed)
├── logs/                    # AutoEvolve + FT logs
│   └── llm_conversations.log  # Full LLM prompt/response log (rotating)
└── state.json               # Live system state (read by dashboard)
```

---

## Tips & Recommendations

**Start conservative.** Use `dry-run` mode for at least a few days before going live. AutoEvolve will burn through generations quickly in the first hours while it learns your market conditions.

**Set the trade gate high enough.** With `min_trades_before_trigger: 10`, you need at least 10 closed trades before any evolution fires. Too low and the LLM gets misleading stats. 20–30 is a good target for 5m timeframes.

**Use cooldown generously.** `cooldown_minutes: 60` prevents rapid-fire evolution if multiple triggers hit at once. On slow markets, 120–240 minutes may be better.

**Monitor the profit drawdown trigger carefully.** The trigger compares total equity (closed PnL + live open unrealized PnL) against a floor derived from closed PnL alone. A 20% threshold sounds large but can be reached quickly on a bad run. Start with 30–40% until you understand your strategy's natural variance. Watch the trigger panel's waiting display — it shows you the pressure building even before the minimum trade count is reached.

**Idle trigger and slot awareness.** If your strategy holds many concurrent positions, the idle trigger will correctly suppress itself when all slots are full. You only need to worry about idleness when slots are free but no new positions are being opened.

**LLM trade history.** The LLM sees the last `trade_history_count` trades including any currently open ones. This gives it context about what the strategy is doing *right now*, not just what happened historically.

**Rollback is safe and instant.** Don't hesitate to manually roll back to a generation that was working. The system will evolve forward again from wherever you land.

**Check `llm_conversations.log`** if evolutions produce poor code. It contains the full prompt and response for every LLM call, making it easy to understand what the model was told and what it decided.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| FT not starting on AutoEvolve launch | Check `freqtrade.config_file` path, ensure FT toggle is `running` |
| `API · FAIL` pill | FT is not running or `api.password` doesn't match `freqtrade_config.json` |
| Strategy error auto-fix loop | Check `ft_error_max_fix_attempts` — after max attempts, manual fix required |
| Trades tab empty | Verify `freqtrade.db_path` points to the correct SQLite file |
| Open trades show no profit | FreqTrade REST API must be reachable — check `api.host`, `api.port`, `api.password` |
| Evolution not triggering | Check trigger conditions in status panel — trade gate is the most common blocker |
| Idle trigger never fires | Check if slots are full — shown in the trigger panel as `slots X/Y` |
| Drawdown trigger shows 0.0% always | Ensure FreqTrade API is reachable (open PnL comes from `/api/v1/status`) |
| Old error re-detected after manual fix | Restart FT via the dashboard button (resets log scan offset) |
| LLM producing bad code | Check `logs/llm_conversations.log` for the full prompt; adjust `exploration_mode` or add `extra_allowed_imports` |

---

## License

MIT — use freely, modify as needed, trade at your own risk.

---

*Built with FreqTrade · FastAPI · Vanilla JS · SQLite · ❤️*
