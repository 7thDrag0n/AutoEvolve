# 🤖 AutoEvolve

**AutoEvolve** is a self-improving trading bot system built on top of [FreqTrade](https://www.freqtrade.io/). It uses a large language model (LLM) to continuously analyse trading performance, rewrite its own strategy code, deploy it, and roll back if results worsen — all without human intervention.

> ⚠️ **This is experimental software. Trade only what you can afford to lose. Always test in dry-run mode first.**

---

<!-- SCREENSHOT: Dashboard overview -->
<!-- *[Insert dashboard screenshot here]* -->

---

## How It Works

AutoEvolve runs alongside FreqTrade and watches its performance in a continuous loop:

```
Watch → Trigger → Analyse (LLM) → Rewrite code → Deploy → Monitor → Keep or Rollback
                      ↑_____________________|
```

1. **Watch** — Every 60 seconds, reads the FreqTrade SQLite database and computes metrics (win rate, profit factor, Sharpe, drawdown, consecutive losses, profit drawdown from peak)
2. **Trigger** — When a trigger condition fires (too many losses, profit eroding, idle strategy, etc.), an evolution cycle begins
3. **Analyse** — The LLM receives the current strategy code, all performance metrics, and recent evolution history
4. **Rewrite** — The LLM produces an improved strategy in two phases: ANALYSE (what's wrong) then CODE (fix it)
5. **Deploy** — The new strategy is validated, sanitised, and deployed to FreqTrade via hot-reload (no restart needed)
6. **Monitor** — The next N trades are observed. If performance drops below thresholds, the previous generation is automatically restored
7. **Fix errors** — If FreqTrade reports a strategy runtime error, AutoEvolve calls the LLM to fix the code and redeploys the same generation

---

## Features

### 🔁 Continuous Self-Improvement
- LLM rewrites the strategy using real trade history and performance context
- Rotates through 10 configurable trading concepts (smart money, multi-timeframe, ATR stops, etc.)
- Three exploration modes: `conservative`, `adaptive`, `aggressive`

### 🎯 Smart Triggers
| Trigger | Description |
|---|---|
| **Consecutive losses** | Fire after N losing trades in a row |
| **Profit drawdown** | Fire when cumulative PnL drops X% from its peak |
| **Idle** | Fire when no new trade has opened in X minutes (even with open positions) |
| **Cooldown** | Minimum gap between evolutions to prevent thrashing |
| **Trade gate** | Don't trigger until enough trades exist for meaningful stats |

### 🛡️ Safety & Rollback
- Automatically monitors N trades after each deploy
- Rolls back to previous generation if win rate or drawdown crosses thresholds
- Monitoring timeout: exits monitoring if no trades happen within the window
- Maximum fix attempts before giving up and requiring manual intervention

### 🩺 Strategy Error Auto-Fix
- Scans the FreqTrade log in real time for runtime errors (even when the API is healthy)
- Detects errors occurring at candle time (e.g. `TypeError in populate_indicators`)
- Calls LLM to fix the code and redeploys **the same generation** (no generation number burned)
- Log scanning uses a byte offset from FT start — old errors from previous sessions never trigger false fixes
- Gives up after configurable max attempts

### 📊 Live Dashboard
- Real-time metrics: win rate, profit factor, Sharpe, max drawdown, PnL
- Trigger status panel: shows all conditions with live countdowns and colour-coded readiness
- Profit drawdown gauge: peak vs current PnL with threshold progress
- Trades table: sorted by close time, with pair, direction, leverage, profit, duration, entry tag, exit reason
- Evolution history: all generations with metrics and changelogs
- Full strategy code viewer
- FT startup log viewer
- One-click controls: Force Evolution, Restart FT, Hot-Reload Config, Rollback to any generation

<!-- SCREENSHOT: Trigger status panel -->
<!-- *[Insert trigger panel screenshot here]* -->

### 🔌 LLM Provider Flexibility
Works with any OpenAI-compatible endpoint — just define a named block in config:

| Provider | Notes |
|---|---|
| **Anthropic** | Claude models (API key required) |
| **OpenAI** | GPT-4o and others |
| **OpenRouter** | Access many models via one API |
| **Chutes.ai** | Free tier with DeepSeek-R1 |
| **Ollama** | Local models (no API cost) |
| **Custom** | Any OpenAI-compatible endpoint |

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
  provider: "chutes"            # or anthropic, openai, openrouter, ollama
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

<!-- SCREENSHOT: Full dashboard with metric cards -->

### Status Pills (top bar)
- `AE · RUNNING` — AutoEvolve loop is active
- `FT · ONLINE` — FreqTrade API is reachable
- `API · OK` — Authentication successful

### Desired State (sidebar)
Two persistent toggles that **survive restarts**:
- **EVOLVE** — pause/resume the evolution loop
- **FREQTRADE** — start/stop FreqTrade (auto-started on AutoEvolve launch if set to running)

### Metric Cards
Real-time stats computed from the FreqTrade database — updated every 60 seconds regardless of system state.

### Trigger Status Panel

<!-- SCREENSHOT: Trigger status panel detail -->

| Row | Meaning |
|---|---|
| **Status** | `WATCHING` / `EVOLVING…` / `FIXING ERROR` / `MONITORING NEW GEN` |
| **Cooldown** | Time remaining before next evolution is allowed |
| **Trade gate** | Trades needed before triggers activate |
| **Loss trigger** | Consecutive losses counted from most recent trade backwards |
| **Idle trigger** | Time since last trade was *entered* (not just whether trades are open) |
| **Profit drawdown** | Current % drop from cumulative PnL peak |
| **Monitoring** | Trades observed after last deploy (rollback window) |

### Rollback
Select any previous generation from the dropdown and click **Rollback to Selected** — the strategy is restored and FreqTrade hot-reloaded instantly.

---

## Configuration Reference

### Evolution
```yaml
evolution:
  exploration_mode: "adaptive"     # conservative | adaptive | aggressive
  concepts_to_try:
    - "smart_money_concepts"
    - "multi_timeframe_filter"
    - "dynamic_stoploss_atr"
    - "volume_analysis"
    # ... 10 concepts rotated across generations
```

### Triggers
```yaml
trigger:
  consecutive_losses:        3     # evolve after N consecutive losses
  min_trades_before_trigger: 10    # minimum trades before any trigger activates
  cooldown_minutes:          60    # minimum gap between evolutions
  idle_trigger_minutes:      20    # evolve if no new trade opened in X min (0 = off)
  profit_drawdown_pct:       20    # evolve if PnL drops X% from peak (0 = off)
  profit_drawdown_min_trades: 5    # minimum trades before drawdown trigger activates
```

### Rollback
```yaml
rollback:
  evaluate_after_n_trades:    30   # trades to observe after deploy
  rollback_if_winrate_below:  0.35
  rollback_if_drawdown_above: 0.15
  monitoring_timeout_minutes: 60   # exit monitoring if no trades for this long
```

### LLM (custom provider example)
```yaml
llm:
  provider: "myprovider"
  myprovider:
    model:    "my-model-name"
    base_url: "https://api.myprovider.com/v1"
    api_key:  "sk-..."
    max_tokens: 8192
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
│   ├── improver.py          # LLM calls, prompt construction, code sanitisation
│   ├── deployer.py          # Strategy file deployment, FreqTrade process management
│   ├── harvester.py         # SQLite reader, metrics computation
│   ├── journal.py           # Evolution history, generation tracking
│   ├── server.py            # FastAPI dashboard backend + WebSocket
│   ├── ft_monitor.py        # FreqTrade health monitoring background thread
│   ├── strategy_template.py # Baseline strategy for generation 1
│   ├── utils.py             # Config, state file, logging helpers
│   └── templates/
│       └── index.html       # Dashboard UI (single-file, no build step)
├── checkpoints/             # Saved strategy generations (auto-managed)
├── logs/                    # AutoEvolve + FT logs
└── state.json               # Live system state (read by dashboard)
```

---

## Tips & Recommendations

**Start conservative.** Use `dry-run` mode for at least a few days before going live. AutoEvolve will burn through generations quickly in the first hours while it learns your market conditions.

**Set the trade gate high enough.** With `min_trades_before_trigger: 10`, you need at least 10 closed trades before any evolution fires. Too low and the LLM gets misleading stats. 20–30 is a good target for 5m timeframes.

**Use cooldown generously.** `cooldown_minutes: 60` prevents rapid-fire evolution if multiple triggers hit at once. On slow markets, 120–240 minutes may be better.

**Monitor the profit drawdown trigger carefully.** A 20% drawdown from peak sounds large but can be reached quickly on a bad run. Start with 30–40% until you understand your strategy's natural variance.

**Rollback is safe and instant.** Don't hesitate to manually roll back to a generation that was working. The system will evolve forward again from wherever you land.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| FT not starting on AutoEvolve launch | Check `freqtrade.config_file` path, ensure FT toggle is `running` |
| `API · FAIL` pill | FT is not running or `api.password` doesn't match `freqtrade_config.json` |
| Strategy error auto-fix loop | Check `ft_error_max_fix_attempts` — after max attempts, manual fix required |
| Trades tab empty | Verify `freqtrade.db_path` points to the correct SQLite file |
| Evolution not triggering | Check trigger conditions in status panel — trade gate is the most common blocker |
| Old error re-detected after manual fix | Restart FT via the dashboard button (resets log scan offset) |

---

## License

MIT — use freely, modify as needed, trade at your own risk.

---

*Built with FreqTrade · FastAPI · Vanilla JS · SQLite · ❤️*
