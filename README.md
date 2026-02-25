# FreqTrade Manager

A web-based management system for FreqTrade that automates the daily optimization cycle: **download data → backtest → hyperopt → extract best params → restart trade**.

## Features

- **Web Dashboard** — real-time process monitoring via WebSocket
- **Multi-Strategy** — configure and run multiple strategies independently
- **Automated Workflow** — scheduled daily optimization cycle
- **Hyperopt Monitor** — real-time epoch analysis by reading `.fthypt` result files directly (most reliable method)
- **Epoch Criteria** — configurable filter chain (e.g., drawdown ≤ 2%, then maximize profit)
- **Auto-Extract & Restart** — extracts winning epoch params and restarts trade automatically
- **Error Recovery** — if workflow fails, trade is restarted with previous config
- **Windows Native** — uses proper Windows process management (`taskkill /T` for process trees)

## Requirements

- Windows 10/11
- Python 3.12+ (same venv as FreqTrade)
- FreqTrade installed via git clone + pip in a venv

## Installation

```bash
# Clone or copy the freqtrade_manager folder anywhere on your system
cd freqtrade_manager

# Create a separate venv for the manager (or use your system Python)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` before first run. All settings are documented inline.

### Key Settings

| Setting | Description | Default |
|---|---|---|
| `freqtrade.directory` | Path to FreqTrade git clone | `D:\python\freqtrade` |
| `freqtrade.venv_path` | Venv folder inside FT dir | `.venv312` |
| `strategies[].name` | Strategy class name | `Predict_LSTM_Futures` |
| `strategies[].config` | Config path (relative to FT dir) | `user_data/strategies/.../config.json` |
| `schedule.cron` | When to run workflow | `0 2 * * *` (daily 2 AM) |
| `web.port` | Dashboard port | `8080` |

### Epoch Criteria

Criteria are evaluated **in order** as a filter chain:

```yaml
epoch_criteria:
  - field: "max_drawdown"      # Filter: keep only epochs with drawdown ≤ 2%
    operator: "<="
    value: 2.0
  - field: "profit_total_pct"  # Sort remaining by profit (descending)
    operator: ">="
    value: 0
    sort: "desc"
```

Available fields: `trades`, `wins`, `draws`, `losses`, `avg_profit`, `profit_total_pct`, `profit_total_abs`, `max_drawdown`, `max_drawdown_abs`, `objective`.

Operators: `<`, `<=`, `>`, `>=`, `==`.

### Timerange

Timeranges for backtest and hyperopt are defined as "days ago":

```yaml
hyperopt:
  timerange_start_days_ago: 30   # Start = today - 30 days
  timerange_end_days_ago: 0      # End = open (today). Set >0 for fixed end date
```

This generates `--timerange 20260110-` (with open end) or `--timerange 20260110-20260209`.

## Usage

```bash
# Start the manager
python main.py

# With custom config path
python main.py --config D:\myconfigs\ftmanager.yaml

# Override port
python main.py --port 9090
```

Open `http://127.0.0.1:8080` in your browser.

## Dashboard

### Process Panel
Start/stop individual FreqTrade processes: trade, download-data, backtest, hyperopt.

### Workflow Panel
- **Run Full Workflow** — triggers the complete cycle manually
- Shows progress steps: Stop Trade → Download → Backtest → Hyperopt → Extract → Restart
- Displays next scheduled run time

### Hyperopt Monitor
- Real-time epoch results as they complete
- Highlighted best epoch based on your criteria
- One-click extraction for any epoch
- Star (★) marks the freqtrade-internal best

### Output Viewer
Live process stdout/stderr. Switch between trade, hyperopt, backtest, download outputs.

## Automated Workflow Sequence

When triggered (manually or by schedule):

1. **Stop active trade** (graceful shutdown with 60s timeout)
2. **Download data** — updates pair data for all configured timeframes
3. **Run backtest** — triggers LSTM model retraining via the strategy's backtest path
4. **Delete old hyperopt results** — ensures clean `.fthypt` file for monitoring
5. **Run hyperopt** — with real-time epoch monitoring via `.fthypt` file
6. **Evaluate epochs** — apply criteria chain to find the best epoch
7. **Extract params** — `freqtrade hyperopt-show -n <epoch>` auto-exports to strategy dir
8. **Restart trade** — with the new optimized parameters

**On failure at any step**: the workflow logs the error and restarts trade with previous parameters if it was running before.

## How Hyperopt Monitoring Works

The system reads **FreqTrade's `.fthypt` result file** directly:

```
user_data/hyperopt_results/strategy_<StrategyName>.fthypt
```

Each line in this file is a JSON object containing full epoch metrics (trades, profit, drawdown, etc.). This is far more reliable than parsing console output.

The monitor polls the file every N seconds (configurable, default 3s), reads new lines, and evaluates against your criteria. Console output parsing is used as a fallback.

## Hyperopt-Show Extraction

When extracting epoch parameters, the system runs:

```
freqtrade hyperopt-show --config <hyperopt_config> -n <epoch_number>
```

If `hyperopt_config` is set in the strategy config, that config is used for extraction (useful if you have a separate config for hyperopt-show that points to the correct strategy directory). Otherwise, the main `config` is used.

FreqTrade's `hyperopt-show` command auto-exports the parameters to the strategy's directory.

## Multiple Strategies

Add more entries under `strategies:` in config.yaml. Each strategy has independent:
- Config files
- Workflow settings (download/backtest/hyperopt parameters)
- Epoch criteria
- Auto-restart behavior

The dashboard has tabs to switch between strategies.

## Logs

- **Console + file**: `freqtrade_manager.log` (rotating, 10MB × 5 files)
- **Dashboard**: real-time system log viewer
- **Process output**: captured per-process, viewable in dashboard

## Troubleshooting

| Issue | Solution |
|---|---|
| "FreqTrade executable not found" | Check `freqtrade.directory` and `venv_path` — the exe should be at `<dir>/<venv>/Scripts/freqtrade.exe` |
| Hyperopt monitor not finding `.fthypt` | Check `user_data/hyperopt_results/` exists. The file is created when hyperopt starts writing results. |
| Process won't stop | Windows process trees can be stubborn. The manager uses `taskkill /T` then `/F /T` as fallback. |
| Schedule not running | Check `schedule.enabled: true` and the cron expression. Verify with the "Next run" display in the dashboard. |
| Epochs not matching criteria | Check `epoch_criteria` in config. Remember `max_drawdown` from `.fthypt` is a fraction (0.02 = 2%), but the system converts it to percentage for you. |

## Architecture

```
main.py                  — Entry point, wires everything together
manager/
  config.py              — YAML config loader + dataclasses
  state.py               — Thread-safe shared state + WebSocket broadcast
  process_manager.py     — Start/stop/monitor subprocesses (Windows-native)
  hyperopt_monitor.py    — .fthypt file watcher + epoch criteria evaluation
  workflow.py            — Orchestrates the full optimization cycle
  scheduler.py           — APScheduler for timed workflow triggers
  web_app.py             — FastAPI routes + WebSocket endpoint
templates/
  index.html             — Dashboard UI (vanilla JS + WebSocket)
```
