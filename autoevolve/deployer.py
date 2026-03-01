"""
autoevolve/deployer.py
======================
CheckpointManager  — save/load strategy generations
FTManager          — start / stop / monitor FreqTrade process
FTHealthMonitor    — background thread: auth-ping every N seconds,
                     auto-repair if strategy error, log if external
Deployer           — deploy new generation + rollback logic
"""
from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .utils import (
    cfg, local_str, append_log, write_state, read_state,
    ft_executable, ft_python, ft_config_path, ft_log_path,
    LOG_MAX_BYTES, LOG_BACKUP_COUNT, BASE_DIR,
)

logger = logging.getLogger("autoevolve.deployer")


# ══════════════════════════════════════════════════════════════
# Checkpoint Manager
# ══════════════════════════════════════════════════════════════
class CheckpointManager:
    def __init__(self):
        self._refresh()

    def _refresh(self):
        raw = cfg("checkpoints", "directory", default="checkpoints")
        p   = Path(raw).expanduser()
        self._dir = p if p.is_absolute() else (BASE_DIR / p).resolve()
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, gen: int, code: str, metrics: dict,
             reason: str, changelog: str = "") -> Path:
        self._refresh()
        d = self._dir / f"gen_{gen:04d}"
        d.mkdir(exist_ok=True)
        (d / "strategy.py").write_text(code, encoding="utf-8")
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        meta = {
            "generation":      gen,
            "created_at":      local_str(),
            "trigger_reason":  reason,
            "changelog":       changelog,
            "metrics_summary": {
                k: metrics.get("metrics", {}).get(k)
                for k in ["win_rate", "profit_factor", "sharpe", "max_drawdown"]
            },
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info(f"Checkpoint saved: gen {gen}")
        self._prune()
        return d

    def load_code(self, gen: int) -> Optional[str]:
        self._refresh()
        f = self._dir / f"gen_{gen:04d}" / "strategy.py"
        return f.read_text(encoding="utf-8") if f.exists() else None

    def latest_gen(self) -> int:
        self._refresh()
        dirs = sorted(self._dir.glob("gen_*"))
        if not dirs:
            return 0
        try:
            return int(dirs[-1].name.replace("gen_", ""))
        except ValueError:
            return 0

    def all_meta(self) -> list[dict]:
        self._refresh()
        result = []
        for d in sorted(self._dir.glob("gen_*")):
            f = d / "meta.json"
            if f.exists():
                try:
                    result.append(json.loads(f.read_text()))
                except Exception:
                    pass
        return result

    def _prune(self):
        max_g = cfg("checkpoints", "max_generations", default=10)
        for d in sorted(self._dir.glob("gen_*"))[:-max_g]:
            shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# FT error classifier
# ══════════════════════════════════════════════════════════════
def _strategy_logger_error_re() -> re.Pattern:
    """
    Match any ERROR line where the logger IS the strategy class itself.
    e.g. "2026-03-01 14:38:40,813 - AutoEvolveStrategy - ERROR - Error in ..."
    The strategy name is read fresh each call so hot-reload of config works.
    """
    sname = cfg("freqtrade", "strategy_name", default="AutoEvolveStrategy")
    return re.compile(
        r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d+\s+-\s+"
        + re.escape(sname)
        + r"\s+-\s+ERROR\s+-",
        re.IGNORECASE,
    )


_STRATEGY_ERROR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    # Hard crashes
    r"AttributeError", r"NameError", r"SyntaxError", r"IndentationError",
    r"ImportError.*strategy", r"Could not load strategy", r"Error loading strategy",
    r"Strategy.*not found",
    r"Impossible to load Strategy",  # FT import failure (preceded by the actual error)
    r"Fatal exception",           # FT process crash with traceback
    r"AttributeError.*DataProvider",  # missing DP method called from strategy
    r"AttributeError.*informative",   # missing method in informative_pairs
    r"object has no attribute",   # broad AttributeError catch in strategy context
    # Runtime errors in strategy methods
    r"TypeError.*populate_", r"TypeError.*custom_stoploss",
    r"TypeError.*leverage", r"TypeError.*confirm_trade", r"TypeError.*custom_exit",
    r"KeyError.*dataframe", r"ValueError.*indicator",
    r"populate_indicators.*error", r"populate_entry_trend.*error",
    r"populate_exit_trend.*error", r"failed.*strategy", r"strategy.*failed",
    # FT per-pair analysis failure (strategy code error at candle time)
    r"Unable to analyze candle",
    r"Exception.*populate_indicators", r"Exception.*populate_entry",
    r"Exception.*populate_exit",
    # Generic unhandled exceptions in strategy context
    r"Error in strategy", r"Strategy.*exception", r"Unhandled exception.*strategy",
    # Specific common errors
    r"unsupported operand type.*populate",
    r"object is not subscriptable.*dataframe",
    r"has no attribute.*dataframe",
    r"ZeroDivisionError.*indicator",
]]

# Error lines that look like strategy errors but are actually infrastructure
_STRATEGY_ERROR_WHITELIST = [re.compile(p, re.IGNORECASE) for p in [
    # "Unable to analyze candle" for exchange/network issues (not code errors)
    r"Unable to analyze candle.*ConnectionError",
    r"Unable to analyze candle.*TimeoutError",
    r"Unable to analyze candle.*NetworkError",
]]

_EXTERNAL_ERROR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"ConnectionError", r"TimeoutError", r"ReadTimeout", r"ConnectTimeout",
    r"requests\.exceptions", r"urllib3", r"socket\.", r"NetworkError",
    r"RateLimit", r"DDosProtection", r"ExchangeNotAvailable",
    r"retrying.*connection", r"Could not connect", r"SSL.*error",
    r"EOF occurred", r"Temporary failure", r"No route to host",
]]


# Regex that matches a FT log timestamp at the START of a line
_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d+\s+-"
)


def _split_log_blocks(log_lines: list[str]) -> list[list[str]]:
    """
    Split raw log lines into logical blocks, where each block is everything
    between two consecutive top-level timestamp lines.
    Lines that do NOT start with a timestamp are continuation lines
    (tracebacks, code snippets, carets) and belong to the preceding block.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in log_lines:
        if _TS_RE.match(line):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


# Any Python exception class at start of a line (bare TypeError, ValueError, etc.)
_EXCEPTION_LINE_RE = re.compile(
    r"^\s*(TypeError|ValueError|AttributeError|KeyError|NameError|"
    r"IndexError|ImportError|RuntimeError|ZeroDivisionError|"
    r"SyntaxError|IndentationError|AssertionError|Exception|"
    r"NotImplementedError|OverflowError|RecursionError|"
    r"Could not load|Error loading|Strategy.*not found)"
    r"[:\s]",
    re.IGNORECASE,
)

# A block is a traceback block if it contains this marker
_TRACEBACK_MARKER = "Traceback (most recent call last)"

# A block implicates strategy code if any line references the strategy file
_STRATEGY_FILE_RE = re.compile(
    r"AutoEvolveStrategy|user_data.strategies|strategy.interface|"
    r"populate_indicators|populate_entry|populate_exit|"
    r"custom_stoploss|custom_exit|leverage_amount|confirm_trade",
    re.IGNORECASE,
)


def _block_is_strategy_error(block_lines: list[str], full_text: str) -> bool:
    """
    A block is a strategy error if ANY of:
    0. Any line is an ERROR logged by the strategy class itself
       (logger name == strategy_name) — catches "Error in populate_*" etc.
    1. Contains a Python Traceback + any exception line
       (the traceback itself is proof it originated in code)
    2. Matches one of the explicit _STRATEGY_ERROR_PATTERNS
    3. Contains "Unable to analyze candle" with an exception description
       (FT's per-pair wrapper around strategy crashes)
    AND it does NOT match external/network patterns.
    """
    # Rule 0: ERROR logged by the strategy itself (most reliable signal)
    # e.g. "2026-03-01 ... - AutoEvolveStrategy - ERROR - Error in populate_indicators: ..."
    _strat_err_re = _strategy_logger_error_re()
    if any(_strat_err_re.search(ln) for ln in block_lines):
        return True

    # Rule 1: Traceback + exception line anywhere in block
    has_traceback = _TRACEBACK_MARKER in full_text
    has_exception = any(_EXCEPTION_LINE_RE.search(ln) for ln in block_lines)
    if has_traceback and has_exception:
        return True

    # Rule 2: Explicit patterns
    if any(pat.search(full_text) for pat in _STRATEGY_ERROR_PATTERNS):
        return True

    # Rule 3: FT's "Unable to analyze candle" wrapper —
    # this line always means a strategy runtime exception
    if ("Unable to analyze candle" in full_text and
            any(_EXCEPTION_LINE_RE.search(ln) for ln in block_lines
                if "Unable to analyze candle" in ln)):
        return True

    return False


def classify_ft_error(log_lines: list[str]) -> tuple[bool, str]:
    """
    Returns (is_strategy_error, full_excerpt).
    Each 'block' is all lines between two consecutive FT timestamp lines.
    Captures the complete block so the LLM sees the full traceback context,
    not just the matched line.
    Deduplicates blocks that share the same root exception.
    """
    if not log_lines:
        return False, ""

    blocks = _split_log_blocks(log_lines)

    matched_blocks: list[str] = []
    seen_keys: set[str] = set()

    for block_idx, block in enumerate(blocks):
        full_text = "\n".join(block)

        # Hard skip: external / network errors
        if any(pat.search(full_text) for pat in _EXTERNAL_ERROR_PATTERNS):
            continue
        if any(pat.search(full_text) for pat in _STRATEGY_ERROR_WHITELIST):
            continue

        if not _block_is_strategy_error(block, full_text):
            continue

        # "Impossible to load Strategy" is just a summary line — the actual
        # cause (e.g. "name 'Decimal' is not defined") is in the preceding
        # WARNING/ERROR block. Prepend it to give the LLM the real error.
        if "Impossible to load Strategy" in full_text and block_idx > 0:
            prev = "\n".join(blocks[block_idx - 1])
            if any(kw in prev for kw in ("WARNING", "ERROR", "Could not import")):
                full_text = prev + "\n" + full_text

        # Deduplicate: key on the exception line (or first error-pattern match)
        key_line = next(
            (ln for ln in block if _EXCEPTION_LINE_RE.search(ln)),
            block[0]
        )
        key = key_line.strip()[:120]
        if key in seen_keys:
            continue
        seen_keys.add(key)

        matched_blocks.append(full_text)
        if len(matched_blocks) >= 5:
            break

    if not matched_blocks:
        return False, ""

    separator = "\n" + "─" * 60 + "\n"
    return True, separator.join(matched_blocks)


# ══════════════════════════════════════════════════════════════
# FT Manager
# ══════════════════════════════════════════════════════════════
class FTManager:
    _TAIL_LINES  = 500   # tracebacks can be 30-50 lines; 500 gives plenty of buffer
    _STARTUP_LOG = BASE_DIR / "logs" / "ft_startup.log"

    # ── Config patcher ────────────────────────────────────────
    def _patch_ft_config(self, config_path: str,
                         strategies_dir: str, sname: str) -> bool:
        """
        Patch strategy_path and strategy into the FT config file.
        Returns True on success, False on JSON error (abort start).
        """
        try:
            p = Path(config_path)
            if not p.exists():
                msg = f"FT config not found: {p}"
                append_log("ERROR", msg); logger.error(msg)
                return False

            raw     = p.read_text(encoding="utf-8")
            cleaned = re.sub(r"//[^\r\n]*", "", raw)        # strip // comments
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned) # trailing commas

            # Escape bare control characters inside JSON strings
            # (common with Windows paths containing \t, \n, etc.)
            def _fix_ctrl(m):
                s = m.group(0)
                return s.translate({
                    0x09: r"\t", 0x0a: r"\n", 0x0d: r"\r",
                    **{c: f"\\u{c:04x}" for c in range(0x00, 0x20)
                       if c not in (0x09, 0x0a, 0x0d)}
                })
            # Only fix ctrl chars that are INSIDE string literals
            cleaned_safe = re.sub(
                r'"(?:[^"\\\n]|\\.)*"',
                _fix_ctrl,
                cleaned,
                flags=re.DOTALL,
            )
            try:
                data = json.loads(cleaned_safe)
            except json.JSONDecodeError as je:
                ls  = cleaned.splitlines()
                ln  = max(0, je.lineno - 1)
                ctx = "\n".join(ls[max(0, ln - 2):ln + 3])
                msg = (f"FT config JSON error: {je.msg} "
                       f"at line {je.lineno} col {je.colno}\n"
                       f"--- context ---\n{ctx}\n---")
                append_log("ERROR", msg); logger.error(msg)
                return False

            if (data.get("strategy_path") != strategies_dir
                    or data.get("strategy") != sname):
                data["strategy_path"] = strategies_dir
                data["strategy"]      = sname
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                append_log("INFO",
                    f"Patched FT config: strategy={sname}  path={strategies_dir}")
            else:
                append_log("INFO", f"FT config OK (strategy={sname})")
            return True

        except Exception as e:
            msg = f"Could not patch FT config: {e}"
            append_log("ERROR", msg); logger.error(msg)
            return False

    # ── Start ─────────────────────────────────────────────────
    def start(self) -> bool:
        # Kill any existing FT instance first (single-instance guarantee)
        if self.is_running():
            append_log("WARNING", "FT already running — stopping before restart")
            self.stop()
            time.sleep(2)

        cfg_path = ft_config_path()
        sname    = cfg("freqtrade", "strategy_name", default="AutoEvolveStrategy")
        sdir_raw = cfg("freqtrade", "strategies_dir", default="")
        sdir     = Path(sdir_raw).expanduser().resolve() if sdir_raw else Path(".")
        extra    = cfg("freqtrade", "extra_flags", default="--dry-run")
        logfile  = ft_log_path()

        logfile.parent.mkdir(parents=True, exist_ok=True)

        ft_exe = ft_executable()
        py     = ft_python()

        if ft_exe is not None:
            cmd_list = [str(ft_exe), "trade"]
        else:
            cmd_list = [str(py), "-m", "freqtrade", "trade"]
            append_log("WARNING", f"freqtrade.exe not found — using: {py} -m freqtrade")

        # Pass strategy and strategy-path as CLI args — avoids parsing
        # the FT config file (which may use non-standard JSON like bare
        # backslashes in Windows paths that are valid for FT but not
        # for Python's strict json.loads).
        cmd_list += [
            "--config",        str(cfg_path),
            "--strategy",      sname,
            "--strategy-path", str(sdir),
            "--logfile",       str(logfile),
        ]
        for arg in extra.split():
            cmd_list.append(arg)

        append_log("INFO", f"Starting FreqTrade: {' '.join(str(x) for x in cmd_list)}")

        self._STARTUP_LOG.parent.mkdir(parents=True, exist_ok=True)

        # FT must run from its own root dir so relative paths in its config
        # (e.g. user_data/...) resolve correctly.
        # Set freqtrade.root_dir in config.yaml — defaults to parent of user_data.
        # FT must run from its own root dir so relative paths in config
        # (e.g. user_data/...) resolve correctly.
        # Set freqtrade.root_dir in config.yaml to override.
        ft_root_raw = cfg("freqtrade", "root_dir", default="")
        if ft_root_raw:
            ft_cwd = str(Path(ft_root_raw).expanduser().resolve())
        else:
            # Walk up from strategies_dir looking for a folder that
            # contains a user_data subfolder — that is the FT root.
            ft_cwd = str(sdir)  # safe fallback
            check = sdir
            for _ in range(8):
                if (check / "user_data").is_dir():
                    ft_cwd = str(check)
                    break
                check = check.parent
                if check == check.parent:  # filesystem root
                    break
        append_log("INFO", f"FT working directory: {ft_cwd}")

        try:
            with open(self._STARTUP_LOG, "w", encoding="utf-8") as ft_out:
                if platform.system() == "Windows":
                    proc = subprocess.Popen(
                        cmd_list,
                        stdout=ft_out,
                        stderr=ft_out,
                        close_fds=True,
                        cwd=ft_cwd,
                    )
                else:
                    proc = subprocess.Popen(
                        cmd_list,
                        stdout=ft_out,
                        stderr=ft_out,
                        start_new_session=True,
                        close_fds=True,
                        cwd=ft_cwd,
                    )

            # Record the log file size at FT start — the health monitor
            # will only scan lines written AFTER this offset, so old error
            # entries from previous sessions never trigger false auto-fixes.
            log_offset = 0
            try:
                lf = ft_log_path()
                if lf.exists():
                    log_offset = lf.stat().st_size
            except Exception:
                pass

            write_state({
                "ft_pid":            proc.pid,
                "ft_status":         "starting",
                "ft_running":        False,
                "ft_desired":        "running",
                "ft_started_at":     local_str(),
                "ft_log_scan_from":  log_offset,
                "ft_error":          None,
                "ft_error_count":    0,
            })
            append_log("INFO",
                f"FreqTrade launched — PID {proc.pid}  "
                f"startup output → {self._STARTUP_LOG}")

            # Wait 4 s then confirm the process is still alive
            time.sleep(4)
            if self._pid_alive(proc.pid):
                write_state({"ft_status": "running", "ft_running": True})
                append_log("INFO", f"FreqTrade running OK — PID {proc.pid}")
                return True
            else:
                try:
                    output     = self._STARTUP_LOG.read_text(
                                     encoding="utf-8", errors="replace")
                    last_lines = "\n".join(output.splitlines()[-30:])
                except Exception:
                    last_lines = "(could not read startup log)"
                msg = (f"FreqTrade exited immediately.\n"
                       f"--- ft_startup.log (last 30 lines) ---\n"
                       f"{last_lines}\n---")
                append_log("ERROR", msg); logger.error(msg)
                write_state({
                    "ft_status":  "crashed",
                    "ft_running": False,
                    "ft_pid":     None,
                    "ft_error":   last_lines[-400:],
                })
                return False

        except Exception as e:
            append_log("ERROR", f"Failed to start FreqTrade: {e}")
            logger.error(f"FT start failed: {e}")
            return False

    # ── Stop ──────────────────────────────────────────────────
    def stop(self) -> bool:
        try:
            if platform.system() == "Windows":
                pid = read_state().get("ft_pid")
                if pid:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                    )
                subprocess.run(
                    ["taskkill", "/F", "/IM", "freqtrade.exe"],
                    capture_output=True,
                )
                # Also cover pip-installed FT running as python.exe
                subprocess.run(
                    ["wmic", "process", "where",
                     "name='python.exe' and commandline like '%freqtrade%trade%'",
                     "call", "terminate"],
                    capture_output=True, timeout=10,
                )
            else:
                subprocess.run(
                    ["pkill", "-f", "freqtrade trade"],
                    capture_output=True,
                )
            write_state({"ft_pid": None, "ft_status": "stopped", "ft_running": False})
            append_log("INFO", "FreqTrade stopped")
            return True
        except Exception as e:
            append_log("ERROR", f"Stop failed: {e}")
            return False

    # ── Restart ───────────────────────────────────────────────
    def restart(self) -> bool:
        self.stop()
        time.sleep(3)
        return self.start()

    # ── PID alive check ───────────────────────────────────────
    def _pid_alive(self, pid: int) -> bool:
        try:
            if platform.system() == "Windows":
                r = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True, text=True, timeout=5,
                )
                return str(pid) in r.stdout
            else:
                import os as _os
                _os.kill(pid, 0)
                return True
        except Exception:
            return False

    # ── Is running ────────────────────────────────────────────
    def is_running(self) -> bool:
        """
        Check if OUR FreqTrade instance is running via PID, then process scan.
        REST API is NOT used here (remote host may always be reachable).
        """
        pid = read_state().get("ft_pid")
        if pid:
            if self._pid_alive(int(pid)):
                return True
            write_state({"ft_pid": None, "ft_status": "stopped", "ft_running": False})
            return False

        # No stored PID — on restart we cannot reliably find FT without a PID.
        # The wmic scan is too broad and matches AutoEvolve itself when it
        # runs from inside the freqtrade directory. Without a PID we must
        # assume FT is NOT running and let the desired-state logic start it.
        return False

    # ── REST helpers (hot-reload only) ────────────────────────
    def _api_auth(self) -> tuple[bool, str]:
        """
        Authenticate to the FT REST API using HTTP Basic Auth.
        FT's /api/v1/token/login expects Basic Auth, not a JSON body.
        Returns (success, token_or_error).
        """
        try:
            import requests as _req
            api  = cfg("freqtrade", "api", default={})
            host = api.get("host", "127.0.0.1")
            port = api.get("port", 8080)
            user = api.get("username", "freqtrade")
            pwd  = api.get("password", "")
            r = _req.post(
                f"http://{host}:{port}/api/v1/token/login",
                auth=(user, pwd),
                timeout=5,
            )
            if r.status_code == 200:
                return True, r.json().get("access_token", "")
            return False, f"HTTP {r.status_code}"
        except Exception as e:
            return False, str(e)

    def api_open_trade_profits(self) -> list[dict]:
        """
        Fetch current unrealized profit for each open trade via FT REST API.
        Returns list of {pair, profit_pct, profit_abs} or [] on failure.
        """
        try:
            ok, token = self._api_auth()
            if not ok:
                return []
            import requests as _req
            api  = cfg("freqtrade", "api", default={})
            host = api.get("host", "127.0.0.1")
            port = api.get("port", 8080)
            r = _req.get(
                f"http://{host}:{port}/api/v1/status",
                headers={"Authorization": f"Bearer {token}"}, timeout=5,
            )
            if r.status_code != 200:
                return []
            trades = r.json() if isinstance(r.json(), list) else r.json().get("trades", [])
            return [
                {
                    "pair":       t.get("pair", ""),
                    "profit_pct": round(float(t.get("profit_pct", t.get("profit_ratio", 0)) * 100), 3),
                    "profit_abs": round(float(t.get("profit_abs", t.get("total_profit_abs", 0))), 4),
                }
                for t in trades
            ]
        except Exception:
            return []

    def api_reload(self) -> bool:
        ok, token = self._api_auth()
        if not ok:
            return False
        try:
            import requests as _req
            api  = cfg("freqtrade", "api", default={})
            host = api.get("host", "127.0.0.1")
            port = api.get("port", 8080)
            r = _req.post(
                f"http://{host}:{port}/api/v1/reload_config",
                headers={"Authorization": f"Bearer {token}"}, timeout=8,
            )
            if r.status_code == 200:
                append_log("INFO", "FreqTrade hot-reload via REST OK")
                return True
        except Exception as e:
            logger.debug(f"FT REST reload: {e}")
        return False

    # ── Log helpers ───────────────────────────────────────────
    def tail_log(self, n: int = _TAIL_LINES) -> list[str]:
        p = ft_log_path()
        if not p.exists():
            return []
        try:
            # Only read log content written after FT was last started.
            # This prevents old error lines from previous sessions triggering
            # false auto-fixes when the user manually fixes and restarts.
            scan_from = read_state().get("ft_log_scan_from", 0) or 0
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                # If log was rotated/truncated, scan_from > size — read all
                if scan_from > size:
                    scan_from = 0
                # Read at most 65536 bytes from scan_from onwards
                start = max(scan_from, size - 65536)
                f.seek(start)
                raw = f.read().decode("utf-8", errors="replace")
            return raw.splitlines()[-n:]
        except Exception as e:
            logger.debug(f"tail_log: {e}")
            return []

    def check_for_strategy_error(self) -> tuple[bool, str]:
        return classify_ft_error(self.tail_log())

    def check_process_died(self) -> bool:
        state = read_state()
        if not state.get("ft_pid"):
            return False
        if state.get("ft_status") == "stopped":
            return False
        return not self.is_running()


# ══════════════════════════════════════════════════════════════
# FT Health Monitor  (background thread)
# ══════════════════════════════════════════════════════════════
    def reset_database(self) -> bool:
        """Wipe the FT trade DB after a strategy deploy.
        SQL first, file deletion fallback.
        Disable with freqtrade.reset_db_after_evolve: false in config.yaml."""
        if not cfg("freqtrade", "reset_db_after_evolve", default=True):
            return False
        raw = cfg("freqtrade", "db_path", default="")
        if not raw:
            append_log("WARNING", "DB reset skipped — db_path not configured")
            return False
        db_path = Path(raw).expanduser().resolve()
        if not db_path.exists():
            append_log("INFO", "DB reset: file not found, nothing to wipe")
            return True
        try:
            import sqlite3 as _sq
            conn = _sq.connect(str(db_path), timeout=10)
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN TRANSACTION")
            for tbl in ("orders", "trades", "pairlocks",
                        "pairlocks_history", "trade_custom_data"):
                try:
                    conn.execute(f"DELETE FROM {tbl}")
                except Exception:
                    pass
            try:
                conn.execute(
                    "DELETE FROM sqlite_sequence WHERE name IN "
                    "('orders','trades','pairlocks','pairlocks_history','trade_custom_data')"
                )
            except Exception:
                pass
            conn.execute("COMMIT")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("VACUUM")
            conn.close()
            append_log("INFO", f"DB wiped via SQL: {db_path.name}")
            return True
        except Exception as e:
            append_log("WARNING", f"SQL wipe failed ({e}) — deleting DB files")
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                try:
                    p.unlink()
                    append_log("INFO", f"Deleted DB file: {p.name}")
                except Exception as e2:
                    append_log("WARNING", f"Could not delete {p.name}: {e2}")
        return True


class FTHealthMonitor:
    """
    Runs in a daemon thread.  Every `interval` seconds:
      1. If FT is not running → skip (just update state, no noise)
      2. Try to authenticate to the FT REST API
      3. Success → clear error state, update last_ping
      4. Failure → classify the FT log:
           • strategy error → signal orchestrator to auto-fix
           • external error → log warning, no action
    """

    def __init__(self, interval: int = 10):
        self.interval       = interval
        self._stop_evt      = threading.Event()
        self._lock          = threading.Lock()

        # public state (read by server.py)
        self._ft_reachable  = False
        self._last_ping     = ""
        self._last_error    = ""
        self._total_checks  = 0
        self._total_failures = 0
        self._fix_attempts  = 0

        # callback injected by Orchestrator
        self.on_strategy_error = None   # callable(excerpt: str)

        self._ft = FTManager()

    # ── Control ───────────────────────────────────────────────
    def stop(self) -> None:
        self._stop_evt.set()

    def status(self) -> dict:
        with self._lock:
            return {
                "ft_reachable":    self._ft_reachable,
                "last_ping":       self._last_ping,
                "last_error":      self._last_error,
                "total_checks":    self._total_checks,
                "total_failures":  self._total_failures,
                "fix_attempts":    self._fix_attempts,
            }

    def increment_fix_attempts(self) -> None:
        with self._lock:
            self._fix_attempts += 1

    def reset_fix_attempts(self) -> None:
        with self._lock:
            self._fix_attempts = 0

    # ── Main loop ─────────────────────────────────────────────
    def run(self) -> None:
        while not self._stop_evt.wait(timeout=self.interval):
            try:
                self._check()
                write_state({"orch_alive": local_str()})  # backup heartbeat during LLM calls
            except Exception as e:
                logger.debug(f"FTHealthMonitor tick error: {e}")

    def _check(self) -> None:
        state = read_state()

        # Skip ping when neither FT nor evolve is desired running — no log noise
        ft_desired     = state.get("ft_desired",     "stopped")
        evolve_desired = state.get("evolve_desired", "stopped")
        if ft_desired != "running" and evolve_desired != "running":
            with self._lock:
                self._ft_reachable = False
            return

        # Check actual process — don't trust stale state.ft_running
        pid_alive = self._ft.is_running()

        if not pid_alive:
            if state.get("ft_running"):
                write_state({"ft_running": False, "ft_status": "stopped", "ft_pid": None})
            if ft_desired == "running" and state.get("ft_running"):
                append_log("WARNING", "FT desired=running but process not found")
            with self._lock:
                self._ft_reachable = False
            return

        with self._lock:
            self._total_checks += 1

        ok, tok_or_err = self._ft._api_auth()
        now = local_str()

        if ok:
            with self._lock:
                self._ft_reachable = True
                self._last_ping    = now
                self._last_error   = ""
            write_state({"ft_api_ok": True, "ft_last_ping": now})

            # Even with API healthy, scan log for strategy runtime errors
            # (FT stays running with API up but strategy can fail per-pair)
            with self._lock:
                self._total_checks_since_scan = getattr(self, "_total_checks_since_scan", 0) + 1
            if self._total_checks_since_scan >= 3:  # every ~30s (3 × 10s interval)
                self._total_checks_since_scan = 0
                is_strategy_err, excerpt = self._ft.check_for_strategy_error()
                if is_strategy_err:
                    # Only trigger if this is a NEW error (not already being fixed)
                    state = read_state()
                    already_fixing = state.get("status") == "fixing_ft_error"
                    if not already_fixing:
                        append_log("WARNING",
                            f"FT health: strategy runtime error detected in log "
                            f"(API still OK) — requesting auto-fix")
                        write_state({"ft_error_full":  excerpt,
                                     "ft_error":       excerpt[:300]})
                        if self.on_strategy_error:
                            try:
                                self.on_strategy_error(excerpt)
                            except Exception as e:
                                logger.error(f"on_strategy_error callback error: {e}")
                else:
                    # Log clean — clear any stale error panel if not already fixed
                    state = read_state()
                    if state.get("ft_error_full") and state.get("status") != "fixing_ft_error":
                        write_state({"ft_error": None, "ft_error_full": None,
                                     "ft_error_count": 0})
            return

        # Auth failed while FT should be running
        with self._lock:
            self._ft_reachable  = False
            self._total_failures += 1
            self._last_error    = tok_or_err

        write_state({"ft_api_ok": False, "ft_last_ping": now})

        # Check if it's still a running process at all
        if not self._ft.is_running():
            # Process is dead — scan log to see if a strategy error caused the crash
            append_log("WARNING",
                f"FT health: process dead (API unreachable) — scanning log for cause")
            is_strategy_err, excerpt = self._ft.check_for_strategy_error()
            if is_strategy_err:
                state = read_state()
                already_fixing = state.get("status") == "fixing_ft_error"
                if not already_fixing:
                    append_log("WARNING",
                        f"FT health: strategy error caused crash — requesting auto-fix")
                    write_state({"ft_error_full": excerpt, "ft_error": excerpt[:300]})
                    if self.on_strategy_error:
                        try:
                            self.on_strategy_error(excerpt, crashed=True)
                        except Exception as e:
                            logger.error(f"on_strategy_error callback error: {e}")
            else:
                append_log("WARNING",
                    f"FT health: process dead, no strategy error found — "
                    f"may be external crash or OOM ({tok_or_err})")
            return

        # Process alive but API down — classify the log
        is_strategy_err, excerpt = self._ft.check_for_strategy_error()

        if is_strategy_err:
            append_log("WARNING",
                f"FT health: API auth failed + strategy error detected — "
                f"requesting auto-fix")
            if self.on_strategy_error:
                try:
                    self.on_strategy_error(excerpt)
                except Exception as e:
                    logger.error(f"on_strategy_error callback error: {e}")
        else:
            # External / network error — just log, no action
            append_log("WARNING",
                f"FT health: API auth failed (external issue?) — "
                f"{tok_or_err}")


# ══════════════════════════════════════════════════════════════
# Deployer
# ══════════════════════════════════════════════════════════════
class Deployer:
    def __init__(self, checkpoints: CheckpointManager):
        self.cp = checkpoints
        self.ft = FTManager()

    def deploy(self, code: str, gen: int, metrics: dict,
               reason: str, changelog: str) -> bool:
        code = self._inject_meta(code, gen, gen - 1, reason, metrics, changelog)
        self.cp.save(gen, code, metrics, reason, changelog)

        sname = cfg("freqtrade", "strategy_name", default="AutoEvolveStrategy")
        dest  = Path(cfg("freqtrade", "strategies_dir", default="")).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{sname}.py").write_text(code, encoding="utf-8")
        append_log("INFO", f"Gen {gen} deployed → {dest / sname}.py")

        reloaded = self.ft.api_reload()
        if not reloaded:
            append_log("WARNING", "REST reload failed — restarting FT")
            reloaded = self.ft.restart()

        write_state({
            "current_gen":    gen,
            "deployed_at":    local_str(),
            "is_rollback":    False,
            "last_changelog": changelog,
            "ft_error":       None,
            "ft_error_count": 0,
        })
        return reloaded

    def rollback(self, to_gen: int) -> bool:
        code = self.cp.load_code(to_gen)
        if not code:
            append_log("ERROR", f"Rollback failed — gen {to_gen} not in checkpoints")
            return False
        sname = cfg("freqtrade", "strategy_name", default="AutoEvolveStrategy")
        dest  = Path(cfg("freqtrade", "strategies_dir", default="")).expanduser()
        (dest / f"{sname}.py").write_text(code, encoding="utf-8")
        append_log("WARNING", f"Rolled back to gen {to_gen}")
        reloaded = self.ft.api_reload()
        if not reloaded:
            self.ft.restart()
        write_state({"current_gen": to_gen, "is_rollback": True})
        return True

    def should_rollback(self, post_metrics: dict) -> bool:
        m       = post_metrics.get("metrics", {})
        wr      = m.get("win_rate", 1.0)
        mdd     = m.get("max_drawdown", 0.0)
        min_wr  = cfg("rollback", "rollback_if_winrate_below",  default=0.35)
        max_mdd = cfg("rollback", "rollback_if_drawdown_above", default=0.15)
        if wr < min_wr:
            append_log("WARNING", f"Win rate {wr:.1%} < {min_wr:.1%} → rollback")
            return True
        if mdd > max_mdd:
            append_log("WARNING", f"Drawdown {mdd:.1%} > {max_mdd:.1%} → rollback")
            return True
        return False

    def _inject_meta(self, code, gen, parent, reason, metrics, changelog) -> str:
        ms = {k: metrics.get("metrics", {}).get(k)
              for k in ["win_rate", "profit_factor", "sharpe", "max_drawdown"]}
        block = (
            f"# GENERATION_METADATA = {{\n"
            f'#   "generation": {gen},\n'
            f'#   "parent": {parent},\n'
            f'#   "created_at": "{local_str()}",\n'
            f'#   "trigger": "{reason}",\n'
            f'#   "metrics": {json.dumps(ms)},\n'
            f'#   "changelog": "{changelog[:200]}"\n'
            f"# }}"
        )
        pat = r"# GENERATION_METADATA = \{.*?# \}"
        if re.search(pat, code, re.DOTALL):
            # Use lambda so `block` is never interpreted as a regex replacement
            # string (backslashes in Windows paths like D:\Python would cause
            # re.error: bad escape otherwise)
            return re.sub(pat, lambda _: block, code, flags=re.DOTALL)
        lines = code.split("\n")
        return lines[0] + "\n" + block + "\n" + "\n".join(lines[1:])
