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
_STRATEGY_ERROR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"AttributeError", r"NameError", r"SyntaxError", r"IndentationError",
    r"TypeError.*populate_", r"TypeError.*custom_stoploss",
    r"TypeError.*leverage", r"TypeError.*confirm_trade", r"TypeError.*custom_exit",
    r"ImportError.*strategy", r"KeyError.*dataframe", r"ValueError.*indicator",
    r"Strategy.*not found", r"Could not load strategy", r"Error loading strategy",
    r"populate_indicators.*error", r"populate_entry_trend.*error",
    r"populate_exit_trend.*error", r"failed.*strategy", r"strategy.*failed",
]]

_EXTERNAL_ERROR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"ConnectionError", r"TimeoutError", r"ReadTimeout", r"ConnectTimeout",
    r"requests\.exceptions", r"urllib3", r"socket\.", r"NetworkError",
    r"RateLimit", r"DDosProtection", r"ExchangeNotAvailable",
    r"retrying.*connection", r"Could not connect", r"SSL.*error",
    r"EOF occurred", r"Temporary failure", r"No route to host",
]]


def classify_ft_error(log_lines: list[str]) -> tuple[bool, str]:
    """Returns (is_strategy_error, excerpt)."""
    for line in log_lines:
        for pat in _EXTERNAL_ERROR_PATTERNS:
            if pat.search(line):
                return False, ""
    matched = []
    for line in log_lines:
        for pat in _STRATEGY_ERROR_PATTERNS:
            if pat.search(line):
                matched.append(line.strip())
                break
    if matched:
        return True, "\n".join(matched[:15])
    return False, ""


# ══════════════════════════════════════════════════════════════
# FT Manager
# ══════════════════════════════════════════════════════════════
class FTManager:
    _TAIL_LINES  = 200
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

            write_state({
                "ft_pid":         proc.pid,
                "ft_status":      "starting",
                "ft_running":     False,
            "ft_desired":     "running",
                "ft_error":       None,
                "ft_error_count": 0,
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
            write_state({"ft_pid": None, "ft_status": "stopped", "ft_running": False, "ft_desired": "stopped"})
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
            write_state({"ft_pid": None, "ft_status": "stopped", "ft_running": False, "ft_desired": "stopped"})
            return False

        # No stored PID — scan for any freqtrade process
        try:
            if platform.system() == "Windows":
                r = subprocess.run(
                    ["wmic", "process", "where",
                     "name='python.exe' or name='freqtrade.exe'",
                     "get", "commandline"],
                    capture_output=True, text=True, timeout=8,
                )
                for line in r.stdout.splitlines():
                    ll = line.lower()
                    if "freqtrade" in ll and "trade" in ll:
                        return True
                return False
            else:
                r = subprocess.run(
                    ["pgrep", "-f", "freqtrade trade"],
                    capture_output=True, text=True,
                )
                return bool(r.stdout.strip())
        except Exception:
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
            with open(p, "rb") as f:
                f.seek(0, 2)
                size  = f.tell()
                block = min(size, 65536)
                f.seek(max(0, size - block))
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
            return

        # Auth failed while FT should be running
        with self._lock:
            self._ft_reachable  = False
            self._total_failures += 1
            self._last_error    = tok_or_err

        write_state({"ft_api_ok": False, "ft_last_ping": now})

        # Check if it's still a running process at all
        if not self._ft.is_running():
            # Process is dead — let orchestrator's process-died logic handle it
            append_log("WARNING",
                f"FT health: API unreachable and process not found "
                f"({tok_or_err})")
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
            return re.sub(pat, block, code, flags=re.DOTALL)
        lines = code.split("\n")
        return lines[0] + "\n" + block + "\n" + "\n".join(lines[1:])
