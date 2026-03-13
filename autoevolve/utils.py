"""
autoevolve/utils.py
All paths are ABSOLUTE, anchored to BASE_DIR (the project folder).
Never relative to CWD — works regardless of where you launch from.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Rotation constants ─────────────────────────────────────────
LOG_MAX_BYTES    = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT = 5

# ── Project root ───────────────────────────────────────────────
# This file: <root>/autoevolve/utils.py  →  parent.parent = <root>
BASE_DIR     = Path(__file__).resolve().parent.parent
STATE_DIR    = BASE_DIR / ".autoevolve"
STATE_FILE   = STATE_DIR / "state.json"
CONTROL_FILE = STATE_DIR / "control.json"
LOG_FILE     = STATE_DIR / "app.log"
CONFIG_FILE  = BASE_DIR / "config.yaml"

STATE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Config loader
# ══════════════════════════════════════════════════════════════
_cfg_lock:  threading.Lock = threading.Lock()
_cfg_cache: dict  = {}
_cfg_mtime: float = 0.0


def load_config(force: bool = False) -> dict:
    global _cfg_cache, _cfg_mtime
    try:
        mtime = CONFIG_FILE.stat().st_mtime
    except FileNotFoundError:
        return _cfg_cache or {}
    with _cfg_lock:
        if force or mtime != _cfg_mtime:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                _cfg_cache = yaml.safe_load(f) or {}
            _cfg_mtime = mtime
    return _cfg_cache


def cfg(*keys: str, default: Any = None) -> Any:
    d = load_config()
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


# ══════════════════════════════════════════════════════════════
# Datetime helpers
# ══════════════════════════════════════════════════════════════
def local_now() -> datetime:
    return datetime.now().astimezone()


def local_str(dt: Optional[datetime] = None) -> str:
    """Timestamp with numeric UTC offset — never long Windows TZ names."""
    if dt is None:
        dt = local_now()
    offset = dt.strftime("%z")
    if len(offset) == 5:
        offset = offset[:3] + ":" + offset[3:]
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " " + offset


def utc_to_local_str(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return local_str(dt.astimezone())
    except Exception:
        return iso_str


# ══════════════════════════════════════════════════════════════
# Logging setup
# ══════════════════════════════════════════════════════════════
class _AppLogHandler(logging.Handler):
    """
    Bridge from Python logging -> append_log() so every logger.xxx() call
    also appears in .autoevolve/app.log (read by the web dashboard).
    Only attaches to the 'autoevolve' logger subtree to avoid noise.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            lvl = record.levelname  # INFO / WARNING / ERROR / DEBUG
            msg = self.format(record)
            _rotate_app_log()
            entry = {"ts": local_str(), "level": lvl, "msg": msg}
            with _log_lock:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass


def setup_logging() -> None:
    raw = cfg("logging", "directory", default="logs")
    p   = Path(raw).expanduser()
    log_dir = p if p.is_absolute() else (BASE_DIR / p).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, cfg("logging", "level", default="INFO"), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt.converter = time.localtime

    log_path = log_dir / f"autoevolve_{local_now().strftime('%Y%m%d')}.log"
    print(f"[AutoEvolve] Log file: {log_path}", flush=True)

    rotating = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
    )
    rotating.setFormatter(fmt)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(rotating)
    root.addHandler(stream)

    # Bridge: autoevolve.* loggers also write to app.log (dashboard viewer)
    app_handler = _AppLogHandler()
    app_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    app_handler.setLevel(level)
    ae_logger = logging.getLogger("autoevolve")
    # Only add once
    if not any(isinstance(h, _AppLogHandler) for h in ae_logger.handlers):
        ae_logger.addHandler(app_handler)


# ══════════════════════════════════════════════════════════════
# Structured app log  (.autoevolve/app.log — dashboard viewer)
# ══════════════════════════════════════════════════════════════
_log_lock = threading.Lock()


def _rotate_app_log() -> None:
    if not LOG_FILE.exists():
        return
    if LOG_FILE.stat().st_size < LOG_MAX_BYTES:
        return
    for i in range(LOG_BACKUP_COUNT, 0, -1):
        src = Path(f"{LOG_FILE}.{i - 1}") if i > 1 else LOG_FILE
        dst = Path(f"{LOG_FILE}.{i}")
        if src.exists():
            try:
                src.rename(dst)
            except Exception:
                pass


def append_log(level: str, msg: str) -> None:
    entry = {"ts": local_str(), "level": level.upper(), "msg": msg}
    try:
        with _log_lock:
            _rotate_app_log()
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[append_log error] {e}", file=sys.stderr)


def read_logs(n: int = 300) -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
        result = []
        for line in lines[-n:]:
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return list(reversed(result))
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
# Rotating writer (journal, etc.)
# ══════════════════════════════════════════════════════════════
def rotating_write(path: Path, content: str) -> None:
    try:
        if path.exists() and path.stat().st_size >= LOG_MAX_BYTES:
            for i in range(LOG_BACKUP_COUNT, 0, -1):
                src = Path(f"{path}.{i - 1}") if i > 1 else path
                dst = Path(f"{path}.{i}")
                if src.exists():
                    try:
                        src.rename(dst)
                    except Exception:
                        pass
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[rotating_write error] {e}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════
# State file
# ══════════════════════════════════════════════════════════════
_state_lock = threading.Lock()


def write_state(update: dict) -> None:
    try:
        with _state_lock:
            existing = _read_state_raw()
            existing.update(update)
            existing["updated_at"] = local_str()
            STATE_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[write_state error] {e}", file=sys.stderr)


def read_state() -> dict:
    with _state_lock:
        return _read_state_raw()


def _read_state_raw() -> dict:
    try:
        return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════
# Control file
# ══════════════════════════════════════════════════════════════
_control_lock: threading.Lock = threading.Lock()

def send_control(action: str, **kwargs) -> None:
    cmd = {"action": action, "sent_at": local_str(), **kwargs}
    with _control_lock:
        try:
            queue: list = []
            if CONTROL_FILE.exists():
                try:
                    existing = json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
                    if isinstance(existing, list):
                        queue = existing
                    elif isinstance(existing, dict):
                        queue = [existing]
                except Exception:
                    queue = []
            queue.append(cmd)
            CONTROL_FILE.write_text(json.dumps(queue, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[send_control error] {e}", file=sys.stderr)


def consume_control() -> Optional[dict]:
    with _control_lock:
        if not CONTROL_FILE.exists():
            return None
        try:
            existing = json.loads(CONTROL_FILE.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                queue = existing
            elif isinstance(existing, dict):
                queue = [existing]
            else:
                CONTROL_FILE.unlink(missing_ok=True)
                return None
            if not queue:
                CONTROL_FILE.unlink(missing_ok=True)
                return None
            cmd = queue.pop(0)
            if queue:
                CONTROL_FILE.write_text(json.dumps(queue, indent=2), encoding="utf-8")
            else:
                CONTROL_FILE.unlink(missing_ok=True)
            return cmd
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════
# Venv / executable resolver
# ══════════════════════════════════════════════════════════════
def resolve_executable(venv_path: str, name: str) -> Optional[Path]:
    """
    Return the absolute path to `name` inside `venv_path`.
    Returns None if venv_path is empty or the file doesn't exist.

    Windows: {venv}/Scripts/{name}.exe
    Linux:   {venv}/bin/{name}
    """
    if not venv_path or not venv_path.strip():
        return None
    venv = Path(venv_path).expanduser().resolve()
    if platform.system() == "Windows":
        candidates = [
            venv / "Scripts" / (name + ".exe"),
            venv / "Scripts" / name,
        ]
    else:
        candidates = [venv / "bin" / name]
    for c in candidates:
        if c.exists():
            return c
    return None


def ft_python() -> Path:
    """Python in the FreqTrade venv. Falls back to sys.executable."""
    p = resolve_executable(cfg("freqtrade", "venv", default=""), "python")
    return p if p is not None else Path(sys.executable)


def ft_executable() -> Optional[Path]:
    """
    FreqTrade CLI in the FT venv.
    Returns None if not found — caller should use `ft_python() -m freqtrade`.
    """
    return resolve_executable(cfg("freqtrade", "venv", default=""), "freqtrade")


def ft_config_path() -> Path:
    raw = cfg("freqtrade", "config_file", default="config.json")
    p   = Path(raw).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def ft_log_path() -> Path:
    raw = cfg("freqtrade", "logfile", default="")
    if not raw:
        return (BASE_DIR / "logs" / "freqtrade.log").resolve()
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def ft_root_path() -> Path:
    """
    Working directory for FreqTrade — where user_data/ lives.
    Set freqtrade.root in config.yaml.  Defaults to BASE_DIR parent
    (assuming AutoEvolveStrategy lives inside user_data/strategies/).
    """
    raw = cfg("freqtrade", "root", default="")
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p).resolve()
    # Auto-detect: walk up from BASE_DIR until we find user_data/
    candidate = BASE_DIR
    for _ in range(5):
        if (candidate / "user_data").is_dir():
            return candidate
        candidate = candidate.parent
    return BASE_DIR
