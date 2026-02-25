#!/usr/bin/env python3
"""
main.py — AutoEvolve entry point
=================================
Run:  python main.py
      python main.py --poll 30

Single process. Starts:
  - FT health monitor thread  (auth-ping every 10 s)
  - Orchestrator loop thread  (evolution logic)
  - uvicorn HTTP server       (dashboard + REST API + WebSocket)

Single-instance enforced via .autoevolve/main.pid
"""
from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import threading
from pathlib import Path

from autoevolve.utils import (
    BASE_DIR, STATE_DIR, cfg, load_config, local_str,
    setup_logging, append_log, write_state,
)

load_config()
setup_logging()

import logging
logger = logging.getLogger("autoevolve.main")


# ══════════════════════════════════════════════════════════════
# Single-instance lock  (PID file)
# ══════════════════════════════════════════════════════════════
PID_FILE = STATE_DIR / "main.pid"


def _pid_alive(pid: int) -> bool:
    try:
        if platform.system() == "Windows":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in r.stdout
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def _acquire_lock() -> None:
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if _pid_alive(old_pid) and old_pid != os.getpid():
                print(
                    f"\n[AutoEvolve] Already running (PID {old_pid}).\n"
                    f"  Kill it: taskkill /F /PID {old_pid}  (Windows)\n"
                    f"           kill {old_pid}               (Linux)\n",
                    flush=True,
                )
                sys.exit(1)
        except (ValueError, OSError):
            pass

    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    import atexit
    atexit.register(_release_lock)

    def _sigterm(*_):
        _release_lock()
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except (OSError, ValueError):
        pass


def _release_lock() -> None:
    try:
        if PID_FILE.exists():
            stored = int(PID_FILE.read_text().strip())
            if stored == os.getpid():
                PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="AutoEvolve")
    parser.add_argument("--poll", type=int,   default=None)
    parser.add_argument("--host",             default=None)
    parser.add_argument("--port", type=int,   default=None)
    args = parser.parse_args()

    _acquire_lock()

    poll    = args.poll or cfg("dashboard", "poll_secs",           default=60)
    host    = args.host or cfg("dashboard", "host",                default="0.0.0.0")
    port    = args.port or cfg("dashboard", "port",                default=8501)
    ft_ping = cfg("freqtrade", "health_check_interval",            default=10)

    logger.info(f"AutoEvolve starting at {local_str()}")
    append_log("INFO",
        f"=== AutoEvolve startup === PID={os.getpid()} "
        f"poll={poll}s ft_ping={ft_ping}s")
    write_state({"ae_pid": os.getpid(), "ae_started": local_str()})

    # 1. FT health monitor
    from autoevolve.deployer import FTHealthMonitor
    monitor = FTHealthMonitor(interval=ft_ping)
    threading.Thread(target=monitor.run, daemon=True,
                     name="ft-health-monitor").start()
    append_log("INFO", f"FT health monitor started (interval={ft_ping}s)")

    # 2. Orchestrator
    from autoevolve.orchestrator import Orchestrator
    orc = Orchestrator(monitor=monitor)
    threading.Thread(target=orc.run, args=(poll,),
                     daemon=True, name="orchestrator").start()
    append_log("INFO", f"Orchestrator started (poll={poll}s)")

    # 3. uvicorn (blocks main thread)
    import uvicorn
    from autoevolve.server import make_app

    app = make_app(orc, monitor)
    append_log("INFO", f"Dashboard -> http://{host}:{port}")
    print(f"\n[AutoEvolve] Dashboard -> http://localhost:{port}\n", flush=True)

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        append_log("INFO", "AutoEvolve shutting down")
        monitor.stop()
        _release_lock()


if __name__ == "__main__":
    main()
