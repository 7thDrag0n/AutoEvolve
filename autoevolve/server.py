"""
autoevolve/server.py
====================
FastAPI dashboard server.

Endpoints:
  GET  /                  -> index.html
  GET  /api/state         -> current state.json
  GET  /api/logs?n=200    -> last N app-log entries
  GET  /api/ft-log?n=100  -> last N FT log lines
  GET  /api/checkpoints   -> all checkpoint metadata
  GET  /api/ft-health     -> FTHealthMonitor status
  POST /api/control       -> {"action": "...", ...}
  WS   /ws                -> push {state, logs, health} every 2 s
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .utils import read_state, read_logs, send_control, append_log

if TYPE_CHECKING:
    from .orchestrator import Orchestrator
    from .deployer import FTHealthMonitor

logger = logging.getLogger("autoevolve.server")

TEMPLATES_DIR = Path(__file__).parent / "templates"


def make_app(orc, monitor) -> FastAPI:
    app = FastAPI(title="AutoEvolve")
    _register_routes(app, orc, monitor)
    return app


def _register_routes(app: FastAPI, orc, monitor) -> None:

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = TEMPLATES_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    @app.get("/api/state")
    async def get_state():
        return JSONResponse(read_state())

    @app.get("/api/logs")
    async def get_logs(n: int = 200):
        return JSONResponse(read_logs(n))

    @app.get("/api/ft-log")
    async def get_ft_log(n: int = 100):
        try:
            lines = orc.deployer.ft.tail_log(n)
        except Exception:
            lines = []
        return JSONResponse({"lines": lines})

    @app.get("/api/checkpoints")
    async def get_checkpoints():
        try:
            meta = orc.checkpoints.all_meta()
        except Exception:
            meta = []
        return JSONResponse(meta)

    @app.get("/api/ft-startup")
    async def get_ft_startup():
        """Returns last FT startup stdout/stderr log."""
        try:
            startup_log = orc.ft._STARTUP_LOG
            if startup_log.exists():
                lines = startup_log.read_text(encoding="utf-8", errors="replace")
                return JSONResponse({"text": lines})
        except Exception as e:
            return JSONResponse({"text": f"Error reading startup log: {e}"})
        return JSONResponse({"text": "(no startup log yet — start FreqTrade first)"})

    @app.get("/api/ft-health")
    async def get_ft_health():
        return JSONResponse(monitor.status())

    @app.post("/api/control")
    async def post_control(payload: dict):
        action = payload.get("action", "")
        if not action:
            return JSONResponse({"ok": False, "error": "missing action"}, status_code=400)
        send_control(action, **{k: v for k, v in payload.items() if k != "action"})
        append_log("INFO", f"Dashboard control: {action}")
        return JSONResponse({"ok": True, "action": action})

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                payload = {
                    "state":  read_state(),
                    "logs":   read_logs(50),
                    "health": monitor.status(),
                }
                await ws.send_text(json.dumps(payload))
                await asyncio.sleep(2)
        except (WebSocketDisconnect, Exception):
            pass
