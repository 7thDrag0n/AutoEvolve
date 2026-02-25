"""
autoevolve/journal.py
=====================
Append-only JSONL evolution journal with rotation (5 × 10 MB).
Every generation event is recorded: triggered, deployed, confirmed,
rolled_back, skipped, llm_failed, ft_error_fixed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .utils import cfg, local_str, append_log, rotating_write

logger = logging.getLogger("autoevolve.journal")

EV_TRIGGERED    = "triggered"
EV_DEPLOYED     = "deployed"
EV_CONFIRMED    = "confirmed"
EV_ROLLED_BACK  = "rolled_back"
EV_SKIPPED      = "skipped"
EV_LLM_FAILED   = "llm_failed"
EV_FT_ERR_FIXED = "ft_error_fixed"   # FT crashed on strategy error → auto-fixed


class Journal:
    """Append-only JSONL log of every evolution event (with rotation)."""

    def __init__(self):
        log_dir = Path(cfg("logging", "directory", default="./logs")).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / "evolution_journal.jsonl"

    def record(self, event: str, gen: int, data: Optional[dict] = None) -> None:
        entry = {
            "ts":    local_str(),
            "event": event,
            "gen":   gen,
            "data":  data or {},
        }
        rotating_write(self._path, json.dumps(entry) + "\n")
        append_log("INFO", f"JOURNAL {event} gen={gen}")
        logger.info(f"Journal: {event} gen={gen}")

    def all(self) -> list[dict]:
        out: list[dict] = []
        if not self._path.exists():
            return out
        for line in self._path.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out

    def history_for_llm(self, n: int = 10) -> str:
        relevant = [e for e in self.all()
                    if e["event"] in (EV_DEPLOYED, EV_ROLLED_BACK,
                                      EV_CONFIRMED, EV_FT_ERR_FIXED)][-n:]
        if not relevant:
            return "No prior evolution history."
        lines = ["=== EVOLUTION HISTORY (most recent last) ==="]
        for e in relevant:
            m  = e.get("data", {}).get("metrics", {})
            cl = e.get("data", {}).get("changelog",
                 e.get("data", {}).get("reason", ""))
            status = {
                EV_DEPLOYED:    "✓ DEPLOYED",
                EV_CONFIRMED:   "✓ CONFIRMED",
                EV_ROLLED_BACK: "✗ ROLLED BACK",
                EV_FT_ERR_FIXED: "⚠ FT-ERROR-FIX",
            }.get(e["event"], e["event"].upper())
            lines.append(
                f"[{e['ts'][:10]}] Gen {e['gen']:>3} {status:16} | "
                f"PF={m.get('profit_factor','?')}  "
                f"WR={m.get('win_rate','?')}  "
                f"DD={m.get('max_drawdown','?')}  "
                f"Sharpe={m.get('sharpe','?')} | "
                f"{str(cl)[:80]}"
            )
        return "\n".join(lines)
