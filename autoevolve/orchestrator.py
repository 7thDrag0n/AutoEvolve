"""
autoevolve/orchestrator.py
==========================
Main control loop for the AutoEvolve system.

Responsibilities:
  1. Monitor FreqTrade SQLite for trade trigger conditions
  2. Detect N consecutive losses → call LLM → deploy new generation
  3. Monitor post-deploy trades → confirm or rollback
  4. Process real-time control commands from the dashboard
  5. Expose _handle_strategy_error() for FTHealthMonitor to call

FT process health (10 s ping + auto-fix) is handled by ft_monitor.py.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from .utils import (
    cfg, load_config, local_str, append_log,
    write_state, read_state, consume_control,
)
from .harvester import Harvester
from .improver  import LLMImprover
from .deployer  import Deployer, CheckpointManager, FTManager
from .journal   import (
    Journal,
    EV_TRIGGERED, EV_DEPLOYED, EV_CONFIRMED,
    EV_ROLLED_BACK, EV_SKIPPED, EV_LLM_FAILED, EV_FT_ERR_FIXED,
)
from .strategy_template import baseline_template

logger = logging.getLogger("autoevolve.orchestrator")


class Orchestrator:
    """
    Main event loop.  main.py calls _tick() in a loop.
    """

    MAX_FIX_ATTEMPTS = 5

    def __init__(self, monitor=None):
        self.harvester   = Harvester()
        self.checkpoints = CheckpointManager()
        self.deployer    = Deployer(self.checkpoints)
        self.improver    = LLMImprover()
        self.journal     = Journal()
        self.ft          = FTManager()
        self._monitor    = monitor
        if monitor is not None:
            monitor.on_strategy_error = self._handle_strategy_error

        self._paused:         bool             = False
        self._monitoring:     bool             = False
        self._last_evo:       Optional[datetime] = None
        self._deploy_at:      Optional[str]    = None
        self._current_gen:    int              = self.checkpoints.latest_gen()

        # FT error auto-fix state (written by _handle_strategy_error)
        self._fix_attempts:   int  = 0
        self._last_error_sig: str  = ""
        self._last_open_trade: datetime = datetime.now().astimezone()  # last time an open trade was seen
        self._evolving:       bool     = False  # True while LLM call in progress

        self._ensure_baseline()

        # Read persistent desired states (survive restarts)
        stale = read_state()
        ft_desired     = stale.get("ft_desired",     "stopped")
        evolve_desired = stale.get("evolve_desired", "stopped")

        # Clear stale ft_pid from previous crashed session
        if stale.get("ft_pid"):
            write_state({"ft_pid": None, "ft_status": "unknown", "ft_running": False})

        self._paused = (evolve_desired != "running")

        write_state({
            "current_gen":    self._current_gen,
            "status":         "running" if not self._paused else "paused",
            "paused":         self._paused,
            "monitoring":     False,
            "ft_running":     self.ft.is_running(),
            "ft_error":       None,
            "ft_error_count": 0,
            "ft_desired":     ft_desired,
            "evolve_desired": evolve_desired,
            "orch_started":   local_str(),
            "orch_alive":     local_str(),
        })

        if ft_desired == "running" and not self.ft.is_running():
            append_log("INFO", "ft_desired=running — auto-starting FreqTrade")
            self.ft.start()

        append_log("INFO",
            f"Orchestrator initialised — gen {self._current_gen} "
            f"ft_desired={ft_desired} evolve_desired={evolve_desired}")

    # ── Main loop ──────────────────────────────────────────────
    def run(self, poll: int = 60) -> None:
        append_log("INFO", f"AutoEvolve loop started (poll={poll}s)")
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                msg = f"Tick error: {e}"
                logger.error(msg, exc_info=True)
                append_log("ERROR", msg)
            time.sleep(poll)

    # ── Single tick (called by main.py every poll seconds) ────
    def _tick(self) -> None:
        load_config()
        write_state({"orch_alive": local_str()})
        self._handle_control()

        ft_running = self.ft.is_running()
        write_state({"ft_running": ft_running})

        if self._paused:
            write_state({"paused": True, "status": "paused", "evolve_desired": "stopped"})
            return

        write_state({"paused": False})

        # ── Rollback monitoring window ─────────────────────────
        if self._monitoring:
            self._check_rollback()
            return

        # ── Evolution trigger ──────────────────────────────────
        losses    = self.harvester.consecutive_losses()
        threshold = cfg("trigger", "consecutive_losses",        default=3)
        min_t     = cfg("trigger", "min_trades_before_trigger", default=10)
        cooldown  = cfg("trigger", "cooldown_minutes",          default=60)

        snap  = self.harvester.snapshot()
        total = snap.get("total_closed", 0)

        write_state({
            "status":             "running",
            "current_gen":        self._current_gen,
            "consecutive_losses": losses,
            "trigger_threshold":  threshold,
            "total_trades":       total,
            "metrics":            snap.get("metrics", {}),
            "snapshot_at":        local_str(),
        })

        # ── Cooldown check (shared by all triggers) ────────────
        def _in_cooldown() -> bool:
            if self._last_evo is None:
                return False
            elapsed = (datetime.now().astimezone() - self._last_evo).total_seconds() / 60
            remaining = cooldown - elapsed
            if remaining > 0:
                write_state({"cooldown_remaining_min": round(remaining, 1)})
                return True
            write_state({"cooldown_remaining_min": 0})
            return False

        # ── Track open trade activity ───────────────────────────
        open_trades = snap.get("total_open", 0)
        if open_trades > 0:
            self._last_open_trade = datetime.now().astimezone()

        # ── Idle trigger: no open trades for X minutes ──────────
        idle_minutes = cfg("trigger", "idle_trigger_minutes", default=0)
        if idle_minutes > 0 and ft_running:
            idle_elapsed = (
                datetime.now().astimezone() - self._last_open_trade
            ).total_seconds() / 60
            write_state({"idle_minutes": round(idle_elapsed, 1),
                         "idle_trigger_minutes": idle_minutes})
            if idle_elapsed >= idle_minutes and not _in_cooldown():
                msg = (f"IDLE TRIGGER: no open trades for "
                       f"{idle_elapsed:.1f} min (threshold={idle_minutes}) "
                       f"-> evolving gen {self._current_gen} -> {self._current_gen + 1}")
                logger.warning(msg); append_log("WARNING", msg)
                self.journal.record(EV_TRIGGERED, self._current_gen,
                                    {"reason": "idle", "idle_minutes": idle_elapsed,
                                     "metrics": snap.get("metrics", {})})
                self._evolve(snap, "idle_no_open_trades")
                return

        # ── Loss trigger ────────────────────────────────────────
        if total < min_t:
            write_state({"waiting_for_trades": min_t - total})
            return

        write_state({"waiting_for_trades": 0})

        if _in_cooldown():
            return

        if losses >= threshold:
            reason = f"{losses}_consecutive_losses"
            msg = (f"TRIGGER: {losses} consecutive losses "
                   f"-> evolving gen {self._current_gen} -> {self._current_gen + 1}")
            logger.warning(msg); append_log("WARNING", msg)
            self.journal.record(EV_TRIGGERED, self._current_gen,
                                {"losses": losses, "metrics": snap.get("metrics", {})})
            self._evolve(snap, reason)

    # ── Strategy error handler (called from ft_monitor.py too) ─
    def _handle_strategy_error(self, excerpt: str, crashed: bool) -> None:
        """
        Called by FTHealthMonitor when a strategy-caused error is detected.
        Calls the LLM to fix the code and redeploys.
        Retries up to MAX_FIX_ATTEMPTS before giving up.
        """
        sig = excerpt[:120]
        if sig == self._last_error_sig:
            if self._fix_attempts >= self.MAX_FIX_ATTEMPTS:
                msg = (f"❌ Max FT error fix attempts "
                       f"({self.MAX_FIX_ATTEMPTS}) reached — manual intervention required.")
                logger.error(msg); append_log("ERROR", msg)
                write_state({"ft_status": "error_max_retries", "ft_error": excerpt[:300]})
                return
        else:
            self._fix_attempts   = 0
            self._last_error_sig = sig

        self._fix_attempts += 1
        write_state({
            "status":         "fixing_ft_error",
            "ft_error":       excerpt[:300],
            "ft_error_count": self._fix_attempts,
        })

        crash_or_live = "crashed" if crashed else "running but erroring"
        msg = (f"⚠ FT strategy error detected ({crash_or_live}) "
               f"— fix attempt {self._fix_attempts}/{self.MAX_FIX_ATTEMPTS} — calling LLM…")
        logger.warning(msg); append_log("WARNING", msg)

        code = self._load_current_code()
        if not code:
            append_log("ERROR", "Cannot load strategy code for error fix")
            return

        perf = self.harvester.snapshot()
        perf["ft_error_context"] = {
            "error_excerpt": excerpt,
            "ft_state":      crash_or_live,
            "fix_attempt":   self._fix_attempts,
            "instruction": (
                "CRITICAL: FreqTrade is failing due to a STRATEGY CODE ERROR. "
                "Your PRIMARY task is to FIX this error so FreqTrade can run. "
                "Performance optimisation is secondary — fix the code first."
            ),
        }

        new_gen  = self._current_gen + 1
        new_code = self.improver.improve(
            code    = code,
            perf    = perf,
            gen     = new_gen,
            reason  = f"ft_error_fix_attempt_{self._fix_attempts}",
            history = self.journal.history_for_llm(),
        )

        if not new_code:
            append_log("ERROR", f"LLM fix failed (attempt {self._fix_attempts})")
            self.journal.record(EV_LLM_FAILED, self._current_gen,
                                {"reason": "ft_error_fix", "error": excerpt[:200]})
            return

        changelog = f"Auto-fix FT error (attempt {self._fix_attempts}): {excerpt[:80]}"
        self.deployer.deploy(new_code, new_gen, perf, "ft_error_fix", changelog)
        self._current_gen    = new_gen
        self._fix_attempts   = 0
        self._last_error_sig = ""

        time.sleep(5)
        if crashed:
            self.ft.start()
        else:
            ok = self.ft.api_reload()
            if not ok:
                self.ft.restart()

        self.journal.record(EV_FT_ERR_FIXED, new_gen, {
            "metrics":       perf.get("metrics", {}),
            "changelog":     changelog,
            "error_excerpt": excerpt[:200],
            "attempt":       self._fix_attempts,
        })
        write_state({
            "status":      "running",
            "current_gen": new_gen,
            "ft_error":    None,
        })
        append_log("INFO",
            f"✅ FT error fix deployed as gen {new_gen} — monitoring restart…")

    # ── Evolution ──────────────────────────────────────────────
    def _evolve(self, snap: dict, reason: str) -> None:
        if self._evolving:
            append_log("WARNING", "Evolution already in progress — skipping")
            return
        self._evolving = True
        self._last_evo = datetime.now().astimezone()
        try:
            self._do_evolve(snap, reason)
        finally:
            self._evolving = False

    def _do_evolve(self, snap: dict, reason: str) -> None:
        write_state({"status": "evolving"})

        code = self._load_current_code()
        if not code:
            msg = "Cannot read current strategy — skipping evolution"
            logger.error(msg); append_log("ERROR", msg)
            self.journal.record(EV_SKIPPED, self._current_gen, {"reason": "no_code"})
            write_state({"status": "running"})
            return

        snap["evolution_history"] = self.journal.history_for_llm()
        new_gen = self._current_gen + 1
        append_log("INFO", f"Calling LLM for generation {new_gen}...")
        write_state({"status": "evolving", "evolving_to_gen": new_gen})

        new_code = self.improver.improve(
            code    = code,
            perf    = snap,
            gen     = new_gen,
            reason  = reason,
            history = snap["evolution_history"],
        )
        if not new_code:
            msg = "LLM returned no valid code — skipping evolution"
            logger.error(msg); append_log("ERROR", msg)
            self.journal.record(EV_LLM_FAILED, self._current_gen, {"reason": reason})
            write_state({"status": "running"})
            return

        changelog = self._extract_changelog(new_code)
        deployed  = self.deployer.deploy(new_code, new_gen, snap, reason, changelog)

        if deployed:
            self._current_gen    = new_gen
            self._deploy_at      = local_str()
            self._monitoring     = True
            self._fix_attempts   = 0
            self._last_error_sig = ""
            write_state({
                "status":         "monitoring",
                "monitoring":      True,
                "current_gen":     new_gen,
                "deploy_at":       self._deploy_at,
                "last_changelog":  changelog,
                "evolving_to_gen": None,
                "ft_error":        None,
                "ft_error_count":  0,
            })
            self.journal.record(EV_DEPLOYED, new_gen, {
                "metrics":   snap.get("metrics", {}),
                "changelog": changelog,
                "trigger":   reason,
            })
            n = cfg("rollback", "evaluate_after_n_trades", default=5)
            append_log("INFO", f"Gen {new_gen} deployed. Monitoring next {n} trades...")
        else:
            append_log("WARNING", f"Deployment of gen {new_gen} failed")
            write_state({"status": "running"})

    def _check_rollback(self) -> None:
        n     = cfg("rollback", "evaluate_after_n_trades", default=5)
        post  = self.harvester.trades_since(self._deploy_at or "")
        count = post.get("count", 0)
        write_state({"monitoring_trades": count, "monitoring_target": n})

        if count < n:
            return

        append_log("INFO",
            f"Rollback evaluation: gen {self._current_gen} after {count} trades")

        if self.deployer.should_rollback(post):
            prev = self._current_gen - 1
            self.deployer.rollback(prev)
            self.journal.record(EV_ROLLED_BACK, self._current_gen, {
                "rolled_back_to": prev,
                "metrics":        post.get("metrics", {}),
                "reason":         "performance_below_threshold",
            })
            self._current_gen = prev
            write_state({"current_gen": prev})
        else:
            append_log("INFO",
                f"✅ Gen {self._current_gen} confirmed after {count} trades")
            self.journal.record(EV_CONFIRMED, self._current_gen,
                                {"metrics": post.get("metrics", {})})

        self._monitoring = False
        write_state({"monitoring": False, "monitoring_trades": 0,
                     "status": "running"})

    # ── Control commands ───────────────────────────────────────
    def _handle_control(self) -> None:
        cmd = consume_control()
        if not cmd:
            return
        action = cmd.get("action", "")
        append_log("INFO", f"Control command: {action}")

        if action == "pause":
            self._paused = True
            write_state({"paused": True, "status": "paused", "evolve_desired": "stopped"})
        elif action == "resume":
            self._paused = False
            write_state({"paused": False, "status": "running", "evolve_desired": "running"})
        elif action == "force_evolve":
            snap = self.harvester.snapshot()
            self._evolve(snap, "manual_force")
        elif action == "rollback":
            gen = cmd.get("generation", self._current_gen - 1)
            if gen >= 1:
                self.deployer.rollback(gen)
                self.journal.record(EV_ROLLED_BACK, self._current_gen,
                                    {"rolled_back_to": gen, "reason": "manual"})
                self._current_gen    = gen
                self._monitoring     = False
                self._fix_attempts   = 0
                self._last_error_sig = ""
        elif action == "reload_config":
            load_config(force=True)
            append_log("INFO", "Config reloaded from disk")
        elif action == "start_ft":
            write_state({"ft_desired": "running"})
            self.ft.start()
        elif action == "stop_ft":
            write_state({"ft_desired": "stopped"})
            self.ft.stop()
            self._fix_attempts   = 0
            self._last_error_sig = ""
            write_state({"ft_status": "stopped", "ft_error": None,
                         "ft_pid": None, "ft_running": False})
        elif action == "restart_ft":
            self.ft.restart()
        elif action == "reload_ft":
            ok = self.ft.api_reload()
            if not ok:
                self.ft.restart()

    # ── Helpers ───────────────────────────────────────────────
    def _load_current_code(self) -> Optional[str]:
        sname = cfg("freqtrade", "strategy_name", default="AutoEvolveStrategy")
        dest  = Path(cfg("freqtrade", "strategies_dir", default="")).expanduser()
        f     = dest / f"{sname}.py"
        if f.exists():
            return f.read_text(encoding="utf-8")
        code = self.checkpoints.load_code(self._current_gen)
        if code:
            return code
        local = Path("strategies") / f"{sname}.py"
        return local.read_text(encoding="utf-8") if local.exists() else None

    def _extract_changelog(self, code: str) -> str:
        for line in code.split("\n")[:60]:
            if "changelog" in line.lower():
                return line.split(":", 1)[-1].strip().strip('"').strip("'")
        return "LLM improvement"

    def _ensure_baseline(self) -> None:
        load_config(force=True)
        sname    = cfg("freqtrade", "strategy_name", default="AutoEvolveStrategy")
        dest_raw = cfg("freqtrade", "strategies_dir", default="")
        if not dest_raw or not str(dest_raw).strip():
            append_log("WARNING", "freqtrade.strategies_dir not set in config.yaml")
            return

        dest   = Path(dest_raw).expanduser().resolve()
        target = dest / f"{sname}.py"
        append_log("INFO", f"strategies_dir → {dest}")

        if target.exists():
            append_log("INFO", f"Strategy found: {target}")
            return

        dest.mkdir(parents=True, exist_ok=True)
        code = None

        code = self.checkpoints.load_code(self._current_gen)
        if code:
            append_log("INFO",
                f"Restoring gen {self._current_gen} from checkpoint → {target}")

        if not code:
            for stale in [dest.parent / f"{sname}.py",
                          dest.parent.parent / f"{sname}.py"]:
                if stale.exists():
                    code = stale.read_text(encoding="utf-8")
                    append_log("WARNING",
                        f"Copying stale strategy {stale} → {target}")
                    break

        if not code:
            # Check if user provided a seed strategy file in config
            seed_path_raw = cfg("freqtrade", "initial_strategy_file", default="")
            if seed_path_raw:
                seed_path = Path(seed_path_raw).expanduser().resolve()
                if seed_path.exists():
                    code = seed_path.read_text(encoding="utf-8")
                    # Rename class if needed so it matches strategy_name
                    import re as _re
                    code = _re.sub(r"class \w+\(", f"class {sname}(", code, count=1)
                    append_log("INFO",
                        f"Seeding gen 1 from: {seed_path}")
                else:
                    append_log("WARNING",
                        f"initial_strategy_file not found: {seed_path} — using built-in template")

        if not code:
            code = baseline_template(sname)
            append_log("INFO", "Using built-in 15m baseline template")

        self.checkpoints.save(1, code, {}, "baseline", "Initial baseline")
        self._current_gen = 1
        append_log("INFO", f"Baseline strategy written → {target}")

        target.write_text(code, encoding="utf-8")
        append_log("INFO", f"Strategy written → {target}")
