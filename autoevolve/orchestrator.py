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

        self._paused = (evolve_desired != "running")

        # Check if FT is actually alive right now (before clearing stale PID)
        ft_actually_running = self.ft.is_running()

        # Now clear stale pid/status from state (does NOT touch ft_desired)
        write_state({
            "current_gen":    self._current_gen,
            "status":         "running" if not self._paused else "paused",
            "paused":         self._paused,
            "monitoring":     False,
            "ft_running":     ft_actually_running,
            "ft_pid":         None,           # cleared until FT.start() sets it
            "ft_status":      "stopped" if not ft_actually_running else "running",
            "ft_started_at":  None,           # cleared — stale uptime from prev session
            "ft_error":       None,
            "ft_error_full":  None,
            "ft_error_count": 0,
            "ft_desired":     ft_desired,
            "evolve_desired": evolve_desired,
            "orch_started":   local_str(),
            "orch_alive":     local_str(),
        })

        if ft_desired == "running" and not ft_actually_running:
            append_log("INFO", "ft_desired=running — auto-starting FreqTrade on restart")
            self.ft.start()
        elif ft_desired == "running" and ft_actually_running:
            append_log("INFO", "ft_desired=running — FreqTrade already running, not restarting")

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
            write_state({"orch_alive": local_str()})  # heartbeat — written even during LLM
            time.sleep(poll)

    # ── Single tick (called by main.py every poll seconds) ────
    def _tick(self) -> None:
        load_config()
        write_state({"orch_alive": local_str()})
        self._handle_control()

        ft_running = self.ft.is_running()
        write_state({"ft_running": ft_running})

        # ── Always refresh metrics regardless of state ───────────
        # (paused and monitoring both previously skipped this, leaving dashboard stale)
        _snap_always = self.harvester.snapshot()
        write_state({
            "total_trades":  _snap_always.get("total_closed", 0),
            "open_trades":      _snap_always.get("total_open", 0),
            "open_trades_list": _snap_always.get("open_list", []),
            "metrics":       _snap_always.get("metrics", {}),
            "recent_trades": _snap_always.get("recent", []),
            "snapshot_at":   local_str(),
        })

        if self._paused:
            write_state({"paused": True, "status": "paused", "evolve_desired": "stopped"})
            return

        write_state({"paused": False})

        # ── Rollback monitoring window ─────────────────────────
        if self._monitoring:
            self._check_rollback()
            return

        # ── Evolution trigger ──────────────────────────────────
        threshold  = cfg("trigger", "consecutive_losses",        default=3)
        min_t      = cfg("trigger", "min_trades_before_trigger", default=10)
        cooldown   = cfg("trigger", "cooldown_minutes",          default=60)
        dd_pct     = cfg("trigger", "profit_drawdown_pct",        default=0)
        dd_min     = cfg("trigger", "profit_drawdown_min_trades", default=5)
        idle_min   = cfg("trigger", "idle_trigger_minutes",       default=0)

        snap   = _snap_always
        losses = snap.get("consecutive_losses", 0)
        total  = snap.get("total_closed", 0)

        # Compute countdown to next possible evolution
        cooldown_remaining = 0
        if self._last_evo:
            elapsed_min = (datetime.now().astimezone() - self._last_evo).total_seconds() / 60
            cooldown_remaining = max(0, round(cooldown - elapsed_min, 1))

        # FT uptime
        ft_pid = read_state().get("ft_pid")
        ft_started = read_state().get("ft_started_at", "")

        # Full trigger status — everything the dashboard needs to show
        write_state({
            "status":               "running",
            "current_gen":          self._current_gen,
            "consecutive_losses":   losses,
            "trigger_threshold":    threshold,
            "total_trades":         total,
            "metrics":              snap.get("metrics", {}),
            "snapshot_at":          local_str(),
            "cooldown_remaining":   cooldown_remaining,
            "last_evo_at":          self._last_evo.isoformat() if self._last_evo else None,
            # Trigger context for dashboard
            "trigger_status": {
                "cooldown_remaining_min": cooldown_remaining,
                "cooldown_total_min":     cooldown,
                "loss_current":           losses,
                "loss_threshold":         threshold,
                "trades_total":           total,
                "trades_needed":          max(0, min_t - total),
                "trades_min":             min_t,
                "in_cooldown":            cooldown_remaining > 0,
                "loss_trigger_ready":     total >= min_t and cooldown_remaining == 0,
                "dd_threshold_pct":       dd_pct,
                "dd_min_trades":          dd_min,
                "idle_threshold_min":     idle_min if idle_min > 0 else None,
            },
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

        # ── Track last trade entry time (from DB, not open count) ─
        # Idle trigger should fire when no NEW trade was opened — having 3
        # open trades doesn't mean the strategy is actively entering positions.
        last_entry_str = snap.get("last_entry_date", "")
        if last_entry_str:
            try:
                last_entry_dt = datetime.fromisoformat(last_entry_str).astimezone()
                if last_entry_dt > self._last_open_trade:
                    self._last_open_trade = last_entry_dt
            except Exception:
                pass

        # ── Idle trigger: no new trade entry for X minutes ──────
        idle_minutes = idle_min
        if idle_minutes > 0 and ft_running:
            idle_elapsed = (
                datetime.now().astimezone() - self._last_open_trade
            ).total_seconds() / 60
            write_state({"idle_minutes": round(idle_elapsed, 1),
                         "idle_trigger_minutes": idle_minutes})
            write_state({"trigger_status": {
                **read_state().get("trigger_status", {}),
                "idle_elapsed_min":  round(idle_elapsed, 1),
                "idle_threshold_min": idle_minutes,
                "idle_trigger_ready": idle_elapsed >= idle_minutes and cooldown_remaining == 0,
            }})
            if idle_elapsed >= idle_minutes and not _in_cooldown():
                msg = (f"IDLE TRIGGER: no new trade entry for "
                       f"{idle_elapsed:.1f} min (threshold={idle_minutes}) "
                       f"-> evolving gen {self._current_gen} -> {self._current_gen + 1}")
                logger.warning(msg); append_log("WARNING", msg)
                self.journal.record(EV_TRIGGERED, self._current_gen,
                                    {"reason": "idle", "idle_minutes": idle_elapsed,
                                     "metrics": snap.get("metrics", {})})
                self._evolve(snap, "idle_no_open_trades")
                return

        # Write idle=disabled state when not configured
        if not idle_minutes:
            write_state({"trigger_status": {
                **read_state().get("trigger_status", {}),
                "idle_elapsed_min":   None,
                "idle_threshold_min": None,
                "idle_trigger_ready": False,
            }})

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
            return

        # ── Profit drawdown trigger ──────────────────────────────
        if dd_pct > 0 and total >= dd_min:
            metrics          = snap.get("metrics", {})
            pnl_drawdown_pct = metrics.get("pnl_drawdown_pct", 0.0)
            peak_pnl         = metrics.get("peak_pnl",         0.0)
            current_pnl      = metrics.get("current_pnl",      0.0)

            write_state({"trigger_status": {
                **read_state().get("trigger_status", {}),
                "dd_current_pct":  round(pnl_drawdown_pct, 2),
                "dd_threshold_pct": dd_pct,
                "dd_peak_pnl":     round(peak_pnl, 4),
                "dd_current_pnl":  round(current_pnl, 4),
                "dd_trigger_ready": pnl_drawdown_pct >= dd_pct,
            }})

            if pnl_drawdown_pct >= dd_pct:
                reason = f"profit_drawdown_{pnl_drawdown_pct:.1f}pct_from_peak"
                msg = (f"TRIGGER: profit drawdown {pnl_drawdown_pct:.1f}% from peak "
                       f"(peak={peak_pnl:.4f} current={current_pnl:.4f} threshold={dd_pct}%) "
                       f"-> evolving gen {self._current_gen} -> {self._current_gen + 1}")
                logger.warning(msg); append_log("WARNING", msg)
                self.journal.record(EV_TRIGGERED, self._current_gen,
                                    {"reason": reason,
                                     "pnl_drawdown_pct": pnl_drawdown_pct,
                                     "peak_pnl": peak_pnl,
                                     "current_pnl": current_pnl,
                                     "metrics": metrics})
                self._evolve(snap, reason)
                return
        else:
            write_state({"trigger_status": {
                **read_state().get("trigger_status", {}),
                "dd_current_pct":   None,
                "dd_threshold_pct": dd_pct,
                "dd_trigger_ready": False,
            }})

    # ── Strategy error handler (called from ft_monitor.py too) ─
    def _handle_strategy_error(self, excerpt: str, crashed: bool = False) -> None:
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
                write_state({"ft_status": "error_max_retries", "ft_error": excerpt[:300], "ft_error_full": excerpt})
                return
        else:
            self._fix_attempts   = 0
            self._last_error_sig = sig

        self._fix_attempts += 1
        write_state({
            "status":         "fixing_ft_error",
            "ft_error":       excerpt[:300],      # short version for pill
            "ft_error_full":  excerpt,             # full for error panel
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

        # Fix attempts overwrite the SAME broken generation — gen number only
        # advances once when the evolution first deployed. Fixes are patches on
        # that gen, not new generations, so the counter doesn't burn numbers.
        broken_gen = self._current_gen   # the gen that has the error
        new_code = self.improver.improve(
            code    = code,
            perf    = perf,
            gen     = broken_gen,        # same gen, just fixing it
            reason  = f"ft_error_fix_attempt_{self._fix_attempts}",
            history = self.journal.history_for_llm(),
        )

        if not new_code:
            append_log("ERROR", f"LLM fix failed (attempt {self._fix_attempts})")
            self.journal.record(EV_LLM_FAILED, self._current_gen,
                                {"reason": "ft_error_fix", "error": excerpt[:200]})
            return

        # Strip backslashes from excerpt used in changelog — Windows paths
        # (e.g. D:/Python/freqtrade) in re.sub replacement strings cause re.error
        safe_excerpt = excerpt[:120].replace("\\", "/").replace("\r", "").replace("\n", " ")
        changelog = f"Auto-fix gen {broken_gen} (attempt {self._fix_attempts}): {safe_excerpt}"
        self.deployer.deploy(new_code, broken_gen, perf, "ft_error_fix", changelog)
        # _current_gen stays the same — we fixed the broken gen, not created a new one
        self._fix_attempts   = 0
        self._last_error_sig = ""

        time.sleep(5)
        if crashed:
            self.ft.start()
        else:
            ok = self.ft.api_reload()
            if not ok:
                self.ft.restart()

        self.journal.record(EV_FT_ERR_FIXED, broken_gen, {
            "metrics":       perf.get("metrics", {}),
            "changelog":     changelog,
            "error_excerpt": excerpt[:200],
            "attempt":       self._fix_attempts,
        })
        # Advance the log scan offset so the next health check won't
        # re-detect the error we just fixed from the same log file
        new_log_offset = 0
        try:
            from .utils import ft_log_path as _flp
            lf = _flp()
            if lf.exists():
                new_log_offset = lf.stat().st_size
        except Exception:
            pass

        write_state({
            "status":            "running",
            "current_gen":       broken_gen,
            "ft_error":          None,
            "ft_error_full":     None,
            "ft_error_count":    0,
            "ft_log_scan_from":  new_log_offset,
        })
        append_log("INFO",
            f"✅ FT error fix deployed gen {broken_gen} — resuming…")

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
            # Reset DB so fresh gen starts with clean trade history
            self.ft.reset_database()
            # Reset the session timestamp so consecutive_losses only counts
            # trades closed after the DB wipe — not the now-deleted history
            write_state({"ft_started_at": local_str()})
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
                "ft_error_full":   None,
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


        # Monitoring timeout: if no trades for too long, exit monitoring and re-enable idle trigger
        monitor_timeout = cfg("rollback", "monitoring_timeout_minutes", default=60)
        if self._deploy_at and monitor_timeout > 0:
            from datetime import datetime
            try:
                from .utils import local_now
                deploy_dt = datetime.fromisoformat(self._deploy_at.replace(" +", "+").replace(" -", "-"))
                elapsed_min = (local_now() - deploy_dt).total_seconds() / 60
            except Exception:
                elapsed_min = 0
            if elapsed_min >= monitor_timeout and count == 0:
                append_log("WARNING",
                    f"Monitoring timeout ({monitor_timeout}min) with 0 trades — "
                    f"exiting monitoring, resetting idle timer")
                self._monitoring = False
                self._last_open_trade = local_now()  # reset idle clock
                write_state({"monitoring": False, "monitoring_trades": 0, "status": "running"})
                return

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
        self._last_open_trade = datetime.now().astimezone()  # reset idle clock after monitoring
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
            # Immediately push updated trigger values to state so dashboard
            # reflects new config without waiting for next tick (up to 60s)
            _threshold  = cfg("trigger", "consecutive_losses",        default=3)
            _min_t      = cfg("trigger", "min_trades_before_trigger", default=10)
            _cooldown   = cfg("trigger", "cooldown_minutes",          default=60)
            _idle       = cfg("trigger", "idle_trigger_minutes",      default=0)
            _mon_target   = cfg("rollback", "trades_to_confirm",        default=5)
            _dd_pct       = cfg("trigger", "profit_drawdown_pct",        default=0)
            _dd_min       = cfg("trigger", "profit_drawdown_min_trades", default=5)
            _cd_rem = 0
            if self._last_evo:
                elapsed = (datetime.now().astimezone() - self._last_evo).total_seconds() / 60
                _cd_rem = max(0, round(_cooldown - elapsed, 1))
            write_state({
                "trigger_threshold": _threshold,
                "trigger_status": {
                    **read_state().get("trigger_status", {}),
                    "cooldown_remaining_min": _cd_rem,
                    "cooldown_total_min":     _cooldown,
                    "loss_threshold":         _threshold,
                    "trades_min":             _min_t,
                    "idle_threshold_min":     _idle if _idle > 0 else None,
                    "loss_trigger_ready":     False,
                    "dd_threshold_pct":       _dd_pct,
                },
                "monitoring_target": _mon_target,
            })
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
