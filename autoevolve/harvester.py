"""
autoevolve/harvester.py
=======================
Reads FreqTrade's SQLite database and produces performance snapshots
for the orchestrator and for the LLM context.

Database facts (verified against actual FT 2024+ schema):
  close_profit        FLOAT   — profit ratio (decimal, e.g. 0.0054 = 0.54%)
  close_profit_abs    FLOAT   — profit in stake currency (USDT)
  realized_profit     FLOAT   — same as close_profit_abs for fully-closed trades;
                                tracks running profit including partial exits
  is_short            BOOLEAN — 1 for short, 0 for long
  leverage            FLOAT   — leverage multiplier (1.0 = no leverage)
  enter_tag           VARCHAR — entry signal label from strategy
  exit_reason         VARCHAR — why trade was closed
  funding_fees        FLOAT   — cumulative funding fees paid/received (futures)
  open_trade_value    FLOAT   — notional USDT value at open
  max_rate / min_rate FLOAT   — high/low price during trade lifetime
  stop_loss_pct       FLOAT   — current stop loss % (negative, e.g. -0.011)
  initial_stop_loss_pct FLOAT — initial stop loss %
  is_stop_loss_trailing BOOLEAN
  trading_mode        VARCHAR — 'FUTURES' or 'SPOT'
  open_date / close_date DATETIME — used to compute trade_duration_min

NOTE: there is NO profit_ratio or profit_abs column in modern FT.
      Old FT schemas (pre-2022) used those names — we handle both.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import cfg, append_log

logger = logging.getLogger("autoevolve.harvester")


# ── Schema version detection ──────────────────────────────────────────────────

def _detect_schema(col_names: set) -> dict:
    """
    Return SQL expressions aliased to canonical names,
    based on which columns are present.
    Returns dict: canonical_name -> sql_expression
    """
    exprs = {}

    # ── profit ratio ──────────────────────────────────────────────
    # Modern FT (2022+): close_profit
    # Old FT (pre-2022): profit_ratio
    if "close_profit" in col_names:
        exprs["profit_ratio"] = "close_profit"
    elif "profit_ratio" in col_names:
        exprs["profit_ratio"] = "profit_ratio"
    elif "realized_profit" in col_names and "stake_amount" in col_names:
        # last resort: derive ratio from abs / stake
        exprs["profit_ratio"] = (
            "CASE WHEN stake_amount > 0 "
            "THEN realized_profit / stake_amount ELSE 0.0 END"
        )
    else:
        exprs["profit_ratio"] = "0.0"

    # ── profit absolute (stake currency) ─────────────────────────
    # Modern FT: close_profit_abs  (= realized_profit for fully closed)
    if "close_profit_abs" in col_names:
        exprs["profit_abs"] = "close_profit_abs"
    elif "profit_abs" in col_names:
        exprs["profit_abs"] = "profit_abs"
    elif "realized_profit" in col_names:
        exprs["profit_abs"] = "realized_profit"
    else:
        exprs["profit_abs"] = "0.0"

    # ── funding fees (futures only) ───────────────────────────────
    if "funding_fees" in col_names:
        exprs["funding_fees"] = "COALESCE(funding_fees, 0.0)"
    else:
        exprs["funding_fees"] = "0.0"

    # ── trade direction ───────────────────────────────────────────
    if "is_short" in col_names:
        exprs["is_short"] = "COALESCE(is_short, 0)"
    else:
        exprs["is_short"] = "0"

    # ── leverage ──────────────────────────────────────────────────
    if "leverage" in col_names:
        exprs["leverage"] = "COALESCE(leverage, 1.0)"
    else:
        exprs["leverage"] = "1.0"

    # ── entry / exit labels ───────────────────────────────────────
    exprs["enter_tag"]   = "COALESCE(enter_tag, '')"   if "enter_tag"   in col_names else "''"
    exprs["exit_reason"] = "COALESCE(exit_reason, '')" if "exit_reason" in col_names else "''"

    # ── stop loss ─────────────────────────────────────────────────
    exprs["stop_loss_pct"]         = "COALESCE(stop_loss_pct, 0.0)"         if "stop_loss_pct"         in col_names else "0.0"
    exprs["initial_stop_loss_pct"] = "COALESCE(initial_stop_loss_pct, 0.0)" if "initial_stop_loss_pct" in col_names else "0.0"
    exprs["is_stop_loss_trailing"] = "COALESCE(is_stop_loss_trailing, 0)"   if "is_stop_loss_trailing" in col_names else "0"

    # ── price extremes during trade ───────────────────────────────
    exprs["max_rate"] = "COALESCE(max_rate, close_rate, 0.0)" if "max_rate" in col_names else "0.0"
    exprs["min_rate"] = "COALESCE(min_rate, close_rate, 0.0)" if "min_rate" in col_names else "0.0"

    # ── notional value ────────────────────────────────────────────
    exprs["open_trade_value"] = "COALESCE(open_trade_value, stake_amount)" if "open_trade_value" in col_names else "stake_amount"

    return exprs


# ── SQL builder ───────────────────────────────────────────────────────────────

def _build_sql(col_names: set, where: str = "") -> str:
    """Build SELECT with only columns that exist, using canonical aliases."""
    exprs = _detect_schema(col_names)

    # Always-present columns (all NOT NULL in FT schema)
    parts = [
        "id",
        "pair",
        "is_open",
        "open_rate",
        "COALESCE(close_rate, 0.0) AS close_rate",
        "stake_amount",
        "amount",
        "open_date",
        "COALESCE(close_date, '') AS close_date",
        f"{exprs['profit_ratio']}     AS profit_ratio",
        f"{exprs['profit_abs']}       AS profit_abs",
        f"{exprs['is_short']}         AS is_short",
        f"{exprs['leverage']}         AS leverage",
        f"{exprs['enter_tag']}        AS enter_tag",
        f"{exprs['exit_reason']}      AS exit_reason",
        f"{exprs['funding_fees']}     AS funding_fees",
        f"{exprs['stop_loss_pct']}    AS stop_loss_pct",
        f"{exprs['initial_stop_loss_pct']} AS initial_stop_loss_pct",
        f"{exprs['is_stop_loss_trailing']} AS is_stop_loss_trailing",
        f"{exprs['max_rate']}         AS max_rate",
        f"{exprs['min_rate']}         AS min_rate",
        f"{exprs['open_trade_value']} AS open_trade_value",
    ]
    # Optional but useful
    for col in ("trading_mode", "base_currency", "stake_currency", "strategy"):
        if col in col_names:
            parts.append(f"COALESCE({col}, '') AS {col}")

    where_clause = f"WHERE {where}" if where else ""
    return f"SELECT {', '.join(parts)} FROM trades {where_clause} ORDER BY COALESCE(close_date, open_date) ASC"


# ── Core loader ───────────────────────────────────────────────────────────────

class Harvester:

    def _open_db(self) -> Optional[tuple]:
        """
        Open the SQLite DB. Returns (conn, col_names set) or None on failure.
        Uses WAL mode for safe reads while FT is writing.
        """
        raw = cfg("freqtrade", "db_path", default="")
        if not raw:
            append_log("WARNING", "harvester: db_path not set in config.yaml")
            return None

        p = Path(raw).expanduser().resolve()
        if not p.exists():
            # DB not yet created (fresh FT instance) — silently return None.
            # Only warn+auto-detect if config is wrong (path set but wrong location).
            # We distinguish: if raw path was explicitly set but missing, try to help.
            # If no path was configured at all, just wait silently.
            if not raw:
                return None   # no db_path configured, nothing to do yet

            # Path was configured but doesn't exist yet — could be fresh FT startup.
            # Auto-detect quietly and only warn once per unique path.
            from .utils import BASE_DIR
            candidates = sorted(
                [c for c in (list(BASE_DIR.rglob("*.sqlite")) +
                              list(BASE_DIR.parent.rglob("*.sqlite")))
                 if any(x in c.name.lower() for x in ("dryrun", "trade", "freqtrade"))
                 and ".autoevolve" not in str(c)],
                key=lambda c: c.stat().st_mtime, reverse=True
            )
            if candidates:
                p = candidates[0]
                # Only log if the auto-detected path differs from configured —
                # i.e. the config is actually wrong, not just "DB not yet created"
                if p != Path(raw).expanduser().resolve():
                    append_log("WARNING",
                        f"harvester: db_path '{raw}' not found — "
                        f"auto-detected: {p}. "
                        f"Set freqtrade.db_path in config.yaml to fix.")
            else:
                # DB simply doesn't exist yet (FT not started or fresh instance)
                return None

        try:
            # Plain connect — safe for concurrent reads with FT writing in WAL mode
            conn = sqlite3.connect(str(p), check_same_thread=False, timeout=10)
            conn.execute("PRAGMA query_only = ON")   # read-only guard

            cols_info = conn.execute("PRAGMA table_info(trades)").fetchall()
            col_names = {row[1] for row in cols_info}
            if not col_names:
                conn.close()
                append_log("WARNING", f"harvester: trades table empty or missing in {p}")
                return None
            return conn, col_names
        except Exception as e:
            append_log("ERROR", f"harvester: cannot open DB {p}: {e}")
            return None

    def _load(self, where: str = "") -> pd.DataFrame:
        """Load trades from DB into a typed DataFrame."""
        result = self._open_db()
        if result is None:
            return pd.DataFrame()
        conn, col_names = result
        try:
            sql = _build_sql(col_names, where)
            df  = pd.read_sql_query(sql, conn)
        except Exception as e:
            append_log("ERROR", f"harvester: query error: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()

        if df.empty:
            return df

        # ── Type enforcement ────────────────────────────────────────
        float_cols = ["profit_ratio", "profit_abs", "leverage", "funding_fees",
                      "open_rate", "close_rate", "stake_amount", "amount",
                      "stop_loss_pct", "initial_stop_loss_pct",
                      "max_rate", "min_rate", "open_trade_value"]
        for c in float_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        bool_cols = ["is_open", "is_short", "is_stop_loss_trailing"]
        for c in bool_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(int)

        str_cols = ["pair", "enter_tag", "exit_reason"]
        for c in str_cols:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)

        # ── Derived columns ─────────────────────────────────────────
        # trade_duration_min: computed from open/close dates
        if "open_date" in df.columns and "close_date" in df.columns:
            try:
                od = pd.to_datetime(df["open_date"], errors="coerce", utc=True)
                cd = pd.to_datetime(df["close_date"], errors="coerce", utc=True)
                df["trade_duration_min"] = ((cd - od)
                    .dt.total_seconds()
                    .div(60)
                    .fillna(0)
                    .clip(lower=0)
                    .round(1))
            except Exception:
                df["trade_duration_min"] = 0.0

        return df

    # ── Public API ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full performance snapshot: closed trades + open count."""
        df = self._load()
        if df.empty:
            return self._empty_snapshot()

        closed = df[df["is_open"] == 0].copy()
        open_  = df[df["is_open"] == 1]

        # Most recent trade entry time (open or closed) — used by idle trigger
        # to detect when the strategy last opened a position
        last_entry_date = ""
        if "open_date" in df.columns and not df.empty:
            try:
                last_entry_date = str(df["open_date"].max())[:19]
            except Exception:
                pass

        metrics = self._compute_metrics(closed)

        # Recent trades (last 20 closed) for LLM context
        recent = []
        for _, r in closed.tail(20).iterrows():
            close_dt = str(r.get("close_date", ""))
            # Format to readable short form: "Feb 27 18:04"
            try:
                from datetime import datetime as _dt
                close_dt_fmt = _dt.fromisoformat(close_dt[:19]).strftime("%b %d %H:%M")
            except Exception:
                close_dt_fmt = close_dt[:16]

            recent.append({
                "pair":         r.get("pair", ""),
                "is_short":     bool(r.get("is_short", 0)),
                "leverage":     round(float(r.get("leverage", 1.0)), 1),
                "profit_pct":   round(float(r.get("profit_ratio", 0)) * 100, 3),
                "profit_abs":   round(float(r.get("profit_abs", 0)),  4),
                "funding_fees": round(float(r.get("funding_fees", 0)), 4),
                "enter_tag":    str(r.get("enter_tag", "")),
                "exit_reason":  str(r.get("exit_reason", "")),
                "duration_min": round(float(r.get("trade_duration_min", 0)), 1),
                "stop_loss_pct":round(float(r.get("stop_loss_pct", 0)) * 100, 2),
                "close_date":   close_dt_fmt,
            })

        return {
            "total_closed": int(len(closed)),
            "total_open":   int(len(open_)),
            "consecutive_losses": self._consecutive_losses(closed),
            "last_entry_date":    last_entry_date,
            "metrics":      metrics,
            "recent":       recent,
        }

    def trades_since(self, since_dt_str: str) -> dict:
        """Trades closed after a given datetime — used by rollback evaluator."""
        if not since_dt_str:
            return self.snapshot()
        try:
            df = self._load(where=f"close_date > '{since_dt_str}' AND is_open = 0")
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            return {"count": 0, "metrics": {}}
        metrics = self._compute_metrics(df)
        return {"count": int(len(df)), "metrics": metrics}

    def open_trade_count(self) -> int:
        """Fast count of currently open trades."""
        result = self._open_db()
        if result is None:
            return 0
        conn, _ = result
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE is_open = 1"
            ).fetchone()[0]
        except Exception:
            n = 0
        finally:
            conn.close()
        return int(n)

    # ── Metrics engine ────────────────────────────────────────────────────────

    def _compute_metrics(self, df: pd.DataFrame) -> dict:
        """Compute all strategy performance metrics from a closed-trades DataFrame."""
        if df.empty:
            return {}

        p   = df["profit_ratio"].astype(float).values
        abs_  = df["profit_abs"].astype(float).values

        wins  = p[p >  0]
        loss  = p[p <= 0]
        n     = len(p)

        # ── Core metrics ──────────────────────────────────────────
        win_rate      = round(len(wins) / n, 4) if n else 0.0
        gross_profit  = float(np.sum(wins))
        gross_loss    = float(abs(np.sum(loss)))
        profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else 999.0

        # ── Risk-adjusted metrics ─────────────────────────────────
        # Sharpe: annualised using 15m candles (96 candles/day × 365 days)
        periods_per_year = 96 * 365
        mean_p = float(np.mean(p)) if n else 0.0
        std_p  = float(np.std(p, ddof=1)) if n > 1 else 0.0
        sharpe  = round((mean_p / std_p) * math.sqrt(periods_per_year), 4) if std_p > 0 else 0.0

        # Sortino: only downside deviation
        neg_p   = p[p < 0]
        down_std = float(np.std(neg_p, ddof=1)) if len(neg_p) > 1 else std_p
        sortino  = round((mean_p / down_std) * math.sqrt(periods_per_year), 4) if down_std > 0 else 0.0

        # Max drawdown (on cumulative profit_ratio)
        cum  = np.cumsum(p)
        peak = np.maximum.accumulate(cum)
        mdd  = round(float(np.max(peak - cum)), 6) if n > 1 else 0.0

        # Expectancy
        avg_win  = round(float(np.mean(wins)), 6) if len(wins) else 0.0
        avg_loss = round(float(np.mean(loss)), 6) if len(loss) else 0.0
        expectancy = round(win_rate * avg_win + (1 - win_rate) * avg_loss, 6)

        # ── Total PnL (USDT) ──────────────────────────────────────
        total_pnl = round(float(np.sum(abs_)), 4)

        # ── Exit reason breakdown ─────────────────────────────────
        exit_reasons: dict = {}
        if "exit_reason" in df.columns:
            for reason, grp in df.groupby("exit_reason"):
                grp_p = grp["profit_ratio"].astype(float).values
                exit_reasons[str(reason)] = {
                    "count":    int(len(grp_p)),
                    "win_rate": round(float(np.mean(grp_p > 0)), 3),
                    "avg_pct":  round(float(np.mean(grp_p)) * 100, 3),
                }

        # ── Direction split ───────────────────────────────────────
        long_wr  = short_wr = None
        if "is_short" in df.columns:
            longs  = df[df["is_short"] == 0]["profit_ratio"].astype(float).values
            shorts = df[df["is_short"] == 1]["profit_ratio"].astype(float).values
            long_wr  = round(float(np.mean(longs  > 0)), 3) if len(longs)  else None
            short_wr = round(float(np.mean(shorts > 0)), 3) if len(shorts) else None

        # ── Entry tag breakdown ───────────────────────────────────
        enter_tags: dict = {}
        if "enter_tag" in df.columns:
            for tag, grp in df.groupby("enter_tag"):
                if not tag:
                    continue
                grp_p = grp["profit_ratio"].astype(float).values
                enter_tags[str(tag)] = {
                    "count":    int(len(grp_p)),
                    "win_rate": round(float(np.mean(grp_p > 0)), 3),
                    "avg_pct":  round(float(np.mean(grp_p)) * 100, 3),
                }

        # ── Per-pair breakdown ────────────────────────────────────
        pair_stats: dict = {}
        if "pair" in df.columns:
            for pair, grp in df.groupby("pair"):
                grp_p = grp["profit_ratio"].astype(float).values
                pair_stats[str(pair)] = {
                    "count":    int(len(grp_p)),
                    "win_rate": round(float(np.mean(grp_p > 0)), 3),
                    "total_pct":round(float(np.sum(grp_p))  * 100, 3),
                }

        # ── Duration stats ────────────────────────────────────────
        dur_stats: dict = {}
        if "trade_duration_min" in df.columns:
            dur = df["trade_duration_min"].astype(float)
            dur_stats = {
                "avg_min":    round(float(dur.mean()), 1),
                "median_min": round(float(dur.median()), 1),
                "max_min":    round(float(dur.max()), 1),
            }

        # ── Funding fee impact (futures) ──────────────────────────
        funding_total = 0.0
        if "funding_fees" in df.columns:
            funding_total = round(float(df["funding_fees"].astype(float).sum()), 4)

        # ── Stop-loss analysis ────────────────────────────────────
        sl_hit_count = 0
        if "exit_reason" in df.columns:
            sl_hit_count = int(df["exit_reason"].str.contains(
                "stop_loss|stoploss|trailing", case=False, na=False
            ).sum())

        # ── Profit drawdown from peak ─────────────────────────────
        # Cumulative PnL curve, find peak, measure current drop from it
        cum_pnl    = np.cumsum(abs_)
        peak_pnl   = float(np.max(cum_pnl)) if len(cum_pnl) else 0.0
        current_pnl = float(cum_pnl[-1]) if len(cum_pnl) else 0.0
        # Drawdown as % of peak (only meaningful when peak > 0)
        if peak_pnl > 0:
            pnl_drawdown_pct = round((peak_pnl - current_pnl) / peak_pnl * 100, 2)
        else:
            pnl_drawdown_pct = 0.0

        return {
            "total":              n,
            "win_rate":           win_rate,
            "peak_pnl":           round(peak_pnl, 4),
            "current_pnl":        round(current_pnl, 4),
            "pnl_drawdown_pct":   pnl_drawdown_pct,
            "profit_factor":  profit_factor,
            "sharpe":         sharpe,
            "sortino":        sortino,
            "max_drawdown":   mdd,
            "total_pnl":      total_pnl,
            "expectancy":     expectancy,
            "avg_win":        avg_win,
            "avg_loss":       avg_loss,
            "gross_profit":   round(gross_profit, 4),
            "gross_loss":     round(-gross_loss, 4),
            "long_wr":        long_wr,
            "short_wr":       short_wr,
            "exit_reasons":   exit_reasons,
            "enter_tags":     enter_tags,
            "pair_stats":     pair_stats,
            "duration":       dur_stats,
            "funding_total":  funding_total,
            "sl_hit_count":   sl_hit_count,
        }

    def _consecutive_losses(self, df: pd.DataFrame) -> int:
        """Count trailing consecutive losing trades."""
        if df.empty:
            return 0
        profits = df["profit_ratio"].astype(float).values[::-1]
        count = 0
        for p in profits:
            if p <= 0:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _empty_snapshot() -> dict:
        return {
            "total_closed": 0, "total_open": 0,
            "consecutive_losses": 0,
            "metrics": {}, "recent": [],
        }
