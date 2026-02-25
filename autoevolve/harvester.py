"""
autoevolve/harvester.py
=======================
Reads FreqTrade's SQLite database and computes a comprehensive
performance snapshot used by the LLM improvement engine and dashboard.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import cfg, local_str, utc_to_local_str

logger = logging.getLogger("autoevolve.harvester")


class Harvester:
    """Extracts and computes trading performance metrics from the FreqTrade SQLite DB."""

    # ── Public API ────────────────────────────────────────────
    def snapshot(self) -> dict:
        """Full performance snapshot for LLM context and dashboard."""
        trades = self._load()
        if trades.empty:
            return {
                "status": "no_trades", "generated_at": local_str(),
                "total_closed": 0, "total_open": 0,
                "consecutive_losses": 0, "metrics": {}, "patterns": {}, "recent": [],
            }
        closed = trades[trades["is_open"] == 0].copy()
        open_  = trades[trades["is_open"] == 1].copy()
        return {
            "status":             "ok",
            "generated_at":       local_str(),
            "total_closed":       len(closed),
            "total_open":         len(open_),
            "consecutive_losses": self._consecutive_losses(closed),
            "metrics":            self._metrics(closed),
            "patterns":           self._patterns(closed),
            "recent":             self._recent(closed),
        }

    def consecutive_losses(self) -> int:
        trades = self._load()
        if trades.empty:
            return 0
        return self._consecutive_losses(trades[trades["is_open"] == 0])

    def trades_since(self, deployed_at: str) -> dict:
        """Metrics for trades opened after a given local datetime string."""
        trades = self._load()
        if trades.empty:
            return {"count": 0, "metrics": {}}
        closed = trades[trades["is_open"] == 0].copy()
        if deployed_at:
            try:
                dt = pd.to_datetime(deployed_at).tz_localize(None)
                open_dt = pd.to_datetime(closed["open_date"]).dt.tz_localize(None)
                closed  = closed[open_dt >= dt]
            except Exception as e:
                logger.debug(f"Date filter error: {e}")
        return {"count": len(closed), "metrics": self._metrics(closed)}

    # ── DB ────────────────────────────────────────────────────
    def _load(self) -> pd.DataFrame:
        raw = cfg("freqtrade", "db_path", default="")
        if not raw:
            return pd.DataFrame()
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            logger.debug(f"DB not found: {p}")
            return pd.DataFrame()
        try:
            conn = sqlite3.connect(str(p))

            # Read actual schema — FT changes column names between versions
            cols_info = conn.execute("PRAGMA table_info(trades)").fetchall()
            col_names = {row[1] for row in cols_info}
            logger.debug(f"DB columns: {sorted(col_names)}")

            # Always-required columns that must exist
            required = ["id", "pair", "is_open", "open_date"]
            for c in required:
                if c not in col_names:
                    logger.warning(f"DB missing required column: {c}")
                    conn.close()
                    return pd.DataFrame()

            # Build SELECT list from only what exists
            parts = ["id", "pair", "is_open"]

            for c in ["open_date", "close_date", "open_rate", "close_rate",
                      "stake_amount", "is_short", "exit_reason", "trade_duration"]:
                if c in col_names:
                    parts.append(c)

            # profit_ratio: prefer direct column, else compute, else zero
            if "profit_ratio" in col_names:
                parts.append("profit_ratio")
            elif "profit_abs" in col_names and "stake_amount" in col_names:
                parts.append("CASE WHEN stake_amount > 0 "
                             "THEN profit_abs / stake_amount ELSE 0.0 END AS profit_ratio")
            else:
                parts.append("0.0 AS profit_ratio")

            if "profit_abs" in col_names:
                parts.append("profit_abs")
            else:
                parts.append("0.0 AS profit_abs")

            sql = f"SELECT {', '.join(parts)} FROM trades ORDER BY open_date ASC"
            df = pd.read_sql_query(sql, conn)
            conn.close()

            # Ensure optional columns exist with safe defaults
            for c, default in [("is_short", 0), ("exit_reason", ""),
                                ("trade_duration", 0), ("close_date", None),
                                ("open_rate", 0.0), ("close_rate", 0.0),
                                ("stake_amount", 0.0), ("profit_abs", 0.0)]:
                if c not in df.columns:
                    df[c] = default

            return df
        except Exception as e:
            logger.warning(f"DB read failed: {e}")
            return pd.DataFrame()

    # ── Metrics ───────────────────────────────────────────────
    def _metrics(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        p    = df["profit_ratio"].astype(float).values
        wins = p[p > 0]; loss = p[p <= 0]
        wr   = len(wins) / len(p) if len(p) else 0
        avg_w = float(np.mean(wins)) if len(wins) else 0
        avg_l = float(np.mean(loss)) if len(loss) else 0
        gp = float(np.sum(wins)); gl = abs(float(np.sum(loss)))
        pf = gp / gl if gl > 0 else float("inf")

        periods = 96 * 365  # 15m candles per year
        sharpe  = (np.mean(p) / np.std(p) * math.sqrt(periods)) \
                  if len(p) > 1 and np.std(p) > 0 else 0.0
        neg     = p[p < 0]
        sortino = (np.mean(p) / np.std(neg) * math.sqrt(periods)) \
                  if len(neg) > 1 and np.std(neg) > 0 else 0.0

        cum  = np.cumsum(p); peak = np.maximum.accumulate(cum)
        mdd  = float(np.max(peak - cum)) if len(cum) else 0.0

        exit_r: dict = {}
        if "exit_reason" in df.columns:
            exit_r = {str(k): int(v)
                      for k, v in df["exit_reason"].value_counts().items()}

        pair_stats: dict = {}
        for pair, g in df.groupby("pair"):
            pp = g["profit_ratio"].astype(float).values
            pair_stats[pair] = {
                "trades":   len(g),
                "win_rate": round(float(np.mean(pp > 0)), 3),
                "total":    round(float(np.sum(pp)), 4),
            }

        long_df  = df[df["is_short"] == 0] if "is_short" in df.columns else df
        short_df = df[df["is_short"] == 1] if "is_short" in df.columns else pd.DataFrame()

        return {
            "total":         len(p),
            "win_rate":      round(wr, 4),
            "avg_win":       round(avg_w, 4),
            "avg_loss":      round(avg_l, 4),
            "profit_factor": round(pf, 4) if pf != float("inf") else 999.0,
            "expectancy":    round((wr * avg_w) + ((1 - wr) * avg_l), 4),
            "sharpe":        round(sharpe, 4),
            "sortino":       round(sortino, 4),
            "max_drawdown":  round(mdd, 4),
            "total_pnl":     round(float(np.sum(p)), 4),
            "total_pnl_abs": round(float(np.sum(df["profit_abs"].astype(float))), 4),
            "exit_reasons":  exit_r,
            "pair_stats":    pair_stats,
            "long_wr":       round(float(np.mean(
                                 long_df["profit_ratio"].astype(float) > 0)), 3)
                             if not long_df.empty else None,
            "short_wr":      round(float(np.mean(
                                 short_df["profit_ratio"].astype(float) > 0)), 3)
                             if not short_df.empty else None,
        }

    def _patterns(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        p = df["profit_ratio"].astype(float).values
        streak = cur = 0
        for x in p:
            cur    = cur + 1 if x <= 0 else 0
            streak = max(streak, cur)
        pt = df.groupby("pair")["profit_ratio"].sum().astype(float)
        return {
            "max_loss_streak": streak,
            "best_pair":       str(pt.idxmax()) if not pt.empty else None,
            "worst_pair":      str(pt.idxmin()) if not pt.empty else None,
        }

    def _recent(self, df: pd.DataFrame, n: int = 20) -> list:
        rows = []
        for _, r in df.tail(n).iterrows():
            rows.append({
                "pair":        str(r.get("pair", "")),
                "side":        "short" if r.get("is_short", 0) == 1 else "long",
                "open":        utc_to_local_str(str(r.get("open_date", ""))),
                "close":       utc_to_local_str(str(r.get("close_date", ""))),
                "profit_pct":  round(float(r.get("profit_ratio", 0)) * 100, 2),
                "profit_abs":  round(float(r.get("profit_abs", 0)), 4),
                "exit_reason": str(r.get("exit_reason", "")),
            })
        return rows

    def _consecutive_losses(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        p = df["profit_ratio"].astype(float).values[::-1]
        c = 0
        for x in p:
            if x <= 0:
                c += 1
            else:
                break
        return c
