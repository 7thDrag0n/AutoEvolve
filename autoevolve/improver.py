"""
autoevolve/improver.py
======================
Two-step LLM improvement engine:
  Step 1 (ANALYSE): LLM diagnoses the strategy weakness in plain English
                    and picks ONE concept to try this generation.
  Step 2 (CODE):    LLM implements the chosen concept, guided by step 1.

Config keys (all under llm:):
  provider          — anthropic / openai / openrouter / chutes / ollama
  timeout           — HTTP timeout seconds (default 600)
  exploration_mode  — conservative / adaptive / aggressive
  concepts_to_try   — list of technique names to rotate through
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from .utils import cfg, append_log, read_state, local_str, BASE_DIR

logger = logging.getLogger("autoevolve.improver")
IMPROVER_VERSION = "v3-two-step"


def _llm_log(direction: str, provider: str, params: dict, content: str) -> None:
    """
    Write a single entry to logs/llm_conversations.log (rotating, max 10MB x 5).
    direction: "REQUEST" or "RESPONSE"
    params: dict of provider/model/temperature/etc
    content: the full prompt or full response text
    """
    import logging.handlers as _lh
    _log = logging.getLogger("autoevolve.llm_conversations")
    if not _log.handlers:
        log_dir = BASE_DIR / cfg("logging", "directory", default="logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        _h = _lh.RotatingFileHandler(
            log_dir / "llm_conversations.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        _h.setFormatter(logging.Formatter("%(message)s"))
        _log.addHandler(_h)
        _log.setLevel(logging.DEBUG)
        _log.propagate = False   # don't pollute main log

    sep = "=" * 80
    meta_lines = [f"  {k}: {v}" for k, v in params.items() if v is not None]
    entry = (
        f"\n{sep}\n"
        f"{local_str()} | {direction} | provider={provider}\n"
        + "\n".join(meta_lines) + "\n"
        f"{sep}\n"
        f"{content}\n"
    )
    _log.debug(entry)

# ══════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════

_ANALYSE_SYSTEM = """\
You are a senior quantitative trading analyst reviewing a FreqTrade crypto futures strategy.
Your job is to diagnose what is WRONG and choose the single best improvement for this generation.

THINK STEP BY STEP:
1. Identify the PRIMARY metric failure (worst vs target).
2. Find the likely ROOT CAUSE by looking at exit_reasons, pair_stats, patterns.
3. Select ONE concept from the APPROVED TECHNIQUES list below that would address the root cause.
4. Explain in 3-5 sentences: what is broken, why, and exactly what to change.
5. Assess risk: what could go wrong with this change.

APPROVED TECHNIQUES (choose ONE):
{concepts}

OUTPUT FORMAT — respond with ONLY this JSON (no prose, no markdown):
{{
  "primary_weakness": "...",
  "root_cause": "...",
  "chosen_concept": "exact name from list above",
  "implementation_plan": "concrete description of what code to add/change",
  "expected_impact": "which metric improves and why",
  "risk": "what could go wrong"
}}
"""

_ANALYSE_USER = """\
CURRENT METRICS:
{perf}

EVOLUTION HISTORY (what was tried before and whether it helped):
{history}

EXPLORATION MODE: {mode}
{mode_guidance}
"""

_CODE_SYSTEM = """\
You are an advanced expert Python programmer with deep knowledge of the FreqTrade platform, cryptocurrency futures markets, math, finances, and quantitative trading strategies. You use all resources and techniques available for writing high-quality, production-grade Python code.

STRICT RULES:
1. Return ONLY valid Python — no markdown fences, no prose outside comments.
2. Class name MUST be exactly: {strategy_name}
3. INTERFACE_VERSION = 3 | can_short = True
4. Zero lookahead bias — signals use only data available at candle close.
5. Preserve ALL method signatures: populate_indicators, populate_entry_trend,
   populate_exit_trend, custom_stoploss, leverage, custom_exit, confirm_trade_entry.
6. Allowed imports: talib, pandas, numpy, scipy, and Python stdlib (logging, datetime,
   collections, typing, math, statistics, etc.) — use freely.
   For FreqTrade imports: use ONLY imports you can verify exist in the FreqTrade source
   code or official documentation. Do NOT guess or invent import paths — if unsure
   whether a FreqTrade symbol exists, do not import it.
   Extra imports enabled by the operator: {extra_imports}
7. FORBIDDEN DataProvider calls — these do NOT exist and will crash FT:
   self.dp.market_pair_list, self.dp.get_pair_dataframe_with_candles,
   self.dp.get_analyzed_dataframe_between, self.dp.get_all_pair_dataframes.
   Only use: self.dp.get_pair_dataframe(pair, timeframe),
   self.dp.current_whitelist(), self.dp.runmode.
7. Start the file with this EXACT comment block (fill in the brackets):

# GENERATION_METADATA = {{
#   "generation": {gen},
#   "created_at": "AUTO",
#   "trigger": "{reason}",
#   "changelog": "[one line summary]"
# }}
# WHAT_CHANGED: [concrete description of what code changed]
# WHY: [reference specific numbers from the performance snapshot]
# HYPOTHESIS: [what metric should improve and by how much]
# RISK: [what could go wrong]

OPTIMISATION TARGETS:
  Profit Factor  > {pf}  |  Sharpe  > {sh}  |  Win Rate  > {wr}  |  Max Drawdown  < {dd}
"""

_CODE_USER = """\
=== ANALYSIS (from step 1 — IMPLEMENT THIS) ===
Concept chosen: {chosen_concept}
Plan: {implementation_plan}
Expected impact: {expected_impact}
Risk to manage: {risk}

=== CURRENT STRATEGY CODE ===
{code}

=== PERFORMANCE SNAPSHOT ===
{perf}

Now return the complete improved strategy as pure Python only.
"""

# ── Mode guidance strings ──────────────────────────────────────
_MODE_GUIDANCE = {
    "conservative": (
        "Metrics are poor or recent gens degraded. "
        "PRIORITY: fix broken fundamentals first. "
        "Choose a concept that reduces losses or improves entry quality. "
        "Do NOT add complexity — simplify if needed."
    ),
    "adaptive": (
        "Metrics are acceptable. "
        "Try ONE new concept that could meaningfully improve the weakest metric. "
        "Build on what worked in previous generations."
    ),
    "aggressive": (
        "Metrics are good. "
        "Experiment boldly — try a complex technique that could unlock a step change. "
        "Acceptable risk of regression since we have rollback protection."
    ),
}

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _select_mode(perf: dict, history: str) -> str:
    """Pick exploration mode: override from config, or auto-select from metrics."""
    override = cfg("evolution", "exploration_mode", default="adaptive")
    if override in ("conservative", "aggressive"):
        return override

    # Auto: look at recent performance
    m  = perf.get("metrics", {})
    wr = m.get("win_rate",      0)
    pf = m.get("profit_factor", 0)
    dd = m.get("max_drawdown",  1)
    targets = cfg("optimization", "targets", default={})
    wr_t = targets.get("win_rate",      0.52)
    pf_t = targets.get("profit_factor", 1.8)
    dd_t = targets.get("max_drawdown",  0.10)

    badly_off = (wr < wr_t * 0.8) or (pf < 1.0) or (dd > dd_t * 1.5)
    on_target = (wr >= wr_t) and (pf >= pf_t) and (dd <= dd_t)

    if badly_off:
        return "conservative"
    if on_target:
        return "aggressive"
    return "adaptive"


def _pick_next_concept(gen: int) -> list[str]:
    """Return the full concept list from config, highlighting the one to try this gen."""
    concepts = cfg("evolution", "concepts_to_try", default=[])
    if not concepts:
        return ["(no concepts configured — use free choice)"]
    return concepts


def _format_concepts(concepts: list[str], gen: int) -> str:
    """Format concept list; mark the suggested one for this generation."""
    if not concepts or concepts == ["(no concepts configured — use free choice)"]:
        return "  (free choice — use your judgement)"
    lines = []
    suggested_idx = (gen - 1) % len(concepts)
    for i, c in enumerate(concepts):
        marker = "  ← SUGGESTED for this generation" if i == suggested_idx else ""
        lines.append(f"  - {c}{marker}")
    return "\n".join(lines)


def _trim_code(code: str, max_chars: int = 7000) -> str:
    if len(code) <= max_chars:
        return code
    lines = code.splitlines()
    head, tail = lines[:70], lines[-70:]
    mid = len(lines) - 140
    return "\n".join(head + [f"    # ... {mid} lines truncated ..."] + tail)


def _trim_perf(perf: dict, max_chars: int = 2500) -> str:
    m = perf.get("metrics", {})
    # Include top 5 worst pairs by total pnl
    pair_stats = m.get("pair_stats") or perf.get("metrics", {}).get("pair_stats", {})
    worst_pairs = {}
    if isinstance(pair_stats, dict):
        sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1].get("total", 0))
        worst_pairs = dict(sorted_pairs[:5])

    slim = {
        "total_closed":       perf.get("total_closed"),
        "total_open":         perf.get("total_open"),
        "consecutive_losses": perf.get("consecutive_losses"),
        "metrics": {
            k: m.get(k) for k in [
                "win_rate", "profit_factor", "sharpe", "sortino",
                "max_drawdown", "total_pnl", "expectancy",
                "avg_win", "avg_loss", "long_wr", "short_wr",
            ]
        },
        "exit_reasons": m.get("exit_reasons", {}),
        "worst_5_pairs": worst_pairs,
        "patterns":      perf.get("patterns", {}),
        "recent_5":      perf.get("recent", [])[-5:],
    }
    if "ft_error_context" in perf:
        slim["ft_error_context"] = perf["ft_error_context"]
    text = json.dumps(slim, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n  ...(truncated)"
    return text


def _trim_history(history: str, max_chars: int = 1500) -> str:
    if not history:
        return "No previous evolutions."
    lines = history.strip().splitlines()
    trimmed = "\n".join(lines[-20:])
    return trimmed[-max_chars:] if len(trimmed) > max_chars else trimmed


# ══════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════

class LLMImprover:

    def improve(
        self,
        code:    str,
        perf:    dict,
        gen:     int,
        reason:  str,
        history: str,
    ) -> Optional[str]:

        targets  = cfg("optimization", "targets", default={})
        sname    = cfg("freqtrade", "strategy_name", default="AutoEvolveStrategy")
        provider = cfg("llm", "provider", default="anthropic")

        trimmed_code    = _trim_code(code)
        trimmed_perf    = _trim_perf(perf)
        trimmed_history = _trim_history(history)
        mode            = _select_mode(perf, trimmed_history)
        concepts        = _pick_next_concept(gen)
        concepts_fmt    = _format_concepts(concepts, gen)

        append_log("INFO",
            f"[improver {IMPROVER_VERSION}] provider={provider} gen={gen} "
            f"mode={mode} reason={reason}")

        # ── Step 1: Analysis ──────────────────────────────────
        analysis = self._step1_analyse(
            provider, trimmed_perf, trimmed_history,
            mode, concepts_fmt,
        )

        if not analysis:
            append_log("WARNING",
                "Analysis step failed — falling back to direct code generation")
            analysis = {
                "chosen_concept":       concepts[gen % len(concepts)] if concepts else "free choice",
                "implementation_plan":  "Improve based on the performance data",
                "expected_impact":      "Improve primary weak metric",
                "risk":                 "Unknown",
            }
        else:
            append_log("INFO",
                f"Analysis: concept={analysis.get('chosen_concept')} | "
                f"weakness={analysis.get('primary_weakness','?')[:80]}")

        # ── Step 2: Code ─────────────────────────────────────
        # Extra imports allowed by the user in config
        extra_imports_list = cfg("llm", "extra_allowed_imports", default=[]) or []
        if isinstance(extra_imports_list, list) and extra_imports_list:
            extra_imports_str = ", ".join(extra_imports_list)
        else:
            extra_imports_str = "none beyond the above"

        code_system = _CODE_SYSTEM.format(
            strategy_name  = sname,
            gen            = gen,
            reason         = reason,
            extra_imports  = extra_imports_str,
            pf = targets.get("profit_factor", 1.8),
            sh = targets.get("sharpe_ratio",  1.5),
            wr = targets.get("win_rate",      0.52),
            dd = targets.get("max_drawdown",  0.10),
        )
        code_user = _CODE_USER.format(
            chosen_concept      = analysis.get("chosen_concept", ""),
            implementation_plan = analysis.get("implementation_plan", ""),
            expected_impact     = analysis.get("expected_impact", ""),
            risk                = analysis.get("risk", ""),
            code                = trimmed_code,
            perf                = trimmed_perf,
        )

        prompt_chars = len(code_system) + len(code_user)
        # Token budget: 1 token ≈ 4 chars. Trim code further if over budget.
        max_tokens_budget = cfg("llm", "max_prompt_tokens", default=50000)
        budget_chars = max_tokens_budget * 4
        while len(code_system) + len(code_user) > budget_chars and len(trimmed_code) > 1000:
            # Shrink code first (most expensive element)
            trimmed_code = _trim_code(trimmed_code, max_chars=max(1000, len(trimmed_code) - 1000))
            code_user = _CODE_USER.format(
                chosen_concept      = analysis.get("chosen_concept", ""),
                implementation_plan = analysis.get("implementation_plan", ""),
                expected_impact     = analysis.get("expected_impact", ""),
                risk                = analysis.get("risk", ""),
                code                = trimmed_code,
                perf                = trimmed_perf,
            )
        final_chars = len(code_system) + len(code_user)
        append_log("INFO", f"Code prompt chars={final_chars} (~{final_chars//4} tokens)")

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                raw = self._call(provider, code_system, code_user)
                if raw:
                    cleaned = self._sanitize(self._strip_fences(raw))
                    if self._validate(cleaned, sname):
                        append_log("INFO",
                            f"LLM success gen={gen} attempt={attempt+1} "
                            f"chars={len(cleaned)}")
                        return cleaned
                    msg = (f"Validation failed attempt {attempt+1}: "
                           f"missing required class/method signatures")
                    logger.warning(msg); append_log("WARNING", msg)
            except Exception as e:
                err = str(e)
                is_rl = "429" in err or "rate" in err.lower() or "too many" in err.lower()
                wait  = (60 if is_rl else 15) * (attempt + 1)
                msg   = (f"LLM error attempt {attempt+1}/{max_attempts} "
                         f"(wait {wait}s): "
                         + ("Rate limited" if is_rl else err[:200]))
                logger.error(msg); append_log("ERROR", msg)
                if attempt < max_attempts - 1:
                    time.sleep(wait)

        append_log("ERROR", f"All LLM attempts failed for gen {gen}")
        return None

    # ── Step 1 ────────────────────────────────────────────────
    def _step1_analyse(
        self,
        provider:     str,
        trimmed_perf: str,
        history:      str,
        mode:         str,
        concepts_fmt: str,
    ) -> Optional[dict]:
        system = _ANALYSE_SYSTEM.format(concepts=concepts_fmt)
        user   = _ANALYSE_USER.format(
            perf         = trimmed_perf,
            history      = history,
            mode         = mode,
            mode_guidance = _MODE_GUIDANCE.get(mode, ""),
        )
        try:
            raw = self._call(provider, system, user)
            if not raw:
                return None
            # Strip any accidental markdown fences
            raw = re.sub(r"```(?:json)?\n?", "", raw)
            raw = re.sub(r"```\s*$",         "", raw, flags=re.MULTILINE).strip()
            # Extract JSON even if wrapped in extra prose
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            logger.debug(f"Analysis step exception: {e}")
        return None

    # ── HTTP call (shared) ────────────────────────────────────
    def _call(self, provider: str, system: str, user: str) -> str:
        # ── Gather config params for logging ─────────────────────
        c = cfg("llm", provider, default={})
        log_params = {
            "model":       c.get("model", "?"),
            "max_tokens":  c.get("max_tokens", "?"),
            "temperature": c.get("temperature", "n/a" if provider in ("anthropic", "ollama") else 0.6),
            "timeout":     cfg("llm", "timeout", default=600),
        }
        if provider == "openrouter":
            log_params["allow_fallbacks"]      = c.get("allow_fallbacks", True)
            log_params["max_price_prompt"]      = c.get("max_price_prompt", 0.10)
            log_params["max_price_completion"]  = c.get("max_price_completion", 0.40)
            log_params["sort"]                  = c.get("sort", "price")

        _llm_log("REQUEST", provider, log_params,
                 f"--- SYSTEM ---\n{system}\n--- USER ---\n{user}")

        t0 = time.time()
        if provider == "anthropic":
            response = self._anthropic(system, user)
        elif provider == "ollama":
            response = self._ollama(system, user)
        else:
            # Any other name treated as OpenAI-compatible REST
            response = self._openai_compat(provider, system, user)

        elapsed = round(time.time() - t0, 1)
        # Try to read model actually used from response (openrouter may route elsewhere)
        actual_model = getattr(self, "_last_actual_model", None)
        resp_params = {"elapsed_sec": elapsed, "chars": len(response)}
        if actual_model and actual_model != c.get("model"):
            resp_params["actual_model_used"] = actual_model   # OpenRouter routed elsewhere
        _llm_log("RESPONSE", provider, resp_params, response)
        return response

    def _anthropic(self, system: str, user: str) -> str:
        try:
            import anthropic as _ant
        except ImportError:
            raise ImportError("pip install anthropic")
        c   = cfg("llm", "anthropic", default={})
        key = c.get("api_key", "") or os.environ.get(c.get("api_key_env", "ANTHROPIC_API_KEY"), "")
        client = _ant.Anthropic(api_key=key)
        resp = client.messages.create(
            model      = c.get("model", "claude-opus-4-6"),
            max_tokens = c.get("max_tokens", 8192),
            system     = system,
            messages   = [{"role": "user", "content": user}],
        )
        return resp.content[0].text

    def _openai_compat(self, provider: str, system: str, user: str) -> str:
        try:
            import requests as _req
        except ImportError:
            raise ImportError("pip install requests")

        c        = cfg("llm", provider, default={})
        key      = c.get("api_key", "") or os.environ.get(c.get("api_key_env", ""), "")
        base_url = c.get("base_url", "https://api.openai.com/v1").rstrip("/")
        model    = c.get("model", "gpt-4o")
        max_tok  = c.get("max_tokens", 16384)
        timeout  = cfg("llm", "timeout", default=600)
        temp     = c.get("temperature", 0.6)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
        }
        if provider == "openrouter":
            headers["HTTP-Referer"] = c.get("site_url", "")
            headers["X-Title"]      = c.get("site_name", "AutoEvolve")

        payload = {
            "model":       model,
            "max_tokens":  max_tok,
            "temperature": temp,
            "messages":    [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }

        # Provider routing block — OpenRouter-specific syntax.
        # Other providers (chutes, openai, etc.) reject unknown top-level keys.
        if provider == "openrouter":
            allow_fallbacks = c.get("allow_fallbacks", True)
            price_prompt    = c.get("max_price_prompt",     0.10)
            price_compl     = c.get("max_price_completion", 0.40)
            sort_by         = c.get("sort", "price")
            payload["provider"] = {
                "allow_fallbacks": allow_fallbacks,
                "max_price": {
                    "prompt":     price_prompt,
                    "completion": price_compl,
                },
                "sort": sort_by,
            }

        key_hint = ("*" + key[-6:]) if len(key) > 6 else "(empty)"
        append_log("INFO",
            f"LLM HTTP POST {base_url}/chat/completions "
            f"model={model} key={key_hint}")

        r = _req.post(
            f"{base_url}/chat/completions",
            headers=headers, json=payload, timeout=timeout,
        )

        if r.status_code != 200:
            body = r.text[:500]
            append_log("ERROR", f"LLM HTTP {r.status_code} from {provider}: {body}")
            if r.status_code == 429:
                raise Exception(f"429 rate limited: {body}")
            if r.status_code == 401:
                raise Exception(f"401 unauthorized — check api_key in config.yaml: {body}")
            raise Exception(f"HTTP {r.status_code}: {body}")

        data  = r.json()
        text  = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        # Capture the model actually used (OpenRouter may route to a different model)
        actual_model = data.get("model", model)
        self._last_actual_model = actual_model   # expose for RESPONSE log in _call
        routed_note = f" (routed → {actual_model})" if actual_model != model else ""
        append_log("INFO",
            f"LLM OK: in={usage.get('prompt_tokens','?')} "
            f"out={usage.get('completion_tokens','?')} tokens{routed_note}")
        return text

    def _ollama(self, system: str, user: str) -> str:
        try:
            import requests as _req
        except ImportError:
            raise ImportError("pip install requests")
        c = cfg("llm", "ollama", default={})
        payload = {
            "model":   c.get("model", "codellama:34b"),
            "prompt":  f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]",
            "stream":  False,
            "options": {"num_predict": c.get("max_tokens", 8192)},
        }
        r = _req.post(
            f"{c.get('base_url', 'http://localhost:11434')}/api/generate",
            json=payload, timeout=cfg("llm", "timeout", default=600),
        )
        r.raise_for_status()
        return r.json().get("response", "")

    # ── Post-processing ───────────────────────────────────────
    def _strip_fences(self, raw: str) -> str:
        raw = re.sub(r"```(?:python)?\n?", "", raw)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
        return raw.strip()

    def _sanitize(self, code: str) -> str:
        """
        Fix common LLM mistakes before deploying:
        - CRLF line endings -> LF
        - Windows paths in string literals (e.g. D:\\Python\\freqtrade) produce
          SyntaxWarning or re.error. Strategy code never needs Windows paths,
          so replace every backslash inside string literals with a forward slash.
        """
        import re as _re

        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')


        # ── Auto-inject missing imports ───────────────────────────
        # The LLM sometimes uses stdlib names without importing them.
        # We detect usage and inject the import if it's absent.
        _AUTO_IMPORTS = [
            ("Decimal",     "from decimal import Decimal"),
            ("dataclass",   "from dataclasses import dataclass, field"),
            ("field",       "from dataclasses import dataclass, field"),
            ("Optional",    "from typing import Optional, List, Dict, Tuple"),
            ("List",        "from typing import Optional, List, Dict, Tuple"),
            ("Dict",        "from typing import Optional, List, Dict, Tuple"),
            ("Tuple",       "from typing import Optional, List, Dict, Tuple"),
            ("Union",       "from typing import Union"),
            ("Any",         "from typing import Any"),
            ("Callable",    "from typing import Callable"),
            ("defaultdict", "from collections import defaultdict"),
            ("deque",       "from collections import deque"),
            ("datetime",    "from datetime import datetime, timezone, timedelta"),
            ("timezone",    "from datetime import datetime, timezone, timedelta"),
            ("timedelta",   "from datetime import datetime, timezone, timedelta"),
            ("math",        "import math"),
            ("statistics",  "import statistics"),
        ]
        _header = "\n".join(code.split("\n")[:80])
        _to_inject = []
        _seen_lines = set()
        for _nm, _imp in _AUTO_IMPORTS:
            if _imp in _seen_lines:
                continue
            if _imp in _header:
                _seen_lines.add(_imp)
                continue
            if re.search(r'\b' + _nm + r'\b', code):
                _to_inject.append(_imp)
                _seen_lines.add(_imp)
        if _to_inject:
            _lines = code.split("\n")
            _insert_at = 0
            for _i, _ln in enumerate(_lines[:80]):
                _s = _ln.strip()
                if _s.startswith("import ") or _s.startswith("from "):
                    _insert_at = _i + 1
            for _imp in reversed(_to_inject):
                _lines.insert(_insert_at, _imp)
            code = "\n".join(_lines)

        # Replace Unicode whitespace look-alikes with plain ASCII space/nothing.
        # LLMs often emit NBSP, NNBSP, thin-space etc. in comments/strings.
        for _uch, _repl in [
            ('\u00ad', ''),    # SOFT HYPHEN — invisible, just remove
            ('\u00a0', ' '),   # NO-BREAK SPACE
            ('\u202f', ' '),   # NARROW NO-BREAK SPACE (NNBSP)
            ('\u2009', ' '),   # THIN SPACE
            ('\u2008', ' '),   # PUNCTUATION SPACE
            ('\u2007', ' '),   # FIGURE SPACE
            ('\u2006', ' '),   # SIX-PER-EM SPACE
            ('\u2005', ' '),   # FOUR-PER-EM SPACE
            ('\u2004', ' '),   # THREE-PER-EM SPACE
            ('\u2003', ' '),   # EM SPACE
            ('\u2002', ' '),   # EN SPACE
            ('\u3000', ' '),   # IDEOGRAPHIC SPACE
            ('\u205f', ' '),   # MEDIUM MATHEMATICAL SPACE
        ]:
            code = code.replace(_uch, _repl)

        def _slashes_to_fwd(m):
            # Replace all backslashes inside the matched string literal
            return m.group(0).replace('\\', '/')

        # Double-quoted single-line literals
        code = _re.sub(r'"[^"\n]*"', _slashes_to_fwd, code)
        # Single-quoted single-line literals
        code = _re.sub(r"'[^'\n]*'", _slashes_to_fwd, code)
        return code

    def _validate(self, code: str, sname: str) -> bool:
        required = [
            f"class {sname}",
            "def populate_indicators",
            "def populate_entry_trend",
            "def populate_exit_trend",
            "INTERFACE_VERSION = 3",
            "can_short = True",
        ]
        for check in required:
            if check not in code:
                logger.warning(f"LLM output missing: {check}")
                return False
        import warnings as _warnings
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("error")   # treat SyntaxWarning as error
                compile(code, "<llm_output>", "exec")
        except (SyntaxError, Exception) as e:
            # Check if this looks like truncation (syntax error near end of file)
            total_lines = code.count("\n")
            err_line = getattr(e, "lineno", 0) or 0
            if err_line and err_line >= total_lines - 5:
                logger.warning(
                    f"LLM output appears TRUNCATED at line {err_line}/{total_lines} "
                    f"— increase max_tokens in config.yaml (currently hitting token limit)"
                )
            else:
                logger.warning(f"LLM syntax/compile error: {e}")
            return False
        return True
