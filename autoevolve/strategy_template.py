"""
autoevolve/strategy_template.py
================================
Generates the Generation-1 baseline FreqTrade strategy.
Kept in a separate module so it's easy to customise without
touching the orchestrator logic.
"""

from __future__ import annotations
from .utils import local_str


def baseline_template(class_name: str) -> str:
    """Return baseline FreqTrade strategy source code."""
    return f'''# GENERATION_METADATA = {{
#   "generation": 1, "parent": null,
#   "created_at": "{local_str()}",
#   "trigger": "baseline",
#   "metrics": {{}},
#   "changelog": "Initial baseline strategy — EMA trend + RSI + Volume + ATR stops"
# }}
# ================================================================
# {class_name} — Generation 1 (Baseline)
# 15m Bybit Futures USDT | can_short = True
# AUTO-MANAGED by AutoEvolve — each generation is checkpointed
# ================================================================

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from functools import reduce
from typing import Optional, Union

import talib.abstract as ta
from freqtrade.strategy import (
    IStrategy, Trade, DecimalParameter, IntParameter,
    stoploss_from_open,
)
from freqtrade.persistence import Trade


class {class_name}(IStrategy):
    """
    Baseline strategy for AutoEvolve generation 1.

    Logic:
      Trend filter : EMA-200 alignment
      Entry        : EMA-8 cross EMA-50 + RSI + volume surge + MACD confirmation
      Exit         : Reverse cross or RSI extreme + BB band
      Stoploss     : ATR-based dynamic trailing (custom_stoploss)
      TP           : Partial exit at 1.5R via custom_exit
    """

    INTERFACE_VERSION   = 3
    timeframe           = "15m"
    can_short           = True

    stoploss            = -0.04
    trailing_stop       = False
    use_custom_stoploss = True

    minimal_roi = {{"0": 0.06, "30": 0.04, "60": 0.025, "120": 0.015}}

    # ── Hyperopt parameters ───────────────────────────────────
    ema_fast   = IntParameter(8,   21,  default=8,   space="buy",  optimize=True)
    ema_slow   = IntParameter(40,  100, default=50,  space="buy",  optimize=True)
    ema_trend  = IntParameter(150, 250, default=200, space="buy",  optimize=True)
    rsi_period = IntParameter(10,  20,  default=14,  space="buy",  optimize=True)
    rsi_buy    = IntParameter(40,  65,  default=55,  space="buy",  optimize=True)
    vol_ma_p   = IntParameter(10,  30,  default=20,  space="buy",  optimize=True)
    vol_mul    = DecimalParameter(1.2, 2.5, default=1.5, space="buy",  optimize=True)
    atr_period = IntParameter(10,  20,  default=14,  space="sell", optimize=True)
    atr_sl_mul = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=True)
    atr_tr_mul = DecimalParameter(1.5, 4.0, default=2.5, space="sell", optimize=True)

    startup_candle_count: int = 210
    process_only_new_candles   = True
    use_exit_signal            = True
    exit_profit_only           = False
    ignore_roi_if_entry_signal = False

    # ── Indicators ────────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for n in [self.ema_fast.value, self.ema_slow.value, self.ema_trend.value]:
            dataframe[f"ema_{{n}}"] = ta.EMA(dataframe, timeperiod=n)

        dataframe["rsi"]       = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe["atr"]       = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe["volume_ma"] = dataframe["volume"].rolling(self.vol_ma_p.value).mean()
        dataframe["vol_surge"] = dataframe["volume"] > dataframe["volume_ma"] * self.vol_mul.value

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]

        bb = ta.BBANDS(dataframe, timeperiod=20)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"]   = bb["middleband"]

        ef = self.ema_fast.value
        es = self.ema_slow.value
        et = self.ema_trend.value
        dataframe["ema_f"] = dataframe[f"ema_{{ef}}"]
        dataframe["ema_s"] = dataframe[f"ema_{{es}}"]
        dataframe["ema_t"] = dataframe[f"ema_{{et}}"]

        # Use shift(1) on the slow EMA to avoid lookahead on crossover detection
        dataframe["cross_up"] = (
            (dataframe["ema_f"] > dataframe["ema_s"]) &
            (dataframe["ema_f"].shift(1) <= dataframe["ema_s"].shift(1))
        )
        dataframe["cross_dn"] = (
            (dataframe["ema_f"] < dataframe["ema_s"]) &
            (dataframe["ema_f"].shift(1) >= dataframe["ema_s"].shift(1))
        )
        return dataframe

    # ── Entry ─────────────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_cond = [
            dataframe["close"] > dataframe["ema_t"],
            dataframe["cross_up"],
            dataframe["rsi"] < self.rsi_buy.value,
            dataframe["rsi"] > 30,
            dataframe["vol_surge"],
            dataframe["macd_hist"] > 0,
            dataframe["close"] < dataframe["bb_upper"] * 0.995,
            dataframe["volume"] > 0,
        ]
        short_cond = [
            dataframe["close"] < dataframe["ema_t"],
            dataframe["cross_dn"],
            dataframe["rsi"] > (100 - self.rsi_buy.value),
            dataframe["rsi"] < 70,
            dataframe["vol_surge"],
            dataframe["macd_hist"] < 0,
            dataframe["close"] > dataframe["bb_lower"] * 1.005,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[reduce(lambda a, b: a & b, long_cond),  "enter_long"]  = 1
        dataframe.loc[reduce(lambda a, b: a & b, short_cond), "enter_short"] = 1
        return dataframe

    # ── Exit ──────────────────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe["ema_f"] < dataframe["ema_s"]) | (dataframe["rsi"] > 75)) &
            (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        dataframe.loc[
            ((dataframe["ema_f"] > dataframe["ema_s"]) | (dataframe["rsi"] < 25)) &
            (dataframe["volume"] > 0),
            "exit_short",
        ] = 1
        return dataframe

    # ── Custom stoploss (ATR trailing) ────────────────────────
    def custom_stoploss(
        self, pair: str, trade: "Trade", current_time: datetime,
        current_rate: float, current_profit: float, after_fill: bool, **kwargs,
    ) -> Optional[float]:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return self.stoploss
        atr = df.iloc[-1].get("atr", 0)
        if not atr:
            return self.stoploss
        sl_d = self.atr_sl_mul.value * atr / trade.open_rate
        if abs(current_profit) < sl_d:
            return stoploss_from_open(-sl_d, current_profit,
                                      is_short=trade.is_short, leverage=trade.leverage)
        tr_d = self.atr_tr_mul.value * atr / current_rate
        return stoploss_from_open(-tr_d, current_profit,
                                  is_short=trade.is_short, leverage=trade.leverage)

    # ── Leverage ──────────────────────────────────────────────
    def leverage(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_leverage: float, max_leverage: float,
        entry_tag: Optional[str], side: str, **kwargs,
    ) -> float:
        return 3.0

    # ── Partial TP ────────────────────────────────────────────
    def custom_exit(
        self, pair: str, trade: "Trade", current_time: datetime,
        current_rate: float, current_profit: float, **kwargs,
    ) -> Optional[Union[str, bool]]:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        atr = df.iloc[-1].get("atr", 0)
        if not atr:
            return None
        target = self.atr_sl_mul.value * atr / trade.open_rate * 1.5
        return "partial_tp" if current_profit >= target else None

    # ── Entry guard ───────────────────────────────────────────
    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime,
        entry_tag: Optional[str], side: str, **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return True
        atr = df.iloc[-1].get("atr", 0)
        # Reject entries during abnormally high volatility spikes
        return not (atr and (atr / rate) > 0.05)
'''
