# ==============================================================
# AutoEvolveStrategy — Generation 1 (Baseline)
# Timeframe: 15m | Market: Bybit Futures USDT
# ==============================================================
# THIS FILE IS AUTO-MANAGED by autoevolve/improver.py
# Do not edit manually unless you know what you're doing.
# Each generation is checkpointed before overwriting.
# ==============================================================
# GENERATION_METADATA = {
#   "generation": 1,
#   "parent_generation": None,
#   "created_at": "auto-set by deployer",
#   "trigger_reason": "baseline",
#   "metrics_at_creation": {},
#   "llm_changelog": "Initial baseline strategy"
# }
# ==============================================================

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from functools import reduce
from typing import Optional, Union, Dict

import talib.abstract as ta
from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    DecimalParameter,
    IntParameter,
    CategoricalParameter,
    informative,
    stoploss_from_open,
    stoploss_from_absolute,
)
from freqtrade.persistence import Trade


class AutoEvolveStrategy(IStrategy):
    """
    AutoEvolve Baseline Strategy — 15m Bybit Futures

    Core logic:
    - Trend filter: EMA 50/200 alignment
    - Entry: EMA8 cross + RSI confirmation + volume surge
    - Exit: ATR trailing stop + partial TP at 1.5R
    - Risk: ATR-based dynamic stoploss

    This strategy is intentionally well-structured so the LLM
    can clearly understand and modify each component independently.
    """

    # ── Strategy metadata ─────────────────────────────────────
    INTERFACE_VERSION = 3
    strategy_version = "1.0.0"  # bumped by deployer each generation

    timeframe = "15m"
    can_short = True

    # ── Stoploss / ROI ─────────────────────────────────────────
    stoploss = -0.04           # initial hard stop (overridden by custom_stoploss)
    trailing_stop = False      # we use custom_stoploss instead
    use_custom_stoploss = True

    minimal_roi = {
        "0":   0.06,
        "30":  0.04,
        "60":  0.025,
        "120": 0.015,
    }

    # ── Hyperopt parameters ────────────────────────────────────
    # Trend filter
    ema_fast = IntParameter(8, 21,   default=8,   space="buy",  optimize=True)
    ema_slow = IntParameter(50, 100, default=50,  space="buy",  optimize=True)
    ema_trend = IntParameter(150, 250, default=200, space="buy", optimize=True)

    # RSI
    rsi_period    = IntParameter(10, 20, default=14, space="buy", optimize=True)
    rsi_buy_max   = IntParameter(40, 65, default=55, space="buy", optimize=True)
    rsi_sell_min  = IntParameter(35, 60, default=45, space="sell", optimize=True)

    # Volume
    vol_ma_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    vol_surge_mul = DecimalParameter(1.2, 2.5, default=1.5, space="buy", optimize=True)

    # ATR stoploss
    atr_period     = IntParameter(10, 20, default=14, space="sell", optimize=True)
    atr_sl_mul     = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=True)
    atr_trail_mul  = DecimalParameter(1.5, 4.0, default=2.5, space="sell", optimize=True)

    # ── Misc ───────────────────────────────────────────────────
    startup_candle_count: int = 210
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # ── Indicator computation ──────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ─ EMAs ─
        for length in [self.ema_fast.value, self.ema_slow.value, self.ema_trend.value]:
            dataframe[f"ema_{length}"] = ta.EMA(dataframe, timeperiod=length)

        # ─ RSI ─
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        # ─ ATR ─
        dataframe["atr"] = ta.ATR(dataframe,
                                   timeperiod=self.atr_period.value)

        # ─ Volume MA ─
        dataframe["volume_ma"] = (
            dataframe["volume"]
            .rolling(window=self.vol_ma_period.value)
            .mean()
        )
        dataframe["volume_surge"] = (
            dataframe["volume"] > dataframe["volume_ma"] * self.vol_surge_mul.value
        )

        # ─ MACD (used as secondary confirmation, LLM may weight it) ─
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"]        = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"]   = macd["macdhist"]

        # ─ Bollinger Bands ─
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe["bb_upper"] = bollinger["upperband"]
        dataframe["bb_mid"]   = bollinger["middleband"]
        dataframe["bb_lower"] = bollinger["lowerband"]
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        )

        # ─ Stochastic RSI (additional momentum filter) ─
        stochrsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe["stochrsi_k"] = stochrsi["fastk"]
        dataframe["stochrsi_d"] = stochrsi["fastd"]

        # ─ EMA cross signals (shift(1) ensures no repainting) ─
        dataframe["ema_fast_val"]  = dataframe[f"ema_{self.ema_fast.value}"]
        dataframe["ema_slow_val"]  = dataframe[f"ema_{self.ema_slow.value}"]
        dataframe["ema_trend_val"] = dataframe[f"ema_{self.ema_trend.value}"]

        dataframe["ema_cross_up"] = (
            (dataframe["ema_fast_val"] > dataframe["ema_slow_val"]) &
            (dataframe["ema_fast_val"].shift(1) <= dataframe["ema_slow_val"].shift(1))
        )
        dataframe["ema_cross_down"] = (
            (dataframe["ema_fast_val"] < dataframe["ema_slow_val"]) &
            (dataframe["ema_fast_val"].shift(1) >= dataframe["ema_slow_val"].shift(1))
        )

        return dataframe

    # ── Entry signals ──────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ─ Long conditions ─
        long_conditions = [
            # Trend: price above EMA200
            dataframe["close"] > dataframe["ema_trend_val"],
            # EMA fast crosses above slow
            dataframe["ema_cross_up"],
            # RSI not overbought
            dataframe["rsi"] < self.rsi_buy_max.value,
            dataframe["rsi"] > 30,
            # Volume confirmation
            dataframe["volume_surge"],
            # MACD histogram positive
            dataframe["macd_hist"] > 0,
            # Not at BB upper (avoid chasing)
            dataframe["close"] < dataframe["bb_upper"] * 0.995,
            # Data quality
            dataframe["volume"] > 0,
        ]

        dataframe.loc[
            reduce(lambda a, b: a & b, long_conditions),
            "enter_long"
        ] = 1

        # ─ Short conditions ─
        short_conditions = [
            # Trend: price below EMA200
            dataframe["close"] < dataframe["ema_trend_val"],
            # EMA fast crosses below slow
            dataframe["ema_cross_down"],
            # RSI not oversold
            dataframe["rsi"] > (100 - self.rsi_buy_max.value),
            dataframe["rsi"] < 70,
            # Volume confirmation
            dataframe["volume_surge"],
            # MACD histogram negative
            dataframe["macd_hist"] < 0,
            # Not at BB lower (avoid chasing)
            dataframe["close"] > dataframe["bb_lower"] * 1.005,
            # Data quality
            dataframe["volume"] > 0,
        ]

        dataframe.loc[
            reduce(lambda a, b: a & b, short_conditions),
            "enter_short"
        ] = 1

        return dataframe

    # ── Exit signals ───────────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # ─ Long exit ─
        long_exit_conditions = [
            (
                (dataframe["ema_fast_val"] < dataframe["ema_slow_val"]) |
                (dataframe["rsi"] > 75) |
                (dataframe["close"] >= dataframe["bb_upper"])
            ),
            dataframe["volume"] > 0,
        ]

        dataframe.loc[
            reduce(lambda a, b: a & b, long_exit_conditions),
            "exit_long"
        ] = 1

        # ─ Short exit ─
        short_exit_conditions = [
            (
                (dataframe["ema_fast_val"] > dataframe["ema_slow_val"]) |
                (dataframe["rsi"] < 25) |
                (dataframe["close"] <= dataframe["bb_lower"])
            ),
            dataframe["volume"] > 0,
        ]

        dataframe.loc[
            reduce(lambda a, b: a & b, short_exit_conditions),
            "exit_short"
        ] = 1

        return dataframe

    # ── Custom stoploss (ATR trailing) ─────────────────────────
    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """
        ATR-based trailing stoploss:
        - Initial: entry ± ATR * atr_sl_mul
        - Once in profit > 1R: trail at ATR * atr_trail_mul
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr", None)

        if atr is None or atr == 0:
            return self.stoploss

        # Calculate 1R (initial risk)
        initial_sl_distance = self.atr_sl_mul.value * atr / trade.open_rate

        # If we haven't reached 1R profit yet → use initial ATR stop
        if abs(current_profit) < initial_sl_distance:
            return stoploss_from_open(
                -initial_sl_distance,
                current_profit,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )

        # Once in profit → trail at atr_trail_mul
        trail_distance = self.atr_trail_mul.value * atr / current_rate
        return stoploss_from_open(
            -trail_distance,
            current_profit,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    # ── Leverage ───────────────────────────────────────────────
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
    ) -> float:
        """Conservative fixed leverage — LLM may make this dynamic."""
        return 3.0

    # ── Custom exit (partial TP) ───────────────────────────────
    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[Union[str, bool]]:
        """
        Partial take profit at 1.5R.
        Returns exit reason string (FreqTrade handles the exit).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr", None)

        if atr is None or atr == 0:
            return None

        # 1.5R profit target
        initial_risk = self.atr_sl_mul.value * atr / trade.open_rate
        profit_target = initial_risk * 1.5

        if current_profit >= profit_target:
            return "partial_tp_1r5"

        return None

    # ── Confirm entry (optional extra filter) ─────────────────
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """Reject entries if spread is too wide (futures safety)."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return True

        last_candle = dataframe.iloc[-1]
        atr = last_candle.get("atr", 0)

        # Skip if ATR is abnormally high (potential news spike)
        if atr > 0 and (atr / rate) > 0.05:
            return False

        return True
