# GENERATION_METADATA = {
#   "generation": 27,
#   "parent": 26,
#   "created_at": "2026-02-27 19:50:15 +02:00",
#   "trigger": "profit_drawdown_23.0pct_from_peak",
#   "metrics": {"win_rate": 0.7, "profit_factor": 2.763, "sharpe": 88.5517, "max_drawdown": 0.032563},
#   "changelog": "Added multi‑timeframe confluence entry filter with pivot break and volume spike"
# }
# WHAT_CHANGED: Introduced 1h and 4h EMA/RSI confluence, volume spike, and recent pivot break as mandatory entry criteria.
# WHY: Current drawdown spikes (max_drawdown 0.032) suggest stop‑losses are triggered; tightening entries should reduce false signals.
# HYPOTHESIS: Win rate ↑ to ≥0.55, profit factor ↑ to ≥1.9, Sharpe ↑ by ~0.2 due to fewer premature exits.
# RISK: Over‑filtering may lower trade count, increasing idle capital and reducing total PnL in strong trends.

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Dict, Any

import talib.abstract as ta
from freqtrade.strategy import IStrategy, Trade, stoploss_from_open


class AutoEvolveStrategy(IStrategy):
    """
    AutoEvolve Strategy Generation 27 — Multi‑timeframe EMA/RSI confluence with volume spike,
    recent pivot break and ATR‑based dynamic stop‑loss.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # Strategy parameters
    timeframe = "5m"
    startup_candle_count = 300  # enough for higher‑timeframe calculations

    minimal_roi = {
        "0": 0.15,
        "45": 0.08,
        "90": 0.04,
        "180": 0
    }

    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False

    # Indicator parameters
    ema_fast_period = 20
    ema_slow_period = 50
    rsi_period = 14
    atr_period = 14
    volume_ma_period = 20
    volume_spike_factor = 1.5
    atr_stop_factor = 1.5

    # ----------------------------------------------------------------------
    # Indicator calculation
    # ----------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all technical indicators for base and higher timeframes."""
        # Base timeframe indicators
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)
        dataframe["volume_ma"] = dataframe["volume"].rolling(self.volume_ma_period).mean()

        # Recent pivot high/low over last 3 candles (excluding current)
        pivot_high = dataframe["high"].shift(1).rolling(3).max()
        pivot_low = dataframe["low"].shift(1).rolling(3).min()
        dataframe["pivot_high"] = pivot_high
        dataframe["pivot_low"] = pivot_low

        # Higher timeframe indicators (1h and 4h)
        pair = metadata.get("pair")
        if not pair:
            return dataframe

        # 1h
        df_1h = self.dp.get_pair_dataframe(pair=pair, timeframe="1h")
        if df_1h is not None and not df_1h.empty:
            df_1h["ema_fast_1h"] = ta.EMA(df_1h, timeperiod=self.ema_fast_period)
            df_1h["ema_slow_1h"] = ta.EMA(df_1h, timeperiod=self.ema_slow_period)
            df_1h["rsi_1h"] = ta.RSI(df_1h, timeperiod=self.rsi_period)
            df_1h["volume_ma_1h"] = df_1h["volume"].rolling(self.volume_ma_period).mean()
            # Align to base timeframe timestamps (use last known value)
            df_1h = df_1h.reindex(dataframe.index, method="ffill")
            dataframe["ema_fast_1h"] = df_1h["ema_fast_1h"]
            dataframe["ema_slow_1h"] = df_1h["ema_slow_1h"]
            dataframe["rsi_1h"] = df_1h["rsi_1h"]
            dataframe["volume_ma_1h"] = df_1h["volume_ma_1h"]

        # 4h
        df_4h = self.dp.get_pair_dataframe(pair=pair, timeframe="4h")
        if df_4h is not None and not df_4h.empty:
            df_4h["ema_fast_4h"] = ta.EMA(df_4h, timeperiod=self.ema_fast_period)
            df_4h["ema_slow_4h"] = ta.EMA(df_4h, timeperiod=self.ema_slow_period)
            df_4h["rsi_4h"] = ta.RSI(df_4h, timeperiod=self.rsi_period)
            df_4h["volume_ma_4h"] = df_4h["volume"].rolling(self.volume_ma_period).mean()
            df_4h = df_4h.reindex(dataframe.index, method="ffill")
            dataframe["ema_fast_4h"] = df_4h["ema_fast_4h"]
            dataframe["ema_slow_4h"] = df_4h["ema_slow_4h"]
            dataframe["rsi_4h"] = df_4h["rsi_4h"]
            dataframe["volume_ma_4h"] = df_4h["volume_ma_4h"]

        return dataframe

    # ----------------------------------------------------------------------
    # Entry signal generation
    # ----------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Set entry signals only when all confluence conditions are satisfied."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # Ensure higher timeframe columns exist; otherwise skip
        required_cols = [
            "ema_fast_1h", "ema_slow_1h",
            "ema_fast_4h", "ema_slow_4h",
            "rsi_1h", "rsi_4h",
            "volume_ma"
        ]
        if not all(col in dataframe.columns for col in required_cols):
            return dataframe

        # EMA cross on 1h (current candle)
        ema_cross_up_1h = (dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]) & (
            dataframe["ema_fast_1h"].shift(1) <= dataframe["ema_slow_1h"].shift(1)
        )
        ema_cross_down_1h = (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]) & (
            dataframe["ema_fast_1h"].shift(1) >= dataframe["ema_slow_1h"].shift(1)
        )

        # Directional bias on 4h
        bias_long_4h = dataframe["ema_fast_4h"] > dataframe["ema_slow_4h"]
        bias_short_4h = dataframe["ema_fast_4h"] < dataframe["ema_slow_4h"]

        # RSI extremes on all timeframes
        rsi_long_cond = (
            (dataframe["rsi"] < 30) &
            (dataframe["rsi_1h"] < 30) &
            (dataframe["rsi_4h"] < 30)
        )
        rsi_short_cond = (
            (dataframe["rsi"] > 70) &
            (dataframe["rsi_1h"] > 70) &
            (dataframe["rsi_4h"] > 70)
        )

        # Volume spike on base timeframe
        vol_spike = dataframe["volume"] > (self.volume_spike_factor * dataframe["volume_ma"])

        # Pivot break
        break_up = dataframe["close"] > dataframe["pivot_high"]
        break_down = dataframe["close"] < dataframe["pivot_low"]

        # Long entry condition: all True
        long_condition = (
            ema_cross_up_1h &
            bias_long_4h &
            rsi_long_cond &
            vol_spike &
            break_up
        )
        dataframe.loc[long_condition, "enter_long"] = 1

        # Short entry condition: all True
        short_condition = (
            ema_cross_down_1h &
            bias_short_4h &
            rsi_short_cond &
            vol_spike &
            break_down
        )
        dataframe.loc[short_condition, "enter_short"] = 1

        return dataframe

    # ----------------------------------------------------------------------
    # Exit signal generation (simple EMA crossover)
    # ----------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals based on EMA crossovers on the base timeframe."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        ema_cross_down = (dataframe["ema_fast"] < dataframe["ema_slow"]) & (
            dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1)
        )
        ema_cross_up = (dataframe["ema_fast"] > dataframe["ema_slow"]) & (
            dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1)
        )

        dataframe.loc[ema_cross_down, "exit_long"] = 1
        dataframe.loc[ema_cross_up, "exit_short"] = 1

        return dataframe

    # ----------------------------------------------------------------------
    # Custom stop‑loss (ATR based)
    # ----------------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, entry_rate: float, **kwargs) -> float:
        """Dynamic stop‑loss: entry_price - ATR * factor for longs, +ATR*factor for shorts."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return self.stoploss

        latest = df.iloc[-1]
        atr = latest.get("atr")
        if pd.isna(atr):
            return self.stoploss

        if trade.is_short:
            stop_price = entry_rate + atr * self.atr_stop_factor
            return (stop_price - current_rate) / current_rate
        else:
            stop_price = entry_rate - atr * self.atr_stop_factor
            return (stop_price - current_rate) / current_rate

    # ----------------------------------------------------------------------
    # Leverage definition
    # ----------------------------------------------------------------------
    def leverage(self, pair: str, is_short: bool, **kwargs) -> float:
        """Fixed leverage of 5× for all trades."""
        return 5.0

    # ----------------------------------------------------------------------
    # Optional custom exit logic (placeholder)
    # ----------------------------------------------------------------------
    def custom_exit(self,
                    pair: str,
                    trade: Trade,
                    current_time: datetime,
                    **kwargs) -> Optional[Dict[str, Any]]:
        """No custom exit – rely on exit signals and ROI."""
        return None

    # ----------------------------------------------------------------------
    # Confirm entry – additional safety checks
    # ----------------------------------------------------------------------
    def confirm_trade_entry(self,
                            pair: str,
                            order_type: str,
                            amount: float,
                            rate: float,
                            time_in_force: str,
                            current_time: datetime,
                            entry_tag: Optional[str],
                            side: str,
                            **kwargs) -> bool:
        """Validate candle range, ATR size and candle direction before allowing entry."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return False

        last = dataframe.iloc[-1]

        # Candle range check (max 4% of price)
        candle_range = last["high"] - last["low"]
        if (candle_range / rate) > 0.04:
            return False

        # ATR should not be excessive (>8% of price)
        if pd.isna(last.get("atr")) or (last["atr"] / rate) > 0.08:
            return False

        # Direction consistency with candle body
        bullish = last["close"] > last["open"]
        bearish = last["close"] < last["open"]
        if side == "long" and not bullish:
            return False
        if side == "short" and not bearish:
            return False

        return True

    # ----------------------------------------------------------------------
    # Custom fee – default to exchange handling
    # ----------------------------------------------------------------------
    def custom_fee(self,
                   pair: str,
                   trade: Trade,
                   order_type: str,
                   amount: float,
                   rate: float,
                   is_short: bool,
                   **kwargs) -> float:
        """Use exchange default fee handling."""
        return 0.0