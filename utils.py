import numpy as np
import pandas as pd
from config import (
    CONFIG,
)  # Ensure parameters.py exists and contains necessary constants
from ta.trend import ADXIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange
from decimal import Decimal, ROUND_DOWN
from typing import Tuple, Type
from logs import *
import backoff


def round_decimal(value, places):
    if not isinstance(value, Decimal):
        value = Decimal(str(value))  # Convert to Decimal only if it's not already
    return value.quantize(Decimal(f"1E-{places}"), rounding=ROUND_DOWN)


def adjust_number(value) -> float:
    # Ensure the value is a float
    value = float(value)

    # Round the value to the specified precision
    rounded_value = round(value, CONFIG["precision"])

    # Convert the rounded value to string with sufficient precision
    formatted_value = f"{rounded_value:.{CONFIG['precision']}f}"

    # Extract the last three decimal places``
    if "." in formatted_value:
        decimal_part = formatted_value.split(".")[1]
        # Ensure we have at least 3 decimal places
        last_three_decimals = decimal_part[-3:].ljust(3, "0")
    else:
        last_three_decimals = "000"  # Handle case where there are no decimals

    # Construct the result as a float
    result_str = "0.00" + last_three_decimals
    return float(result_str)


def log_and_raise(message: str, exception_type: Type[Exception] = Exception):
    """
    Logs an error message and raises an exception.

    Args:
        message (str): The error message to log and include in the exception.
        exception_type (Type[Exception], optional): The type of exception to raise. Defaults to Exception.

    Raises:
        The specified exception type with the given message.
    """
    logger.error(message)
    raise exception_type(message)


async def calculate_adx(
    df: pd.DataFrame, period: int = CONFIG["atr_period"]
) -> pd.DataFrame:
    """
    Calculate various technical indicators.
    """
    adx = ADXIndicator(df["high"], df["low"], df["close"], window=period)
    df["adx"] = adx.adx()
    df["+DI"] = adx.adx_pos()
    df["-DI"] = adx.adx_neg()

    vwap = VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"])
    df["vwap"] = vwap.volume_weighted_average_price()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period)
    df["atr"] = atr.average_true_range()

    return df


async def check_engulfing_vectorized(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Vectorized check for bullish and bearish engulfing patterns.
    """
    is_bullish_engulfing = (
        (df["close"].shift(1) < df["open"].shift(1))
        & (df["close"] > df["open"])
        & (df["close"] > df["open"].shift(1))
        & (df["open"] < df["close"].shift(1))
    )

    is_bearish_engulfing = (
        (df["close"].shift(1) > df["open"].shift(1))
        & (df["close"] < df["open"])
        & (df["close"] < df["open"].shift(1))
        & (df["open"] > df["close"].shift(1))
    )

    return is_bullish_engulfing, is_bearish_engulfing


async def calculate_adaptive_thresholds(
    df: pd.DataFrame, rsi_baseline: float = CONFIG["buying_rsi"], atr_factor: float = 2
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate adaptive thresholds for RSI based on ATR.
    """
    if "atr" not in df.columns:
        raise ValueError("DataFrame must contain 'atr' column")

    # Calculate adaptive RSI low and high thresholds
    df["rsi_low"] = rsi_baseline - (df["atr"] * atr_factor).round(CONFIG["precision"])
    df["rsi_high"] = (
        100 - rsi_baseline + (df["atr"] * atr_factor).round(CONFIG["precision"])
    )

    # Return the calculated thresholds
    return df["rsi_low"].iloc[-1], df["rsi_high"].iloc[-1]


async def apply_total_signal(
    df: pd.DataFrame,
    lookback_period: int = CONFIG["rsi_period"],
    bb_width_window: int = 100,
    adx_threshold: float = 25,
    volume_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Apply total signal based on multiple technical indicators and volume.
    """
    required_columns = [
        "high",
        "low",
        "close",
        "open",
        "volume",
        "rsi",
        "bbl",
        "bbh",
        "bb_width",
    ]

    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    try:
        df = await calculate_adx(df, lookback_period)

        await calculate_adaptive_thresholds(df)
        df["volume_confirmed"] = (
            df["volume"]
            > df["volume"].rolling(window=lookback_period).mean() * volume_factor
        )
        bb_width_threshold = (
            df["bb_width"].rolling(window=bb_width_window).quantile(0.2)
        )
        (
            df["is_bullish_engulfing"],
            df["is_bearish_engulfing"],
        ) = await check_engulfing_vectorized(df)

        # Condition dictionary
        cond = {
            "price_below_bb": df["close"] < df["bbl"],
            "price_above_bb": df["close"] > df["bbh"],
            "rsi_overbought": df["rsi"] < df["rsi_low"],
            "rsi_oversold": df["rsi"] > df["rsi_high"],
            "bullish_di_cross": (df["+DI"] > df["-DI"])
            & (df["+DI"].shift(1) <= df["-DI"].shift(1)),
            "bearish_di_cross": (df["-DI"] > df["+DI"])
            & (df["-DI"].shift(1) <= df["+DI"].shift(1)),
            "price_above_vwap": df["close"] > df["vwap"],
            "price_below_vwap": df["close"] < df["vwap"],
            "is_bullish_engulfing": df["is_bullish_engulfing"],
            "is_bearish_engulfing": df["is_bearish_engulfing"],
        }

        # Define buy and sell conditions
        buy_conditions = cond["rsi_overbought"] & (
            cond["bullish_di_cross"]
            | cond["price_below_vwap"]
            | cond["is_bullish_engulfing"]
            | cond["price_below_bb"]
        )
        sell_conditions = cond["rsi_oversold"] & (
            cond["bearish_di_cross"]
            | cond["price_above_vwap"]
            | cond["is_bearish_engulfing"]
            | cond["price_above_bb"]
        )

        # Generate signals
        df["BuySignal"] = np.where(buy_conditions, 1, 0)
        df["SellSignal"] = np.where(sell_conditions, 1, 0)
        df["TotalSignal"] = df["BuySignal"] - df["SellSignal"]

        # Log RSI thresholds
        logger.info(
            f"Buy: {df['BuySignal'].iloc[-1]}, Sell: {df['SellSignal'].iloc[-1]}, RSI Low: {df['rsi_low'].iloc[-1]}, RSI High: {df['rsi_high'].iloc[-1]}"
        )

        return df
    except Exception as e:
        logger.error(f"Error in apply_total_signal: {e}")
        raise
