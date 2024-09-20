import pandas as pd
import numpy as np
import time
from typing import Optional

from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from ta.volatility import AverageTrueRange, BollingerBands
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from logs import logger
from logs import LoggingMonitor
from decimal import Decimal
from config import CONFIG
from concurrent.futures import ThreadPoolExecutor
from utils import round_decimal
from utils import apply_total_signal, adjust_number

# Add new imports
from sklearn.cluster import KMeans

import joblib
import asyncio
import traceback  # Add this import
import os

precision = CONFIG["precision"]


async def train_and_save_ml_model(
    df: pd.DataFrame,
    model_path: str = "rf_model.joblib",
    scaler_path: str = "scaler.joblib",
    hmm_path: str = "hmm_model.joblib",
):
    logger.info(f"Initial DataFrame shape: {df.shape}")
    logger.info(f"Initial DataFrame columns: {df.columns}")
    logger.info(f"Initial DataFrame non-null counts:\n{df.count()}")
    logger.info(f"First few rows of initial DataFrame:\n{df.head()}")

    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr",
        "rsi",
        "SMA",
        "MACD",
        "MACD_Signal",
        "SMA_50",
        "SMA_200",
        "bbm",
        "bbh",
        "bbl",
        "bb_width",
    ]

    df = handle_nan_values(df)

    # Check if all required features are present
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logger.error(f"Missing features in dataframe: {missing_features}")
        return None, None, None, None, None, None

    X = df[features]
    y = df["close"].shift(-1)

    logger.info(f"X shape before removing NaNs: {X.shape}")
    logger.info(f"y shape before removing NaNs: {y.shape}")

    # Remove rows with NaN values
    valid_indices = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_indices]
    y = y[valid_indices]

    logger.info(f"X shape after removing NaNs: {X.shape}")
    logger.info(f"y shape after removing NaNs: {y.shape}")

    # Remove last row as it will have NaN in y due to shift
    X = X[:-1]
    y = y[:-1]

    logger.info(f"Final X shape: {X.shape}")
    logger.info(f"Final y shape: {y.shape}")

    if X.empty or y.empty:
        logger.error(
            "After preprocessing, X or y is empty. Cannot proceed with model training."
        )
        return None, None, None, None, None, None

    logger.info(f"Shape of X before splitting: {X.shape}")
    logger.info(f"NaN count in X before splitting: {X.isna().sum().sum()}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ValueError as e:
        logger.error(f"Error in train_test_split: {str(e)}")
        return None, None, None, None, None, None

    logger.info(f"Shape of X_train: {X_train.shape}")
    logger.info(f"Shape of y_train: {y_train.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Shape of X_train_scaled: {X_train_scaled.shape}")
    logger.info(f"Shape of X_test_scaled: {X_test_scaled.shape}")

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        model = await loop.run_in_executor(pool, train_model, X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Random Forest Model performance - MSE: {mse:.4f}, R2: {r2:.4f}")

    # Train HMM model
    hmm_features = ["close", "volume", "rsi", "MACD"]
    hmm_data = df[hmm_features].copy()

    # Handle NaN values
    hmm_data = hmm_data.ffill().ffill(axis=1).bfill().bfill(axis=1)

    if hmm_data.empty:
        logger.error("No data left after handling NaNs. Unable to train HMM model.")
        return model, scaler, None, None, mse, r2

    hmm_scaler = StandardScaler()
    hmm_scaled_data = hmm_scaler.fit_transform(hmm_data)

    logger.info(f"Shape of hmm_scaled_data: {hmm_scaled_data.shape}")

    try:
        with ProcessPoolExecutor() as pool:
            hmm_model = await loop.run_in_executor(
                pool, train_hmm_model, hmm_scaled_data
            )
    except Exception as e:
        logger.error(f"Error training HMM model: {str(e)}")
        hmm_model = None

    # Save models and scalers
    await asyncio.gather(
        loop.run_in_executor(None, joblib.dump, model, model_path),
        loop.run_in_executor(None, joblib.dump, scaler, scaler_path),
    )

    logger.info(f"Random Forest Model saved to {model_path}")
    logger.info(f"Random Forest Scaler saved to {scaler_path}")

    if hmm_model is not None:
        await asyncio.gather(
            loop.run_in_executor(None, joblib.dump, hmm_model, hmm_path),
            loop.run_in_executor(None, joblib.dump, hmm_scaler, "hmm_scaler.joblib"),
        )
        logger.info(f"HMM Model saved to {hmm_path}")
        logger.info("HMM Scaler saved to hmm_scaler.joblib")
    else:
        logger.warning("HMM Model not saved due to training error")

    return model, scaler, hmm_model, hmm_scaler, mse, r2


def handle_nan_values(data: pd.DataFrame) -> pd.DataFrame:
    nan_count = data.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in the data. Handling NaNs...")
        data = data.ffill().bfill()  # Forward fill then backward fill
        remaining_nan_count = data.isna().sum().sum()
        if remaining_nan_count > 0:
            logger.warning(
                f"Unable to fill all NaN values. {remaining_nan_count} NaNs remaining. Dropping rows with NaN..."
            )
            data = data.dropna()
    return data


def train_model(X_train_scaled, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    mse_scores = -cv_scores
    r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")

    logger.info(
        f"Cross-validated MSE: {np.mean(mse_scores):.4f} (+/- {np.std(mse_scores) * 2:.4f})"
    )
    logger.info(
        f"Cross-validated R2: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores) * 2:.4f})"
    )

    # Train on full dataset
    model.fit(X_train_scaled, y_train)

    return model


# async def train_model(df_with_indicators):
#     rf_model, rf_scaler, hmm_model, hmm_scaler, mse, r2 = await train_and_save_ml_model(
#         df_with_indicators
#     )
#     # Use the returned values as needed
#     return rf_model, rf_scaler, hmm_model, hmm_scaler, mse, r2


def preprocess_data(data, n_components=2):
    # Handle NaN and inf values
    data = np.nan_to_num(data, nan=0.0, posinf=1e30, neginf=-1e30)

    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    logger.info(f"Data shape after PCA: {pca_data.shape}")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    return pca_data


def train_hmm_model(hmm_data, random_state=42):
    logger.info(f"Starting HMM model training with data shape: {hmm_data.shape}")

    ns = [2, 3, 4]  # Try different numbers of components
    best_overall_model = None
    best_overall_score = float("-inf")

    for n in ns:
        logger.info(f"Trying HMM with {n} components")
        h = hmm.GaussianHMM(
            n_components=n,
            n_iter=2000,  # Increase max iterations
            tol=1e-6,  # Decrease tolerance for stricter convergence
            random_state=random_state,
            covariance_type="full",
        )

        try:
            h.fit(hmm_data)
            score = h.score(hmm_data)
            if score > best_overall_score:
                best_overall_model = h
                best_overall_score = score
            logger.info(f"HMM with {n} components: score = {score}")
        except Exception as e:
            logger.warning(f"Error in HMM training with {n} components: {str(e)}")

    if best_overall_model is None:
        raise ValueError("Failed to train any valid HMM model")

    logger.info(
        f"Best model: {best_overall_model.n_components} components, score: {best_overall_score}"
    )
    return best_overall_model


def fallback_state_calculation(data):
    # Simple fallback method using K-means clustering
    n_clusters = 3  # You can adjust this
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return kmeans.fit_predict(data)


async def train_hmm_model_with_timeout(hmm_data, timeout=300):  # 5 minutes timeout
    loop = asyncio.get_running_loop()
    try:
        with ProcessPoolExecutor() as pool:
            hmm_model = await asyncio.wait_for(
                loop.run_in_executor(pool, train_hmm_model, hmm_data),
                timeout=timeout,
            )
        return hmm_model
    except asyncio.TimeoutError:
        logger.error(f"HMM training timed out after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Error during HMM training: {str(e)}")
        logger.exception("Traceback:")
        return None


async def calculate_hmm_states(df):
    # Prepare data for HMM
    data = df[["close", "volume", "rsi", "MACD"]].copy()

    # Check for NaN values
    nan_count = data.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in the data. Handling NaNs...")
        data = data.ffill().bfill()

        if data.isna().sum().sum() > 0:
            logger.error("Unable to fill all NaN values. Dropping rows with NaN...")
            data = data.dropna()

            if data.empty:
                logger.error(
                    "No data left after dropping NaN values. Unable to fit HMM."
                )
                df["HMM_State"] = np.nan
                return df

    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    pca_data = pca.fit_transform(scaled_data)

    logger.info(f"Data shape after PCA: {pca_data.shape}")

    # Try HMM training
    try:
        hmm_model = await train_hmm_model_with_timeout(pca_data)

        if hmm_model is None:
            raise ValueError("HMM training failed or timed out.")

        # Predict hidden states
        hidden_states = hmm_model.predict(pca_data)
        df["HMM_State"] = hidden_states

    except Exception as e:
        logger.warning(
            f"HMM training or prediction failed: {str(e)}. Using fallback method."
        )
        df["HMM_State"] = fallback_state_calculation(data)

    # Validate HMM states
    if df["HMM_State"].isna().all():
        logger.warning(
            "HMM state calculation produced all NaN values. Using fallback method."
        )
        df["HMM_State"] = fallback_state_calculation(data)

    logger.info("HMM states or fallback states calculated successfully")
    return df


async def load_model_and_predict(
    df: pd.DataFrame,
    model_path: str = "rf_model.joblib",
    scaler_path: str = "scaler.joblib",
    hmm_path: str = "hmm_model.joblib",
) -> pd.DataFrame:
    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr",
        "rsi",
        "SMA",
        "MACD",
        "MACD_Signal",
        "SMA_50",
        "SMA_200",
        "bbm",
        "bbh",
        "bbl",
        "bb_width",
    ]
    hmm_features = ["close", "volume", "rsi", "MACD"]

    if not all(
        os.path.exists(path)
        for path in [model_path, scaler_path, hmm_path, "hmm_scaler.joblib"]
    ):
        raise FileNotFoundError(
            "One or more model files not found. Please train the models first."
        )

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        rf_model, rf_scaler, hmm_model, hmm_scaler = await asyncio.gather(
            loop.run_in_executor(pool, joblib.load, model_path),
            loop.run_in_executor(pool, joblib.load, scaler_path),
            loop.run_in_executor(pool, joblib.load, hmm_path),
            loop.run_in_executor(pool, joblib.load, "hmm_scaler.joblib"),
        )

    logger.info("All models and scalers loaded successfully")

    # Random Forest predictions
    X_predict = df[features].copy()
    X_predict_scaled = await loop.run_in_executor(None, rf_scaler.transform, X_predict)
    raw_predictions = await loop.run_in_executor(
        None, rf_model.predict, X_predict_scaled
    )
    raw_predictions = np.round(raw_predictions, CONFIG["precision"])
    df["ML_Prediction"] = raw_predictions

    # HMM predictions
    hmm_data = df[hmm_features].copy()

    # Handle NaN values in hmm_data
    # hmm_data = hmm_data.fillna(method="ffill").fillna(method="bfill")
    hmm_data = hmm_data.ffill().bfill()
    hmm_scaled_data = await loop.run_in_executor(None, hmm_scaler.transform, hmm_data)

    try:
        hidden_states = await loop.run_in_executor(
            None, hmm_model.predict, hmm_scaled_data
        )
        df["HMM_State"] = hidden_states
    except Exception as e:
        logger.error(f"Error in HMM prediction: {str(e)}")
        df["HMM_State"] = np.nan  # Set HMM_State to NaN if prediction fails

    logger.info(
        f"Predictions completed and added to dataframe {df['HMM_State'].iloc[-1]}"
    )

    return df


async def get_historical_data(
    client,
    symbol: str = CONFIG["symbol"],
    interval: str = CONFIG["interval"],
    limit: int = CONFIG["limit"],
) -> Optional[pd.DataFrame]:
    raw_data = await fetch_and_process_data(client, symbol, interval, limit)
    if raw_data is None:
        return None

    df = await calculate_indicators(raw_data)
    df = await load_model_and_predict(df)  # Move this up before calculate_signals
    df = await calculate_signals(df)
    df = await calculate_hmm_states(df)
    df = await apply_total_signal(df)
    df = await calculate_trade_parameters(df)

    await log_signals(df, symbol)

    return await clean_dataframe(df)


async def fetch_and_process_data(
    client, symbol: str, interval: str, limit: int
) -> Optional[pd.DataFrame]:
    try:
        raw_data = client.klines(symbol, interval, limit=limit)
        if not raw_data:
            logger.warning(f"No raw data fetched for {symbol}")
            return None

        df = pd.DataFrame(
            raw_data,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "qav",
                "num_trades",
                "taker_base_vol",
                "taker_quote_vol",
                "ignore",
            ],
        )
        df.index = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df.dropna(inplace=True)

        if df.empty:
            logger.warning(f"Processed DataFrame is empty for {symbol}")
            return None

        return df
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {e}")
        return None


async def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Input DataFrame shape: {df.shape}")
    logger.info(f"Input DataFrame columns: {df.columns}")
    logger.info(f"Input DataFrame non-null counts:\n{df.count()}")

    # ATR
    atr_indicator = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["atr"] = atr_indicator.average_true_range()

    # RSI
    rsi_indicator = RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_indicator.rsi()

    # SMA
    sma_indicator = SMAIndicator(close=df["close"], window=20)
    df["SMA"] = sma_indicator.sma_indicator()

    # MACD
    macd_indicator = MACD(
        close=df["close"], window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD"] = macd_indicator.macd()
    df["MACD_Signal"] = macd_indicator.macd_signal()

    # SMA 50 and 200
    df["SMA_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()

    # Only calculate SMA_200 if we have enough data
    if len(df) >= 200:
        df["SMA_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()
    else:
        logger.warning("Not enough data for SMA_200. Using SMA_50 instead.")
        df["SMA_200"] = df["SMA_50"]

    # Handle NaN values
    df = df.ffill().bfill()

    # Check remaining NaNs
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Remaining NaN counts:\n{nan_counts}")

        # Fill SMA_200 NaNs with SMA_50 or closing price
        df["SMA_200"].fillna(df["SMA_50"], inplace=True)
        df["SMA_200"].fillna(df["close"], inplace=True)

    # Bollinger Bands
    bb_indicator = BollingerBands(close=df["close"], window=14, window_dev=2)
    df["bbm"] = bb_indicator.bollinger_mavg()
    df["bbh"] = bb_indicator.bollinger_hband()
    df["bbl"] = bb_indicator.bollinger_lband()
    df["bb_width"] = bb_indicator.bollinger_wband()

    # Handle NaN values
    df = df.ffill().bfill()

    logger.info(f"Output DataFrame shape: {df.shape}")
    logger.info(f"Output DataFrame columns: {df.columns}")
    logger.info(f"Output DataFrame non-null counts:\n{df.count()}")
    logger.info(f"NaN counts:\n{df.isna().sum()}")

    return df


async def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["MACD_Crossover"] = df["MACD"] > df["MACD_Signal"]
    df["UPTREND"] = df["SMA_50"] > df["SMA_200"]

    current_price = df["close"].iloc[-1]
    last_sma = df["SMA"].iloc[-1]

    df["current_l"] = current_price < last_sma
    df["current_h"] = current_price > last_sma

    # Add HMM-based signals only if HMM_State is present
    if "HMM_State" in df.columns:
        logger.info(
            f"HMM_State found in DataFrame. Adding HMM-based signals {df['HMM_State'].iloc[-1]}."
        )
        df["HMM_BuySignal"] = (df["HMM_State"] == 0) & (df["HMM_State"].shift(1) != 0)
        df["HMM_SellSignal"] = (df["HMM_State"] == 2) & (df["HMM_State"].shift(1) != 2)
    else:
        logger.warning("HMM_State not found in DataFrame. Skipping HMM-based signals.")
        df["HMM_BuySignal"] = False
        df["HMM_SellSignal"] = False

    return df


async def calculate_trade_parameters(df: pd.DataFrame) -> pd.DataFrame:
    current_price = Decimal(str(df["close"].iloc[-1]))
    # atr = Decimal(str(df["atr"].iloc[-1]))
    # last_rsi = Decimal(str(df["rsi"].iloc[-1]))
    current_change = Decimal(str(df["close"].pct_change().iloc[-1]))
    ml_prediction = Decimal(str(df["ML_Prediction"].iloc[-1]))

    # Compute risk amount and prices
    risk_amount = round_decimal(
        Decimal(str(CONFIG["account_balance"]))
        * Decimal(str(CONFIG["risk_percentage"])),
        CONFIG["precision"],
    )
    buy_price = round_decimal(
        current_price * (Decimal("1") - Decimal(str(CONFIG["deviation_percentage"]))),
        CONFIG["precision"],
    )
    sell_price = round_decimal(
        current_price * (Decimal("1") + Decimal(str(CONFIG["deviation_percentage"]))),
        CONFIG["precision"],
    )
    profit = round_decimal(
        current_price * (Decimal("1") + Decimal(str(CONFIG["profit_target"]))),
        CONFIG["precision"],
    )
    stop_loss_value = round_decimal(
        current_price - Decimal(str(CONFIG["trailing_stop_loss_offset"])),
        CONFIG["precision"],
    )
    quantity = max(int(risk_amount / buy_price), CONFIG["minimum_lot_size"])

    buy_signals = bool(df["BuySignal"].iloc[-1] == 1)
    sell_signals = bool(df["SellSignal"].iloc[-1] == 1)

    stop_loss_value = (
        ml_prediction
        if (buy_signals and current_price > ml_prediction)
        else stop_loss_value
    )
    profit = (
        ml_prediction if (sell_signals and current_price > ml_prediction) else profit
    )

    # Update DataFrame with calculated values
    df["Buy_Price"] = buy_price
    df["Sell_Price"] = sell_price
    df["Stop_Loss"] = stop_loss_value
    df["Risk_Amount"] = risk_amount
    df["Profit_Target"] = profit
    df["Quantity"] = quantity
    df["Price_Change"] = current_change
    # Calculate trading signals
    # Revised logic for long entry
    df["long_entry"] = (
        df["current_l"]
        & (~df["MACD_Crossover"] | ~df["UPTREND"])
        & (df["ML_Prediction"] < df["close"])
        & (df["BuySignal"] == 1)
        & df["HMM_BuySignal"]  # Add HMM buy signal condition
        # & df["is_bullish_engulfing"]
    )

    df["short_entry"] = (
        df["current_h"]
        & (df["MACD_Crossover"] | df["UPTREND"])
        & (df["ML_Prediction"] > df["close"])
        & (df["SellSignal"] == 1)
        & df["HMM_SellSignal"]  # Add HMM sell signal condition
        # & df["is_bearish_engulfing"]
    )

    # Calculate exit signals
    df["exit_long"] = (
        (df["close"].pct_change() > float(profit))
        | (df["close"] < float(stop_loss_value))
        | (df["ML_Prediction"] < df["close"])
    )
    df["exit_short"] = (
        (df["close"].pct_change() < -float(profit))
        | (df["close"] > (df["close"] + float(stop_loss_value)))
        | (df["ML_Prediction"] > df["close"])
    )

    # Finalize buy and sell signals
    df["Buy"] = df["long_entry"] | df["exit_short"]
    df["Sell"] = df["short_entry"] | df["exit_long"]
    return df


async def log_signals(df: pd.DataFrame, symbol: str) -> None:
    if df["Buy"].iloc[-1] or df["Sell"].iloc[-1]:
        logger.info(
            "Sell and Buy signals for %s:\n%s",
            symbol,
            df[["close", "rsi", "bbm", "bbh", "bbl", "atr", "SMA", "ML_Prediction"]]
            .tail(1)
            .to_string(),
        )
    else:
        logger.info(
            "No buy or sell signals for %s:\n%s",
            symbol,
            df[["close", "rsi", "bbm", "bbh", "bbl", "atr", "SMA", "ML_Prediction"]]
            .tail(1)
            .to_string(),
        )


async def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "SMA_50",
        "SMA_200",
        "MACD",
        "MACD_Signal",
        "bbm",
        "bbh",
        "bbl",
        "atr",
        "SMA",
        "volume",
        "close_time",
        "qav",
        "num_trades",
        "taker_base_vol",
        "taker_quote_vol",
        "ignore",
    ]
    df = df.drop(columns_to_drop, axis=1, errors="ignore")
    return df
