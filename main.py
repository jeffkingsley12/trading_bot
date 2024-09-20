import asyncio

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from config import CONFIG
from binance.spot import Spot
from history_data import (
    fetch_and_process_data,
    calculate_indicators,
    train_and_save_ml_model,
    calculate_hmm_states,
)
from prepare_env import get_api_key
from logs import logger

api_key, api_secret = get_api_key()
symbol = CONFIG["symbol"]


async def train_model():
    try:
        client = Spot(api_key, api_secret)
        logger.info(f"Fetching historical data for {symbol}")
        historical_data = await fetch_and_process_data(
            client, CONFIG["symbol"], CONFIG["interval"], CONFIG["limit"]
        )

        if historical_data is None or historical_data.empty:
            logger.error("Failed to fetch historical data or data is empty")
            return

        # Log some sample data
        logger.info(f"Sample of historical data:\n{historical_data.head()}")

        logger.info("Calculating indicators")
        df_with_indicators = await calculate_indicators(historical_data)

        # Check for NaN values and handle them
        # Check for NaN values and handle them
        nan_count = df_with_indicators.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values. Attempting to handle...")
            df_with_indicators = df_with_indicators.ffill().bfill()
            remaining_nan_count = df_with_indicators.isna().sum().sum()
            if remaining_nan_count > 0:
                logger.warning(
                    f"Unable to handle all NaNs. {remaining_nan_count} remaining."
                )
                # Instead of returning, we'll continue but exclude SMA_200 from our features
                df_with_indicators = df_with_indicators.drop("SMA_200", axis=1)
                logger.info("Dropped SMA_200 from feature set due to NaN values.")

        # Ensure we have enough data
        if len(df_with_indicators) < 100:  # Adjust this threshold as needed
            logger.error(
                f"Insufficient data: only {len(df_with_indicators)} rows available."
            )
            return

        # Define features, excluding SMA_200 if it was dropped
        features = [
            col
            for col in df_with_indicators.columns
            if col not in ["close", "HMM_State", "SMA_200"]
        ]

        logger.info("Training and saving ML model")
        model, scaler, hmm_model, hmm_scaler, mse, r2 = await train_and_save_ml_model(
            df_with_indicators
        )

        if model is None:
            logger.error("Failed to train model")
            return

        # # Handle potential NaN or infinite values
        # df_with_indicators = df_with_indicators.replace(
        #     [np.inf, -np.inf], np.nan
        # ).dropna()

        # Perform cross-validation for more reliable performance estimates
        features = [
            col
            for col in df_with_indicators.columns
            if col not in ["close", "HMM_State"]
        ]
        X = df_with_indicators[features]
        y = df_with_indicators["close"]

        cv_mse_scores = cross_val_score(
            model, X, y, cv=5, scoring="neg_mean_squared_error"
        )
        cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

        logger.info(
            f"Cross-validated MSE: {-np.mean(cv_mse_scores):.4f} (+/- {np.std(cv_mse_scores) * 2:.4f})"
        )
        logger.info(
            f"Cross-validated R2: {np.mean(cv_r2_scores):.4f} (+/- {np.std(cv_r2_scores) * 2:.4f})"
        )

        logger.info("Calculating HMM states")
        df_with_hmm = await calculate_hmm_states(df_with_indicators)

        if (
            "HMM_State" in df_with_hmm.columns
            and not df_with_hmm["HMM_State"].isna().all()
        ):
            logger.info("HMM states calculated successfully")

            # Log distribution of HMM states
            state_distribution = df_with_hmm["HMM_State"].value_counts(normalize=True)
            logger.info(f"HMM State Distribution:\n{state_distribution}")
        else:
            logger.warning("HMM state calculation failed or produced all NaN values")

        logger.info(f"Model training complete. RF MSE: {mse:.4f}, R2: {r2:.4f}")

        # Save the final dataframe
        df_with_hmm.to_csv("processed_data_with_hmm.csv", index=False)
        logger.info("Saved processed data with HMM states to CSV")

        # Optionally, you could add a function to visualize your results
        # await visualize_results(df_with_hmm, model)

    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        logger.exception("Traceback:")


# Optionally, add a visualization function
async def visualize_results(df, model):
    try:
        import matplotlib.pyplot as plt

        # Plot actual vs predicted prices
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["close"], label="Actual Price")
        plt.plot(df.index, model.predict(df[features]), label="Predicted Price")
        plt.title("Actual vs Predicted Prices")
        plt.legend()
        plt.savefig("price_prediction.png")
        logger.info("Saved price prediction plot")

        # Plot feature importances
        importances = pd.DataFrame(
            {"feature": features, "importance": model.feature_importances_}
        )
        importances = importances.sort_values("importance", ascending=False)
        plt.figure(figsize=(10, 6))
        plt.bar(importances["feature"], importances["importance"])
        plt.title("Feature Importances")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        logger.info("Saved feature importances plot")

    except Exception as e:
        logger.error(f"An error occurred during result visualization: {str(e)}")
        logger.exception("Traceback:")


if __name__ == "__main__":
    asyncio.run(train_model())
