from decimal import Decimal

CONFIG = {
    "interval": "15m",  # 48h = 2 days
    "wallet": "Spot",
    "limit": "5000",  # historical Trade Data
    "decimal_places": Decimal("5"),
    "symbol": "DOGEFDUSD",
    "account_balance": Decimal("25"),
    "risk_percentage": Decimal("0.07"),
    "deviation_percentage": Decimal("0.00098"),
    "precision": 5,
    "minimum_lot_size": Decimal("40.0"),
    "profit_target": Decimal("0.00082"),
    "trailing_stop_loss_offset": Decimal("0.00052"),
    "rsi_period": 15,
    "sma_period": 30,
    "atr_period": 15,
    "buying_rsi": 50,
    "best_params": [1, 1.5, 2, 2.5, 3],
    "error_retry_interval": float("0.5"),
    "error_retry_count": float("0.5"),
    "position_check_interval": float("0.5"),
    # Rate Limiters
    "initial_rate": Decimal("5"),
    "max_rate": Decimal("1"),
    "max_slippage": Decimal("0.001"),  # Maximum slippage allowed
    # New additions
    "atr_multiplier": Decimal("1.5"),  # Initial value, to be optimized
    "stop_loss_type": "volatility",  # Options: "fixed", "trailing", "volatility"
    "fixed_stop_loss": Decimal("0.00075"),  # Used if stop_loss_type is "fixed"
    "max_position_size": Decimal("100"),  # Maximum position size in quote currency
    "max_open_trades": 5,  # Maximum number of open trades allowed
    "use_ml_model": True,  # Whether to use machine learning predictions
    "ml_model_threshold": Decimal("0.6"),  # Threshold for ML model predictions
    "backtest_period": "1d",  # Period for backtesting (e.g., "6m", "1y", "2y")
    "optimization_metric": "sharpe_ratio",  # Metric to optimize (e.g., "profit_loss_ratio", "sharpe_ratio")
    "rebalance_interval": "7d",  # How often to rebalance the portfolio
    "log_level": "INFO",  # Logging level (e.g., "DEBUG", "INFO", "WARNING")
    "trailing_stop_percentage": Decimal("0.02"),  # 2% trailing stop,
    "partial_take_profit_levels": [
        (Decimal("0.01"), Decimal("0.2")),  # 1% profit, sell 20%
        (Decimal("0.02"), Decimal("0.3")),  # 2% profit, sell 30%
        (Decimal("0.03"), Decimal("0.5")),  # 3% profit, sell 50%
    ],
}
