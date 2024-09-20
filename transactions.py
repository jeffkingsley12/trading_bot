import asyncio
from typing import Any
from decimal import Decimal


import tracemalloc
from position import PositionTracker, TradingStrategy
from rate_limiters import RateLimiter

from config import CONFIG

from logs import logger
from orders import cancel_open_orders

tracemalloc.start()


class TradingExecutor:
    def __init__(
        self,
        symbol: str,
        client: Any,
        tracker: Any,
        execute_order,
        symbol_info,
        current_price,
        sell_price,
        buy_price,
        stop_loss,
        profit_target,
        orders,
        rate_limiter: RateLimiter,
    ):
        self.symbol = symbol
        self.client = client
        self.symbol_info = symbol_info
        self.rate_limiter = rate_limiter
        self.position_tracker = PositionTracker(
            symbol=CONFIG["symbol"],
            order=orders,  # Assuming you have orders list
            client=client,  # Assuming you have a Binance client instance
            tracker=tracker,  # Assuming you have a balance tracker instance
            execute_order=execute_order,  # Assuming you have an balance updater after order execution function
            symbol_info=symbol_info,  # Assuming you have symbol info
            current_price=Decimal(current_price),  # Initial price
            sell_price=Decimal(sell_price),  # Initial sell price
            buy_price=Decimal(buy_price),  # Initial buy price
            stop_loss=Decimal(stop_loss),  # Example stop loss
            profit_target=Decimal(profit_target),  # Example profit target
            rate_limiter=self.rate_limiter,  # Assuming you have a rate limiter instance
            strategy=TradingStrategy(),  # Create an instance of the strategy
        )
        self.orders = orders
        self.cancel_open_orders = cancel_open_orders

    async def execute_trading_strategy(
        self,
        symbol: str,
        quantity: Decimal,
        buy_price: Decimal,
        sell_price: Decimal,
        long_entry: bool,
        short_entry: bool,
        exit_long: bool,
        exit_short: bool,
        current_price: Decimal,
        stop_loss: Decimal,
        profit_target: Decimal,
        reversal_pattern: bool,
        bullish_engulfing: bool,
        bearish_engulfing: bool,
    ):
        logger.info(f"Orders data passed to PositionTracker: {len(self.orders[1])}")
        try:
            await self.position_tracker.initialize(self.orders)
            # Monitor and manage existing positions
            # await self.position_tracker.monitor_positions(
            #     symbol,
            #     bullish_engulfing,
            #     bearish_engulfing,
            # )

            start_time = asyncio.get_event_loop().time()

            # Handle entries
            if long_entry:
                await self.position_tracker.enter_position(
                    symbol,
                    quantity,
                    buy_price,
                    "long",
                    stop_loss,
                    profit_target,
                    delay_seconds=0,
                )

            if short_entry:
                await self.position_tracker.enter_position(
                    symbol,
                    -quantity,
                    sell_price,
                    "short",
                    stop_loss,
                    profit_target,
                    delay_seconds=0,
                )

            # Handle exits
            if exit_long:
                await self.position_tracker.exit_position(
                    symbol,
                    Decimal(str(quantity)),
                    Decimal(str(sell_price)),
                    "exit_long",
                    Decimal(str(CONFIG["max_slippage"])),
                )

            if exit_short:
                await self.position_tracker.exit_position(
                    symbol,
                    Decimal(str(-quantity)),
                    Decimal(str(sell_price)),
                    "exit_short",
                    Decimal(str(CONFIG["max_slippage"])),
                )

            # Monitor and manage existing positions

            await self.position_tracker.monitor_positions(
                symbol,
                bullish_engulfing,
                bearish_engulfing,
            )

            # Log current position value
            position_value = await self.position_tracker.get_position_value(symbol)
            logger.info(f"Current position value for {symbol}: {position_value}")

            end_time = asyncio.get_event_loop().time()
            logger.info(f"Strategy execution time: {end_time - start_time} seconds")

        except asyncio.CancelledError:
            logger.warning(f"Strategy execution for {symbol} was cancelled")
            try:
                response = await cancel_open_orders(self.client, symbol=symbol)
                return response
            except Exception as e:
                logger.error(f"Error cancelling open orders: {str(e)}")

        except Exception as e:
            logger.error(
                f"Error in strategy execution for {symbol}: {str(e)}", exc_info=True
            )
            await self.position_tracker.emergency_closure(symbol)

    async def __aenter__(self):
        await self.position_tracker.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.position_tracker.__aexit__(exc_type, exc, tb)
