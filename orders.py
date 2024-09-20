import asyncio
from binance.error import ClientError
from binance.lib.utils import get_timestamp
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
from validate import validate_order
from decimal import Decimal
from config import CONFIG
from utils import round_decimal
import aiohttp

from rate_limiters import create_order_with_retry
from enums import Order, OrderSide, OrderStatus, OrderType, TimeInForce, SymbolInfo
from logs import logger

first_run_timestamp = datetime.now()


async def get_time_range() -> tuple[int, int]:
    end_time = datetime.now()
    return (
        int(first_run_timestamp.timestamp() * 1000),
        int(end_time.timestamp() * 1000),
    )


async def my_trades(client: Any, symbol: str) -> List[Dict[str, Any]]:
    try:
        start_time, end_time = await get_time_range()
        response = client.get_orders(
            symbol=symbol,
            limit=100,
            startTime=start_time,
            endTime=end_time,
            timestamp=get_timestamp(),
        )

        if isinstance(response, list):
            logger.info("There are %s Previous Trades", len(response))
            return response
        else:
            logger.error("Unexpected response type: %s", type(response))
            return []
    except ClientError as error:
        logger.error(
            "Found trade error. status: %s, error code: %s, error message: %s",
            error.status_code,
            error.error_code,
            error.error_message,
        )
        return []


async def get_orders(client: Any, symbol: str):
    try:
        response = client.get_open_orders(symbol)

        if isinstance(response, list):
            logger.info("There are %s Open orders", len(response))
            return response
        else:
            logger.error("Unexpected response type: %s", type(response))
            return []
    except ClientError as error:
        logger.error(
            "Found order error. status: %s, error code: %s, error message: %s",
            error.status_code,
            error.error_code,
            error.error_message,
        )
        return []


async def cancel_open_orders(client: Any, symbol: str) -> List[Dict[str, Any]]:
    try:
        open_orders = await get_orders(client, symbol)
        if not open_orders:
            logger.info(f"No open orders found for {symbol} to be canceled")
            return []

        response = client.cancel_open_orders(symbol=symbol)
        logger.info(f"All open orders for {symbol} cancelled")
        return response
    except ClientError as error:
        logger.error(
            "Client Error in cancel_open_orders - status: %s, error code: %s, error message: %s",
            error.status_code,
            error.error_code,
            error.error_message,
        )
    except Exception as e:
        logger.error(f"Unexpected error in cancel_open_orders: {str(e)}")
    return []


async def check_time_sync(client: Any) -> None:
    try:
        server_time_response = client.time()
        server_time = server_time_response["serverTime"]
        local_time = int(time.time() * 1000)
        diff = abs(server_time - local_time)
        logger.info(
            f"Time difference between local system and Binance server: {diff} ms"
        )
        if diff > 1000:
            logger.warning(
                "System time is out of sync with Binance server. This may cause API signature errors."
            )
    except ClientError as e:
        logger.error(f"Failed to check time synchronization: {str(e)}")


async def place_order(
    client,
    symbol_info: SymbolInfo,
    is_buy: bool,
    quantity: Decimal,
    price: Decimal,
    minimum_lot_size: Decimal,
    order_type: OrderType,
    time_in_force: TimeInForce,
    stop_price: Optional[Decimal] = None,
    max_retries=3,
    position_tracker: Optional["PositionTracker"] = None,
) -> Optional[Order]:
    for attempt in range(max_retries):
        try:
            validation_result, adjusted_quantity, adjusted_price = await validate_order(
                quantity, price, minimum_lot_size, is_buy
            )

            if not validation_result.is_valid:
                logger.error(
                    f"{'Buy' if is_buy else 'Sell'} order validation failed: {validation_result.error_message}"
                )
                return None

            side = "BUY" if is_buy else "SELL"
            order_params = {
                "symbol": validation_result.symbol,
                "side": side,
                "type": order_type.value,
                "quantity": str(adjusted_quantity),
            }

            if order_type in [
                OrderType.LIMIT,
                OrderType.STOP_LOSS_LIMIT,
                OrderType.TAKE_PROFIT_LIMIT,
            ]:
                order_params.update(
                    {"timeInForce": time_in_force.value, "price": str(adjusted_price)}
                )

            if order_type in [
                OrderType.STOP_LOSS,
                OrderType.STOP_LOSS_LIMIT,
                OrderType.TAKE_PROFIT,
                OrderType.TAKE_PROFIT_LIMIT,
            ]:
                if stop_price is None or stop_price <= 0:
                    raise ValueError(
                        f"Valid stop price must be specified for {order_type.value} orders"
                    )
                order_params["stopPrice"] = str(
                    round_decimal(stop_price, CONFIG["precision"])
                )

            response = await create_order_with_retry(client, order_params)

            if not response:
                logger.error(
                    "Order placement failed: No response received from the exchange"
                )
                return None

            order = Order(
                order_id=response.get("orderId"),
                status=OrderStatus(response.get("status", "NEW")),
                symbol=response.get("symbol"),
                side=OrderSide(response.get("side", side)),
                quantity=Decimal(response.get("origQty", str(adjusted_quantity))),
                price=Decimal(response.get("price", str(adjusted_price))),
                type=OrderType(response.get("type", order_type.value)),
                time_in_force=TimeInForce(response.get("timeInForce", time_in_force)),
                executed_quantity=Decimal(response.get("executedQty", "0")),
                cumulative_quote_quantity=Decimal(
                    response.get("cummulativeQuoteQty", "0")
                ),
                client_order_id=response.get("clientOrderId", ""),
                time=response.get("transactTime", 0),
                stop_price=Decimal(response.get("stopPrice", str(stop_price or 0))),
            )

            logger.info(f"Order placed successfully: {order}")

            if position_tracker:
                await position_tracker.update_position(symbol_info["symbol"], order)

            return order

        except (ClientError, aiohttp.ClientError) as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to place order after {max_retries} attempts: {e}")
                raise
            logger.warning(
                f"Order placement failed, retrying ({attempt + 1}/{max_retries}): {e}"
            )
            await asyncio.sleep(1)

    return None


async def check_balance(client, asset: str, required_amount: Decimal) -> bool:
    try:
        account_info = client.account()
        balance = next(
            (
                Decimal(balance["free"])
                for balance in account_info["balances"]
                if balance["asset"] == asset
            ),
            Decimal("0"),
        )

        if balance >= required_amount:
            logger.info(
                f"Sufficient balance of {asset}: {balance} (required: {required_amount})"
            )
            return True
        logger.warning(
            f"Insufficient balance of {asset}: {balance} (required: {required_amount})"
        )
        return False

    except Exception as e:
        logger.error(
            f"Error fetching wallet information for {asset}: {str(e)}", exc_info=True
        )
        return False
