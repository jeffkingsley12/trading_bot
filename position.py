from parse import get_price_symbol
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from decimal import Decimal
from binance.error import ClientError
from logs import logger
from utils import round_decimal
from asyncio import Semaphore
import time as time_module
from config import CONFIG
from orders import place_order, get_orders, check_balance, cancel_open_orders
from rate_limiters import (
    rate_limited,
    RateLimiter,
    InsufficientBalance,
    OrderPlacementError,
)
import traceback
import heapq
from enums import (
    Order,
    OrderState,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    OrderEntry,
    Position,
    ExitAttempt,
)
from functools import lru_cache
from collections import deque
from validate import get_symbol_info, validate_order

exit_attempt = ExitAttempt()


class OrderQueue:
    def __init__(self):
        self.queue = []  # Using a list for heapq operations
        self.order_id_counter = 0  # To ensure unique ordering for same price/time

    def add(self, order_id: int, quantity: Decimal, price: Decimal):
        self.order_id_counter += 1
        heapq.heappush(self.queue, (price, self.order_id_counter, order_id, quantity))
        logger.debug(
            f"Order added to queue: Order ID={order_id}, Quantity={quantity}, Price={price}"
        )
        logger.debug(f"Queue after add: {self.queue}")

    def remove_order(self, order_id: int) -> Optional[Tuple[int, Decimal, Decimal]]:
        for i, (price, _, id_, quantity) in enumerate(self.queue):
            if id_ == order_id:
                removed = self.queue.pop(i)
                heapq.heapify(self.queue)  # Restore heap property
                logger.debug(
                    f"Removed order: ID={order_id}, Quantity={quantity}, Price={price}"
                )
                logger.debug(f"Queue after removal: {self.queue}")
                logger.info(f"Remaining quantity in queue: {self.total_quantity()}")
                return (id_, quantity, price)
        logger.warning(f"Order ID {order_id} not found in queue")
        return None

    def remove(self, quantity: Decimal) -> List[Tuple[int, Decimal, Decimal]]:
        removed = []
        remaining = quantity
        logger.debug(f"Attempting to remove {quantity} from queue")
        logger.debug(f"Queue before removal: {self.queue}")

        while remaining > 0 and self.queue:
            price, _, order_id, order_quantity = self.queue[0]
            if order_quantity <= remaining:
                heapq.heappop(self.queue)
                removed.append((order_id, order_quantity, price))
                remaining -= order_quantity
                logger.debug(
                    f"Removed entire order: ID={order_id}, Quantity={order_quantity}"
                )
            else:
                remove_quantity = remaining
                removed.append((order_id, remove_quantity, price))
                self.queue[0] = (
                    price,
                    self.queue[0][1],
                    order_id,
                    order_quantity - remove_quantity,
                )
                heapq.heapify(self.queue)  # Restore heap property
                remaining = Decimal("0")
                logger.debug(
                    f"Partially removed order: ID={order_id}, Removed={remove_quantity}, Remaining={order_quantity - remove_quantity}"
                )

        if remaining > 0:
            logger.warning(
                f"Could not remove entire requested quantity. Remaining: {remaining}"
            )
            raise ValueError(
                f"Insufficient quantity in queue. Requested: {quantity}, Available: {quantity - remaining}"
            )

        logger.debug(f"Orders removed from queue: {removed}")
        logger.debug(f"Queue after removal: {self.queue}")
        logger.info(f"Remaining quantity in queue: {self.total_quantity()}")
        return removed

    def total_quantity(self) -> Decimal:
        total = sum(quantity for _, _, _, quantity in self.queue)
        logger.debug(f"Total quantity in queue: {total}")
        return total

    def total_price(self) -> Decimal:
        total = sum(quantity * price for price, _, _, quantity in self.queue)
        logger.debug(f"Total price in queue: {total}")
        return total

    def __len__(self):
        return len(self.queue)

    def peek(self) -> Optional[Tuple[int, Decimal, Decimal]]:
        if self.queue:
            price, _, order_id, quantity = self.queue[0]
            return (order_id, quantity, price)
        return None

    def is_empty(self) -> bool:
        return len(self.queue) == 0


class TradingStrategy:
    @staticmethod
    def is_stop_loss_triggered(
        side: OrderSide,
        current_price: Decimal,
        average_price: Decimal,
        stop_loss: Decimal,
    ) -> bool:
        logger.info(
            f"Stop Loss Check: Current Price={current_price}, Position Average Price={average_price}, Stop Loss={stop_loss}"
        )
        if side == OrderSide.BUY:
            return current_price <= average_price * (Decimal(1) - stop_loss)
        else:
            return current_price >= average_price * (Decimal(1) + stop_loss)

    @staticmethod
    async def get_market_condition(
        bullish_engulfing: bool, bearish_engulfing: bool, symbol: str
    ) -> str:
        try:
            # Get current market conditions

            if bullish_engulfing:
                return "bullish"
            elif bearish_engulfing:
                return "bearish"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error determining market condition for {symbol}: {str(e)}")
            return "unknown"

    @staticmethod
    def is_profit_target_reached(
        side: OrderSide,
        current_price: Decimal,
        average_price: Decimal,
        profit_target: Decimal,
    ) -> bool:
        if side == OrderSide.BUY:
            logger.info("💰💰💰THIS PROFITS QUICK ON BUY SIDE💰💰💰")
            return current_price >= average_price * (Decimal(1) + profit_target)
        else:
            logger.info("💰💰💰THIS PROFITS QUICK ON SELL SIDE💰💰💰")
            return current_price <= average_price * (Decimal(1) - profit_target)

    @staticmethod
    def is_reversal_pattern_valid(
        side: OrderSide, bullish_engulfing: bool, bearish_engulfing: bool
    ) -> bool:
        logger.info("THIS REVERSAL QUICK")
        return (side == OrderSide.BUY and bearish_engulfing) or (
            side == OrderSide.SELL and bullish_engulfing
        )

    @staticmethod
    def is_trailing_stop_triggered(
        side: OrderSide,
        current_price: Decimal,
        average_price: Decimal,
        stop_loss: Decimal,
    ) -> bool:
        if side == OrderSide.BUY:
            return current_price <= average_price * (Decimal(1) - stop_loss)
        else:
            return current_price >= average_price * (Decimal(1) + stop_loss)


class PositionTracker:
    def __init__(
        self,
        symbol: str,
        order: Any,
        client: Any,
        tracker: Any,
        execute_order,
        symbol_info,
        current_price: Decimal,
        sell_price: Decimal,
        buy_price: Decimal,
        stop_loss: Decimal,
        profit_target: Decimal,
        rate_limiter: rate_limited,
        strategy: TradingStrategy,
    ):
        self.symbol = symbol
        self.client = client
        self.config = CONFIG
        self.order = order
        self.tracker = tracker
        self.execute_order = execute_order
        self.current_price: Decimal = current_price
        self.sell_price: Decimal = sell_price
        self.buy_price: Decimal = buy_price
        self.stop_loss: Decimal = stop_loss
        self.profit_target: Decimal = profit_target
        self.cancel_open_orders = cancel_open_orders
        self.orders: Dict[int, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.get_symbol_info: Dict = get_symbol_info
        self.symbol_info = symbol_info
        self.buy_queue: Dict[str, OrderQueue] = {}
        self.sell_queue: Dict[str, OrderQueue] = {}
        self.monitoring_tasks: Dict[int, asyncio.Task] = {}
        self.rate_limiter = RateLimiter(CONFIG["initial_rate"], CONFIG["max_rate"])
        self.lock = asyncio.Lock()
        self.strategy = strategy
        self.order_semaphore = Semaphore(10)  # Limit to 10 concurrent order processing
        self.trailing_stop_percentage = CONFIG.get(
            "trailing_stop_percentage", Decimal("0.02")
        )
        self.profit_percentage = CONFIG.get("profit_target", Decimal("0.05"))
        self.partial_take_profit_levels = CONFIG.get("partial_take_profit_levels", [])
        self.max_position_size = CONFIG.get("max_position_size", Decimal("1000"))
        self.exit_attempt = ExitAttempt()
        self.order_lock = asyncio.Lock()
        self.update_position_lock = asyncio.Lock()  # Separate lock for update_position

        # logger.debug(f"Initializing PositionTracker for {self.symbol_info}")

    async def initialize(
        self, orders: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Initializes the PositionTracker with historical and open order data.

        Args:
            orders: A tuple containing two lists:
                - The first list should contain dictionaries of historical orders.
                - The second list should contain dictionaries of open orders.

        Returns:
            A list of open order dictionaries, or an empty list if there was an error.
        """
        logger.info(f"Initializing PositionTracker for {orders}")

        historical_orders, open_orders = orders

        # Process historical orders
        for order_data in orders:
            await self.update_orders(order_data)

        # Return the open orders
        return open_orders

    async def enter_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        entry_type: str,
        stop_loss: Decimal,
        take_profit: Decimal,
        delay_seconds: int = 1,
    ) -> Optional[dict]:
        try:
            # Implement delay if specified
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            # Determine order side
            side = OrderSide.BUY if entry_type == "long" else OrderSide.SELL

            # Parse symbol for assets
            quote_asset = symbol[-5:]  # Assumes 4-letter quote asset like USDT
            base_asset = symbol[:-5]

            # Calculate required balance and check against maximum position size
            required_balance = quantity * price if side == OrderSide.BUY else quantity
            max_quantity = self.max_position_size / price
            quantity = min(quantity, max_quantity)

            # Check balance
            asset_to_check = quote_asset if side == OrderSide.BUY else base_asset
            try:
                await self.tracker.check_balance(
                    asset_to_check, required_balance, "free"
                )
            except InsufficientBalance as e:
                logger.warning(
                    f"Insufficient balance for {entry_type} position on {symbol}: {str(e)}"
                )
                return None

            # Place the order
            order_response = await place_order(
                self.client,
                symbol_info=self.symbol_info,
                is_buy=(side == OrderSide.BUY),
                quantity=quantity,
                price=price,
                minimum_lot_size=CONFIG["minimum_lot_size"],
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                stop_price=None,
            )

            if not order_response:
                raise OrderPlacementError(
                    f"Failed to enter {entry_type} position for {symbol}"
                )

            # Update local balance tracker
            if side == OrderSide.BUY:
                await self.tracker.update_balance(
                    quote_asset, -required_balance, "free"
                )
                await self.tracker.update_balance(base_asset, quantity, "locked")
            else:
                await self.tracker.update_balance(base_asset, -quantity, "free")
                await self.tracker.update_balance(
                    quote_asset, required_balance, "locked"
                )

            # Update position information
            await self.update_position(symbol, order_response)

            # Set trailing stop (if applicable)
            if self.use_trailing_stop:
                await self.set_trailing_stop(
                    symbol, price, self.trailing_stop_percentage
                )

            # Place stop loss order
            await self.place_stop_loss_order(symbol, stop_loss)

            # Place take profit order
            await self.place_take_profit_order(symbol, quantity, price, take_profit)

            logger.info(
                f"Successfully entered {entry_type} position for {symbol}: {order_response}"
            )
            return order_response

        except ClientError as e:
            logger.error(
                f"Binance API error while entering position for {symbol}: {str(e)}"
            )
            return None
        except OrderPlacementError as e:
            logger.error(str(e))
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error while entering position for {symbol}: {str(e)}",
                exc_info=True,
            )
            return None

    async def exit_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        exit_type: str,
        max_slippage: Decimal = Decimal("0.0002"),
    ) -> Optional[Order]:
        try:
            # Fetch current price and check cooldown period concurrently
            self.current_price
            can_attempt = self.exit_attempt.can_attempt()

            if not can_attempt:
                logger.info(
                    f"Cooldown period active for {exit_type} {symbol}. Skipping exit attempt."
                )
                return None

            min_acceptable_price = (price * quantity) * (Decimal("1") - max_slippage)

            if self.current_price < min_acceptable_price:
                logger.warning(
                    f"Current price {self.current_price} is below the minimum acceptable price {min_acceptable_price}"
                )
                return None

            logger.info(f"Attempting to {exit_type} position for {symbol}")

            position = self.positions.get(symbol)
            if not position or not position.is_open or position.side != OrderSide.BUY:
                logger.warning(f"No open LONG position found for {symbol}")
                return None

            actual_quantity = position.net_quantity
            if actual_quantity == 0:
                logger.warning(
                    f"Position quantity for {exit_type} {symbol} is zero. Skipping exit."
                )
                return None

            entry_price = position.average_price
            pnl_percentage = (
                (self.current_price - entry_price) / entry_price * Decimal("100")
            )
            logger.info(
                f"Potential P/L for {exit_type}: {entry_price} ({pnl_percentage:.2f}%)"
            )

            await self.rate_limiter.wait()
            close_response = await self.close_position(
                symbol, actual_quantity, price, exit_type
            )

            if not close_response:
                logger.error(f"Failed to close {exit_type} position for {symbol}")
                return None

            logger.info(f"Closed {exit_type} position: {close_response}")
            actual_exit_price = Decimal(close_response.price)
            actual_pnl_percentage = (
                (actual_exit_price - entry_price) / entry_price * Decimal("100")
            )
            logger.info(
                f"Actual P/L for {exit_type}: {actual_exit_price} ({actual_pnl_percentage:.2f}%)"
            )

            return close_response

        except ClientError as e:
            logger.error(f"Binance API error while exiting position: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error in exit_position for {symbol}: {e}", exc_info=True
            )
            await self.reset_failed_close_position(symbol)

        return None

    async def update_orders(self, orders: List[Dict[str, Any]]):
        for order_data in orders:
            await self.update_order(order_data)

    async def update_order(self, order_data: Union[Dict[str, Any], Order]) -> None:
        try:
            logger.info(f"Order Data Type: {type(order_data)}, {order_data}")

            start_time = time.perf_counter()

            if not isinstance(order_data, (dict, Order)):
                logger.error(f"Unsupported order data type: {type(order_data)}")
                return

            order_obj = (
                order_data
                if isinstance(order_data, Order)
                else Order.from_dict(order_data)
            )

            async with self.order_lock:
                if order_obj.order_id not in self.orders:
                    self.orders[order_obj.order_id] = order_obj
                else:
                    # Update existing order
                    existing_order = self.orders[order_obj.order_id]
                    existing_order.status = order_obj.status
                    existing_order.quantity = order_obj.quantity
                    existing_order.price = order_obj.price
                    existing_order.time = order_obj.time

                    if existing_order.status != order_obj.status:
                        logger.info(
                            f"Updated order {order_obj.order_id}: Status changed from {existing_order.status} to {order_obj.status}"
                        )

            # Move update_position outside of the order_lock
            await self.update_position(order_obj.symbol, order_obj)

            # Batch logging
            log_messages = [
                f"Updated position for {order_obj.symbol}: {self.positions.get(order_obj.symbol)}",
                f"Order processing took {time.perf_counter() - start_time:.2f} seconds",
            ]
            logger.info("\n".join(log_messages))

        except Exception as e:
            logger.error(f"Error updating order: {e}", exc_info=True)
            logger.error(f"Problematic order data: {order_data}")

    async def close_position(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        close_type: str = "take_profit",
    ) -> Optional[Order]:
        await asyncio.sleep(1)

        try:
            buy_entries = list(self.buy_queue[symbol].queue)

            if not buy_entries:
                logger.warning(
                    f"No open buy orders found for {symbol}. Skipping close."
                )
                return None

            close_quantity = min(quantity, sum(entry[1] for entry in buy_entries))
            if close_quantity == Decimal("0"):
                logger.warning(
                    f"Position quantity for {symbol} is zero. Skipping close."
                )
                return None

            position = self.positions.get(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}. Cannot close.")
                return None

            close_side = (
                OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            )
            quantity = round_decimal(close_quantity, CONFIG["decimal_places"])

            current_price = Decimal(self.current_price)

            if close_type == "market":
                close_price = current_price
                order_type = OrderType.MARKET
                time_in_force = TimeInForce.GTC
            elif close_type == "limit":
                close_price = max(self.sell_price, price)
                order_type = OrderType.LIMIT
                time_in_force = TimeInForce.GTC
            elif close_type in ["stop_loss", "take_profit"]:
                close_price = price
                order_type = (
                    OrderType.STOP_LOSS_LIMIT
                    if close_type == "stop_loss"
                    else OrderType.TAKE_PROFIT_LIMIT
                )
                time_in_force = TimeInForce.GTC
            else:
                logger.warning(f"Unknown close type: {close_type}. Using market price.")
                close_price = current_price
                order_type = OrderType.MARKET
                time_in_force = None

            close_price = round_decimal(close_price, CONFIG["decimal_places"])

            logger.info(
                f"Attempting to close position: symbol={symbol}, side={close_side}, quantity={quantity}, price={close_price}, type={close_type}"
            )

            stop_price = (
                close_price if close_type in ["stop_loss", "take_profit"] else None
            )

            response = await place_order(
                self.client,
                self.symbol_info,
                is_buy=(close_side == OrderSide.BUY),
                quantity=quantity,
                price=float(close_price),
                minimum_lot_size=CONFIG["minimum_lot_size"],
                order_type=order_type,
                time_in_force=time_in_force,
                stop_price=stop_price,
            )

            if response:
                order = (
                    Order.from_dict(response)
                    if isinstance(response, dict)
                    else response
                )
                await self.update_position(symbol, order)
                self.orders[order.order_id] = order
                await self.start_monitoring(symbol, order.order_id)
                logger.info(f"Close order placed successfully: {response}")
                return order
            else:
                logger.error(
                    f"Failed to close position: {symbol}, {quantity}, {close_price}"
                )
                await self.reset_monitors_and_orders(symbol)
                return None

        except Exception as error:
            logger.error(
                f"Error in close_position for {symbol}: {str(error)}", exc_info=True
            )
            await self.reset_monitors_and_orders(symbol)
            return None

    async def get_open_positions(self, symbol: str) -> List[Position]:
        try:
            async with self.lock:
                position = self.positions.get(symbol)
                open_orders = [
                    order
                    for order in self.orders.values()
                    if order.symbol == symbol and order.status == OrderStatus.NEW
                ]
                logger.info(f"Open orders for {symbol}: {open_orders}")
                if position and position.net_quantity != 0:
                    logger.debug(f"Open position for {symbol}: {position}")
                    return [position]
                elif open_orders:
                    total_quantity = sum(
                        Decimal(order.quantity) for order in open_orders
                    )
                    average_price = (
                        sum(
                            Decimal(order.price) * Decimal(order.quantity)
                            for order in open_orders
                        )
                        / total_quantity
                    )
                    open_position = Position(
                        symbol, total_quantity, average_price, open_orders[0].side
                    )
                    logger.debug(
                        f"Open position (from open orders) for {symbol}: {open_position}"
                    )
                    return [open_position]
                else:
                    logger.debug(f"No open position for {symbol}")
                    return []

        except Exception as e:
            logger.error(f"Error getting open positions for {symbol}: {e}")
            return []

    #############start
    # @rate_limited
    async def get_open_orders(self, symbol):
        try:
            raw_orders = await get_orders(self.client, symbol)
            logger.debug(f"Raw orders from exchange: {raw_orders}")

            orders = []
            for order_data in raw_orders:
                try:
                    order = Order.from_dict(order_data)
                    orders.append(order)
                except Exception as e:
                    logger.error(f"Error converting order data to Order object: {e}")
                    logger.debug(f"Problematic order data: {order_data}")
                    # Continue processing other orders

            logger.debug(f"Converted orders: {orders}")
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders for {symbol}: {str(e)}")
            return []

    async def sync_with_exchange(self):
        try:
            logger.debug("Fetching open orders from exchange")
            exchange_orders = await self.get_open_orders(self.symbol)
            logger.debug(f"Fetched {len(exchange_orders)} open orders")

            logger.debug("Updating local order tracking")
            for order in exchange_orders:
                logger.debug(f"Processing order: {type(order)}")
                if isinstance(order, dict):
                    logger.warning(
                        f"Received dictionary instead of Order object: {order}"
                    )
                    order = Order.from_dict(order)

                logger.debug(f"Order ID: {order.order_id}")
                self.orders[order.order_id] = order

            logger.debug(f"Updated {len(exchange_orders)} orders")

            # Remove any local orders not present on exchange
            local_order_ids = set(self.orders.keys())
            exchange_order_ids = set()
            for order in exchange_orders:
                if isinstance(order, dict):
                    logger.warning(f"Unexpected dict in exchange_orders: {order}")
                    order = Order.from_dict(order)
                exchange_order_ids.add(order.order_id)

            for order_id in local_order_ids - exchange_order_ids:
                del self.orders[order_id]

            # Fetch account information
            # account_info = self.client.account()

            for asset, balance in self.tracker.balances.items():
                asset = balance["asset"]
                free = round_decimal(Decimal(balance["free"]), CONFIG["precision"])
                locked = round_decimal(Decimal(balance["locked"]), CONFIG["precision"])
                total = free + locked

                if total > Decimal("1"):  # Process only if there is a balance
                    # Determine the side based on the locked amount
                    side = OrderSide.BUY if locked > Decimal("0") else None

                    if asset not in self.positions:
                        logger.info(
                            f"Adding position for {asset}: {total} ({free} free, {locked} locked)"
                        )

                        # Calculate average price based on open orders
                        average_price = (
                            await self.calculate_average_price_from_open_orders(asset)
                        )
                        asset = (
                            self.symbol
                            if (asset == "FDUSD" or "DOGE" or "SHIB" or "USDT")
                            else asset
                        )
                        self.positions[asset] = Position(
                            symbol=asset,
                            net_quantity=total,
                            average_price=average_price,
                            side=side,
                            entry_price=average_price,
                            absolute_quantity=abs(total),
                            additional_info={},
                        )
                    else:
                        logger.info(
                            f"Updating position for {asset}: {total} ({free} free, {locked} locked)"
                        )
                        self.positions[asset].net_quantity = total
                        self.positions[asset].absolute_quantity = abs(total)

            logger.info(f"Synced with exchange for {self.symbol}")

        except Exception as e:
            logger.error(f"Error syncing with exchange: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(
                f"Error traceback: {''.join(traceback.format_tb(e.__traceback__))}"
            )
            raise

    async def calculate_average_price_from_open_orders(self, asset):
        open_orders = await self.get_open_orders(self.symbol)
        relevant_orders = [
            order for order in open_orders if order.symbol.startswith(asset)
        ]

        if not relevant_orders:
            return Decimal("0")

        total_quantity = sum(order.quantity for order in relevant_orders)
        weighted_sum = sum(order.price * order.quantity for order in relevant_orders)

        return weighted_sum / total_quantity if total_quantity > 0 else Decimal("0")

    async def update_position(self, symbol: str, order: Order):
        async with self.lock:
            try:
                if symbol not in self.buy_queue:
                    self.buy_queue[symbol] = OrderQueue()
                    self.sell_queue[symbol] = OrderQueue()
                    logger.info(f"Created new order queues for {symbol}")

                logger.info(f"Updating position for {symbol}: {order}")
                executed_quantity = Decimal(order.executed_quantity)
                remaining_quantity = Decimal(str(order.quantity)) - executed_quantity

                position_change = Decimal("0")
                avg_price_change = Decimal("0")
                removed: List[Tuple[int, Decimal, Decimal]] = []

                if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    try:
                        if order.side == OrderSide.BUY:
                            queue_to_remove_from = self.sell_queue[symbol]
                            queue_to_add_to = self.buy_queue[symbol]
                        else:  # SELL
                            queue_to_remove_from = self.buy_queue[symbol]
                            queue_to_add_to = self.sell_queue[symbol]

                        # Check if the queue has enough quantity before removing
                        if queue_to_remove_from.total_quantity() < executed_quantity:
                            logger.warning(
                                f"Insufficient quantity in {order.side} queue for {symbol}. Adjusting execution."
                            )
                            executed_quantity = queue_to_remove_from.total_quantity()

                        removed = queue_to_remove_from.remove(executed_quantity)
                        closed_position = sum(r[1] for r in removed)
                        position_change = (
                            closed_position
                            if order.side == OrderSide.BUY
                            else -closed_position
                        )

                        for removed_order in removed:
                            queue_to_add_to.add(
                                removed_order[0],
                                removed_order[1],
                                Decimal(str(order.price)),
                            )
                            price_diff = (
                                Decimal(str(order.price)) - removed_order[2]
                                if order.side == OrderSide.BUY
                                else removed_order[2] - Decimal(str(order.price))
                            )
                            avg_price_change += removed_order[1] * price_diff

                        remaining = executed_quantity - closed_position
                        if remaining > 0:
                            queue_to_add_to.add(
                                int(order.order_id),
                                remaining,
                                Decimal(str(order.price)),
                            )
                            position_change += (
                                remaining if order.side == OrderSide.BUY else -remaining
                            )
                            avg_price_change += (
                                remaining * Decimal(str(order.price))
                                if order.side == OrderSide.BUY
                                else -remaining * Decimal(str(order.price))
                            )

                            # Create a position for the opposite side when order is filled
                            opposite_queue = (
                                self.sell_queue[symbol]
                                if order.side == OrderSide.BUY
                                else self.buy_queue[symbol]
                            )
                            opposite_queue.add(
                                int(order.order_id),
                                remaining,
                                Decimal(str(order.price)),
                            )

                        logger.info(f"Position change for {symbol}: {position_change}")
                        logger.info(
                            f"Average price change for {symbol}: {avg_price_change}"
                        )

                    except ValueError as e:
                        logger.error(f"Error updating position for {symbol}: {e}")
                        # Handle the error appropriately, maybe by adjusting the execution quantity
                        removed = []
                        position_change = Decimal("0")
                        avg_price_change = Decimal("0")

                elif order.status == OrderStatus.CANCELED:
                    logger.info(f"Order canceled: {order.order_id} for {symbol}")
                    queue = (
                        self.buy_queue[symbol]
                        if order.side == OrderSide.BUY
                        else self.sell_queue[symbol]
                    )
                    removed_order = queue.remove_order(int(order.order_id))

                    if removed_order:
                        _, canceled_quantity, canceled_price = removed_order
                        removed = [removed_order]
                        logger.info(
                            f"Removed canceled order from queue for {symbol}: {canceled_quantity} {order.side}"
                        )

                        # Return the canceled quantity to the respective queue
                        if order.side == OrderSide.BUY:
                            self.buy_queue[symbol].add(
                                int(order.order_id), canceled_quantity, canceled_price
                            )
                        else:
                            self.sell_queue[symbol].add(
                                int(order.order_id), canceled_quantity, canceled_price
                            )

                        # Adjust the position
                        position_change = (
                            -canceled_quantity
                            if order.side == OrderSide.BUY
                            else canceled_quantity
                        )
                        avg_price_change = (
                            -canceled_quantity * canceled_price
                            if order.side == OrderSide.BUY
                            else canceled_quantity * canceled_price
                        )

                        logger.info(
                            f"Returned canceled order to {order.side} queue for {symbol}: {canceled_quantity}"
                        )

                        # Update the orders dictionary
                        self.orders.pop(order.order_id, None)
                    else:
                        logger.warning(
                            f"Canceled order not found in queue: {order.order_id} for {symbol}"
                        )

                await self._recalculate_position(symbol)

                # Calculate realized PNL only for filled or partially filled orders
                realized_pnl = Decimal("0")
                if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    realized_pnl = self._calculate_realized_pnl(symbol, removed, order)
                    logger.info(f"Realized PNL for {symbol}: {realized_pnl}")

                # Update position information
                if symbol not in self.positions:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        net_quantity=Decimal("0"),
                        average_price=Decimal("0"),
                        side=None,
                        entry_price=Decimal("0"),
                        absolute_quantity=Decimal("0"),
                        additional_info={},
                    )

                position = self.positions[symbol]
                position.net_quantity += position_change
                position.absolute_quantity = abs(position.net_quantity)
                if position.absolute_quantity != 0:
                    position.average_price = (
                        position.average_price
                        * (position.absolute_quantity - abs(position_change))
                        + avg_price_change
                    ) / position.absolute_quantity
                else:
                    position.average_price = Decimal("0")
                position.side = (
                    OrderSide.BUY
                    if position.net_quantity > 0
                    else OrderSide.SELL
                    if position.net_quantity < 0
                    else None
                )

                # Create opposite position with a new order ID
                opposite_side = (
                    OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
                )
                opposite_queue = (
                    self.sell_queue[symbol]
                    if order.side == OrderSide.BUY
                    else self.buy_queue[symbol]
                )

                # Generate a new order ID for the opposite position
                new_order_id = await self._generate_new_order_id()

                opposite_queue.add(
                    new_order_id,
                    executed_quantity,
                    Decimal(str(order.price)),
                )
                logger.info(
                    f"Created opposite {opposite_side} position for {symbol}: {executed_quantity} @ {order.price} with new order ID: {new_order_id}"
                )

                # Create a new order object for the opposite position
                new_order_kwargs = {
                    "order_id": str(new_order_id),
                    "symbol": symbol,
                    "side": opposite_side,
                    "quantity": str(executed_quantity),
                    "price": str(order.price),
                    "status": OrderStatus.NEW,  # Set initial status as NEW
                    "type": order.type
                    if hasattr(order, "type")
                    else OrderType.LIMIT,  # Use original order type or default to LIMIT
                }

                # Add optional fields if they exist in the original order
                if hasattr(order, "time_in_force"):
                    new_order_kwargs["time_in_force"] = order.time_in_force

                new_order = Order(**new_order_kwargs)

                # Add the new order to the orders dictionary
                self.orders[str(new_order_id)] = new_order

                logger.info(f"Updated position for {symbol}: {position}")

            except Exception as e:
                logger.error(
                    f"Unexpected error updating position for {symbol}: {str(e)}"
                )
                logger.error(traceback.format_exc())

    async def _generate_new_order_id(self):
        # This is a simple implementation. You might want to use a more robust method
        # depending on your system's requirements
        return int(time.time() * 1000)  # Millisecond timestamp as order ID

    async def _handle_take_profit_filled(self, symbol: str, order: Order):
        # Close the position or update it accordingly
        position = self.positions.get(symbol)
        if position:
            close_quantity = min(position.absolute_quantity, order.executed_quantity)
            await self.close_position(
                symbol, close_quantity, order.price, "take_profit"
            )

        # Remove the filled take-profit order from tracking
        self._remove_order_from_tracking(symbol, order)

    def _remove_order_from_tracking(self, symbol: str, order: Order):
        if order.order_id in self.orders:
            del self.orders[order.order_id]

        queue = (
            self.buy_queue[symbol]
            if order.side == OrderSide.BUY
            else self.sell_queue[symbol]
        )
        queue.remove(order.quantity)

    async def start_monitoring(self, symbol: str, order_id: int) -> None:
        if order_id not in self.monitoring_tasks:
            task = asyncio.create_task(self.monitor_order_status(symbol, order_id))
            self.monitoring_tasks[order_id] = task

    async def monitor_order_status(self, symbol: str, order_id: int) -> None:
        try:
            while True:
                await self.rate_limiter.wait()
                order = self.orders.get(order_id)

                if order is None:
                    logger.warning(f"Order {order_id} for {symbol} not found")
                    break

                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED):
                    logger.info(
                        f"Order {order_id} {order.status.value.lower()} for {symbol}."
                    )
                    await self.update_position(symbol, order)
                    break

                updated_order = await self.get_order_from_exchange(symbol, order_id)
                if updated_order:
                    async with self.lock:
                        self.orders[order_id] = updated_order
                    logger.info(
                        f"Updated order {order_id} {updated_order.status.value.lower()} for {symbol}."
                    )

                    if updated_order.status in (
                        OrderStatus.FILLED,
                        OrderStatus.CANCELED,
                        OrderStatus.PARTIALLY_FILLED,
                    ):
                        await self.update_position(symbol, updated_order)
                        break

                await asyncio.sleep(0.5)  # Wait for 5 seconds before checking again
        except Exception as error:
            logger.error(
                f"Error monitoring order {order_id} for {symbol}: {str(error)}"
            )
        finally:
            self.monitoring_tasks.pop(order_id, None)

    async def get_order_from_exchange(
        self, symbol: str, order_id: int
    ) -> Optional[Order]:
        try:
            await self.rate_limiter.wait()
            order_data = self.client.get_order(symbol=symbol, orderId=order_id)
            return Order.from_dict(order_data)
        except ClientError as e:
            logger.error(f"Error fetching order {order_id} for {symbol}: {e}")
            return None

    async def get_position_value(self, symbol: str) -> Decimal:
        async with self.lock:
            position = self.positions.get(symbol)
            if position:
                return round_decimal(
                    position.net_quantity * position.average_price,
                    CONFIG["decimal_places"],
                )
            return Decimal("0")

    # @rate_limited
    async def close_all_positions(self, symbol: str) -> List[Order]:
        open_positions = await self.get_open_positions(symbol)
        close_tasks = [
            self.close_position(
                symbol, position.net_quantity, position.average_price, "all"
            )
            for position in open_positions
        ]
        closed_orders = await asyncio.gather(*close_tasks, return_exceptions=True)
        successfully_closed = [
            order for order in closed_orders if isinstance(order, Order)
        ]
        logger.info(
            f"Closed {len(successfully_closed)} out of {len(open_positions)} positions for {symbol}"
        )
        return successfully_closed

    async def emergency_closure(self, symbol: str) -> Tuple[List[Order], List[Order]]:
        logger.warning(f"Initiating emergency closure for {symbol}")
        try:
            cancel_task = asyncio.create_task(
                self.cancel_open_orders(self.client, symbol)
            )
            close_task = asyncio.create_task(self.close_all_positions(symbol))

            cancelled_orders, closed_positions = await asyncio.gather(
                cancel_task, close_task
            )

            logger.info(
                f"Emergency closure completed for {symbol}. Cancelled orders: {len(cancelled_orders)}, Closed positions: {len(closed_positions)}"
            )
            return cancelled_orders, closed_positions
        except Exception as error:
            logger.error(f"Error during emergency closure for {symbol}: {str(error)}")
            return [], []

    async def set_trailing_stop(self, symbol, current_price, trailing_percentage):
        position = self.positions.get(symbol)

        if not position:
            return

        if position.side == OrderSide.BUY:
            limit_price = current_price * (1 - trailing_percentage)  # Corrected logic
        else:  # position.side == OrderSide.SELL
            # 1. Calculate Potential New Profit Target
            new_profit_target = current_price * (
                1 + trailing_percentage
            )  # Trailing for SELL

            # 2. Update Only If More Favorable
            self.profit_target = max(self.profit_target, new_profit_target)

            # 3. Set limit_price to the adjusted profit_target
            limit_price = self.profit_target

        trailing_stop_order = await self.close_position(
            symbol, position.net_quantity, limit_price, "stop_loss"
        )

        if trailing_stop_order:
            logger.info(
                f"Placed trailing stop order for {symbol}: {trailing_stop_order}"
            )
        else:
            logger.error(f"Failed to place trailing stop order for {symbol}")

    # @rate_limited
    async def monitor_positions(
        self,
        symbol: str,
        bullish_engulfing: bool,
        bearish_engulfing: bool,
    ) -> None:
        while True:
            try:
                await self.get_current_price(symbol)
                for symbol, position in self.positions.items():
                    if position.absolute_quantity > 0:
                        try:
                            logger.info(f"Checking position for {symbol}")
                            await self.check_position_status(
                                symbol, position, bullish_engulfing, bearish_engulfing
                            )
                        except Exception as e:
                            logger.error(
                                f"Error checking position status for {symbol}: {str(e)}",
                                exc_info=True,
                            )
                    else:
                        logger.info(
                            f"Skipping position check for {symbol} as quantity is 0"
                        )

                # Sleep for a configured interval before the next check
                # await asyncio.sleep(CONFIG["position_check_interval"])
                break
            except asyncio.CancelledError:
                logger.info("Position monitoring task was cancelled")
                await asyncio.sleep(CONFIG["error_retry_interval"])
                break
            except Exception as e:
                logger.error(f"Error in position monitoring: {str(e)}", exc_info=True)
                await asyncio.sleep(CONFIG["error_retry_interval"])
                break

    # @rate_limited
    async def get_current_price(self, symbol: str) -> Decimal:
        try:
            if not self.current_price:
                # Update the current price
                logger.info("Updating current price")
                await self.sync_with_exchange()

            df = await get_price_symbol(self.symbol)

            # self.current_price = df  # updating Price Globally.
            logger.info(f"Current price: {self.current_price} for {symbol}")
            return round_decimal(df, CONFIG["precision"])

        except Exception as e:
            logger.error(
                f"Error getting current price for {symbol}: {str(e)}", exc_info=True
            )
            return Decimal("0")  # Return a default value to prevent further errors

    # @rate_limited
    async def check_position_status(
        self,
        symbol: str,
        position: Position,
        bullish_engulfing: bool,
        bearish_engulfing: bool,
    ) -> List[Optional[Order]]:
        try:
            if position.net_quantity == Decimal("0"):
                logger.warning(
                    f"No open position for {symbol}. Skipping position check."
                )
                return []

            if not await self.is_position_size_valid(
                symbol, abs(position.net_quantity)
            ):
                logger.warning(
                    f"Invalid position size for {symbol}. Skipping status check."
                )
                return []

            results = []
            entries = list(
                self.buy_queue[symbol].queue
                if position.side == OrderSide.BUY
                else self.sell_queue[symbol].queue
            )
            market_condition = await self.strategy.get_market_condition(
                bullish_engulfing, bearish_engulfing, symbol
            )

            for entry_order_id, entry_quantity, entry_price in entries:
                pnl_percentage = self.calculate_pnl_percentage_for_entry(
                    entry_price, self.current_price, position.side
                )
                logger.info(
                    f"Current P/L for {symbol} entry {entry_order_id}: {pnl_percentage:.2f}%"
                )

                result = await self.check_exit_conditions(
                    symbol,
                    entry_quantity,
                    entry_price,
                    position.side,
                    pnl_percentage,
                    market_condition,
                    bullish_engulfing,
                    bearish_engulfing,
                )
                if result:
                    results.append(result)

            return results

        except asyncio.CancelledError:
            logger.warning(f"Position check for {symbol} was cancelled")
            return []
        except Exception as e:
            logger.error(f"Error checking position status for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    async def check_exit_conditions(
        self,
        symbol: str,
        quantity: Decimal,
        entry_price: Decimal,
        side: OrderSide,
        pnl_percentage: Decimal,
        market_condition: str,
        bullish_engulfing: bool,
        bearish_engulfing: bool,
    ) -> Optional[Order]:
        conditions: List[Tuple[str, bool]] = [
            ("Risk management", await self.is_position_size_valid(symbol, quantity)),
            (
                "Profit target",
                self.strategy.is_profit_target_reached(
                    side, self.current_price, entry_price, self.profit_target
                ),
            ),
            (
                "Quick profit",
                (side == OrderSide.BUY and pnl_percentage >= Decimal("0.3"))
                or (side == OrderSide.SELL and pnl_percentage <= Decimal("-0.3")),
            ),
            (
                "Significant profit",
                (side == OrderSide.BUY and pnl_percentage >= Decimal("0.5"))
                or (side == OrderSide.SELL and pnl_percentage <= Decimal("-0.5")),
            ),
            (
                "Minimize loss",
                (side == OrderSide.BUY and pnl_percentage <= Decimal("-0.3"))
                or (side == OrderSide.SELL and pnl_percentage >= Decimal("0.3")),
            ),
            (
                "Reversal pattern",
                self.strategy.is_reversal_pattern_valid(
                    side, bullish_engulfing, bearish_engulfing
                ),
            ),
            # (
            #     "Stop loss",
            #     self.strategy.is_stop_loss_triggered(
            #         side, self.current_price, entry_price, self.stop_loss
            #     ),
            # ),
            # (
            #     "Trailing stop",
            #     self.strategy.is_trailing_stop_triggered(
            #         side, self.current_price, entry_price, self.trailing_stop_percentage
            #     ),
            # ),
            (
                "Unfavorable market condition",
                not self.is_market_condition_favorable(market_condition, side),
            ),
        ]

        for reason, condition in conditions:
            if condition:
                return await self.close_position_with_logging(
                    symbol,
                    quantity,
                    self.current_price,
                    reason,
                    self.get_exit_type(reason),
                )

        return None

    def get_exit_type(self, reason: str) -> str:
        if reason in ["Stop loss", "Trailing stop"]:
            return "stop_loss"
        elif reason in ["Profit target", "Quick profit", "Significant profit"]:
            return "take_profit"
        else:
            return "limit"

    def is_market_condition_favorable(
        self, market_condition: str, side: OrderSide
    ) -> bool:
        return (side == OrderSide.BUY and market_condition == "bullish") or (
            side == OrderSide.SELL and market_condition == "bearish"
        )

    async def close_position_with_logging(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        reason: str,
        exit_type: str,
    ) -> Optional[Order]:
        logger.info(
            f"Attempting to close position due to {reason}: {symbol}, {quantity}, {price}"
        )
        close_response = await self.close_position(symbol, quantity, price, exit_type)
        if close_response:
            logger.info(f"{reason} triggered. Closed position: {close_response}")
            return close_response
        else:
            logger.error(
                f"Failed to close position on {reason.lower()}: {symbol}, {quantity}, {price}"
            )
            return None

    async def place_stop_loss_order(self, symbol: str, stop_price: Decimal):
        position = self.positions.get(symbol)
        if not position:
            logger.warning(
                f"No position found for {symbol}. Cannot place stop loss order."
            )
            return None

        quantity = abs(position.net_quantity)
        side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

        # Calculate the limit price slightly below/above the stop price
        limit_price = (
            Decimal(self.profit_target)
            if side == OrderSide.SELL
            else Decimal(stop_price)
        )
        limit_price = round_decimal(limit_price, CONFIG["precision"])

        return await self.close_position(symbol, quantity, limit_price, "stop_loss")

    async def find_stop_loss_order(self, symbol: str) -> Optional[Order]:
        try:
            open_orders = await self.get_open_orders(symbol)
            for order in open_orders:
                if order["type"] == "STOP_LOSS_LIMIT":
                    return Order.from_dict(order)
            return None
        except Exception as e:
            logger.error(f"Error finding stop loss order for {symbol}: {str(e)}")
            return None

    async def update_stop_loss_order(self, symbol: str, new_stop_loss: Decimal):
        try:
            # Find the existing stop loss order
            existing_stop_loss_order = await self.find_stop_loss_order(symbol)

            if existing_stop_loss_order:
                # Cancel the existing stop loss order
                await self.cancel_order(symbol, existing_stop_loss_order.order_id)
                logger.info(f"Cancelled existing stop loss order for {symbol}")

            # Place a new stop loss order
            position = self.positions.get(symbol)
            if not position:
                logger.warning(
                    f"No position found for {symbol}. Cannot update stop loss."
                )
                return

            quantity = abs(position.net_quantity)
            # side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

            # use close_position_with_logging instead of place_order

            new_stop_loss_order = await self.close_position_with_logging(
                symbol,
                quantity,
                self.current_price,
                "Reversal pattern",
                self.get_exit_type("Reversal pattern"),
            )
            if new_stop_loss_order:
                logger.info(
                    f"Placed new stop loss order for {symbol} at {new_stop_loss}"
                )
                # Update the order in our tracking
                await self.update_order(new_stop_loss_order)
            else:
                logger.error(f"Failed to place new stop loss order for {symbol}")

        except Exception as e:
            logger.error(f"Error updating stop loss order for {symbol}: {str(e)}")

    ####double check from here

    def calculate_pnl_percentage_for_entry(
        self, entry_price: Decimal, current_price: Decimal, side: OrderSide
    ) -> Decimal:
        if entry_price == 0:
            logger.info(
                f"Entry price is equal {entry_price}, returning 0 which may cause decimal.DivisionByZero"
            )
            return 0  # Or handle appropriately (log, raise exception)

        if side == OrderSide.BUY:
            return (current_price - entry_price) / entry_price * 100
        else:
            return (entry_price - current_price) / entry_price * 100

    def calculate_take_profit_quantity(
        self, entry_quantity: Decimal, pnl_percentage: Decimal
    ) -> Decimal:
        # logic for partial take profit here

        for profit_level, sell_percentage in self.partial_take_profit_levels:
            if pnl_percentage >= profit_level:
                return entry_quantity * sell_percentage

        return Decimal("0")

    async def update_trailing_stop(
        self, symbol: str, current_price: Decimal, quantity: Decimal
    ):
        position = self.positions.get(symbol)
        if not position:
            logger.warning(
                f"No position found for {symbol}. Cannot update trailing stop."
            )
            return

        # Calculate the new stop price
        if position.side == OrderSide.BUY:
            new_stop_price = current_price * (1 - self.trailing_stop_percentage)
        else:
            new_stop_price = current_price * (1 + self.trailing_stop_percentage)

        new_stop_price = round_decimal(new_stop_price, CONFIG["precision"])

        # Find existing stop loss order
        stop_loss_order = await self.find_stop_loss_order(symbol)

        if stop_loss_order:
            # If the new stop price is more favorable, update the stop loss
            if (
                position.side == OrderSide.BUY
                and new_stop_price > Decimal(stop_loss_order.stop_price)
            ) or (
                position.side == OrderSide.SELL
                and new_stop_price < Decimal(stop_loss_order.stop_price)
            ):
                await self.cancel_order(symbol, stop_loss_order.order_id)
                new_stop_loss_order = await self.place_stop_loss_order(
                    symbol, new_stop_price
                )
                if new_stop_loss_order:
                    logger.info(
                        f"Updated trailing stop for {symbol} to {new_stop_price}"
                    )
                else:
                    logger.error(f"Failed to update trailing stop for {symbol}")
        else:
            # If no stop loss exists, create a new one
            new_stop_loss_order = await self.place_stop_loss_order(
                symbol, new_stop_price
            )
            if new_stop_loss_order:
                logger.info(
                    f"Placed new trailing stop for {symbol} at {new_stop_price}"
                )
            else:
                logger.error(f"Failed to place new trailing stop for {symbol}")

    async def set_tighter_stop_loss(
        self, symbol: str, position: Position, entry_price: Decimal
    ) -> None:
        try:
            new_stop_loss = round_decimal(entry_price, CONFIG["precision"]) * (
                1 - Decimal("0.01")
                if position.side == OrderSide.BUY
                else 1 + Decimal("0.02")
            )
            logger.info(f"Setting tighter stop loss at {new_stop_loss} for {symbol}")
            await self.update_stop_loss_order(symbol, Decimal(new_stop_loss))

        except Exception as e:
            # if fail remove stop loss order
            await self.remove_position(symbol, position.order_id)
            logger.error(f"Error setting tighter stop loss for {symbol}: {str(e)}")

    async def remove_position(self, symbol: str, order_id: int):
        async with self.lock:
            position = self.positions.get(symbol)
            if position:
                # Remove the order from the corresponding queue
                if position.side == OrderSide.BUY:
                    self.buy_queue[symbol].remove(Decimal(order_id))
                    await self.cancel_orders(symbol, [order_id])
                else:
                    self.sell_queue[symbol].remove(Decimal(order_id))
                    await self.cancel_orders(symbol, order_id)

                # Update the position based on the remaining orders in the queues
                await self.update_position_from_queues(symbol)

    async def update_position_from_queues(self, symbol: str):
        async with self.lock:
            buy_queue = self.buy_queue.get(symbol, OrderQueue())
            sell_queue = self.sell_queue.get(symbol, OrderQueue())

            net_quantity = buy_queue.total_quantity() - sell_queue.total_quantity()
            new_total = buy_queue.total_price() - sell_queue.total_price()

            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol, Decimal("0"), Decimal("0"), None
                )

            position = self.positions[symbol]
            quantity_change = net_quantity - position.net_quantity
            new_average_price = (
                new_total / abs(net_quantity) if net_quantity != 0 else Decimal("0")
            )
            position.update(quantity_change, new_average_price)

            logger.info(f"Updated position for {symbol}: {position}")

    # @rate_limited
    async def cancel_order(self, symbol: str, order_id: int):
        try:
            await self.rate_limiter.wait()
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Cancelled order {order_id} for {symbol}: {result}")

            # Update position after cancelling the order
            cancelled_order = Order.from_dict(result)
            await self.update_position(symbol, cancelled_order)

        except Exception as e:
            logger.error(f"Error cancelling order {order_id} for {symbol}: {str(e)}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.cancel_all_monitoring_tasks()

    async def cancel_all_monitoring_tasks(self):
        tasks = list(self.monitoring_tasks.values())
        self.monitoring_tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def reset_failed_close_position(self, symbol: str):
        logger.warning(f"Resetting failed close position for {symbol}")
        try:
            # Cancel any existing orders for this symbol
            await self.cancel_open_orders(self.client, symbol)

            # Reset the position tracking for this symbol
            if symbol in self.positions:
                del self.positions[symbol]

            if symbol in self.buy_queue:
                self.buy_queue[symbol] = OrderQueue()

            if symbol in self.sell_queue:
                self.sell_queue[symbol] = OrderQueue()

            # Remove any monitoring tasks for this symbol
            tasks_to_cancel = [
                task
                for order_id, task in self.monitoring_tasks.items()
                if self.orders.get(order_id, Order(symbol="")).symbol == symbol
            ]
            for task in tasks_to_cancel:
                task.cancel()

            # Remove orders for this symbol from tracking
            self.orders = {
                order_id: order
                for order_id, order in self.orders.items()
                if order.symbol != symbol
            }

            logger.info(f"Reset completed for failed close position on {symbol}")
        except Exception as e:
            logger.error(
                f"Error in reset_failed_close_position for {symbol}: {str(e)}",
                exc_info=True,
            )

    async def reset_monitors_and_orders(
        self, symbol: str, failed_order_id: Optional[int] = None
    ):
        logger.warning(f"Resetting monitors and specific orders for {symbol}")

        # Cancel monitoring tasks for this symbol

        # tasks_to_cancel = [
        #     task
        #     for order_id, task in self.order_tasks.items()
        #     if self.orders.get(order_id, Order.dummy()).symbol == symbol
        # ]
        tasks_to_cancel = [
            task
            for order_id, task in self.order_tasks.items()
            if order_id in self.orders and self.orders[order_id].symbol == symbol
        ]
        for order_id, task in tasks_to_cancel:
            task.cancel()
            del self.monitoring_tasks[order_id]

        # Remove the failed order from tracking if provided
        if failed_order_id is not None and failed_order_id in self.orders:
            del self.orders[failed_order_id]

        # Fetch current open orders from the exchange
        try:
            open_orders = await self.get_open_orders(symbol)
            logger.debug(f"Open orders for {symbol}: {open_orders}")

            exchange_order_ids = set(order.order_id for order in open_orders)
            logger.debug(f"Exchange order IDs for {symbol}: {exchange_order_ids}")

            # Remove orders that no longer exist on the exchange
            self.orders = {
                order_id: order
                for order_id, order in self.orders.items()
                if order.symbol != symbol or order_id in exchange_order_ids
            }
            logger.debug(f"Updated self.orders for {symbol}: {self.orders}")

            # Update position based on remaining orders
            await self.update_position_from_queues(symbol)

        except Exception as e:
            logger.error(
                f"Error fetching open orders during reset for {symbol}: {str(e)}"
            )
            # You might want to add more specific error handling here

        logger.info(f"Reset completed for {symbol}")

    async def _recalculate_position(self, symbol: str):
        # async with self.lock:
        buy_quantity = self.buy_queue[symbol].total_quantity()
        sell_quantity = self.sell_queue[symbol].total_quantity()
        net_quantity = buy_quantity - sell_quantity

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                net_quantity=Decimal("0"),
                average_price=Decimal("0"),
                side=None,
                entry_price=Decimal("0"),
                absolute_quantity=Decimal("0"),
                additional_info={},
            )

        position = self.positions[symbol]

        # Calculate the new average price
        new_total = (
            self.buy_queue[symbol].total_price() - self.sell_queue[symbol].total_price()
        )
        new_average_price = (
            new_total / abs(net_quantity) if net_quantity != 0 else Decimal("0")
        )

        # Update the position
        position.net_quantity = net_quantity
        position.average_price = new_average_price
        position.absolute_quantity = abs(net_quantity)
        position.side = (
            OrderSide.BUY
            if net_quantity > 0
            else OrderSide.SELL
            if net_quantity < 0
            else None
        )

        logger.info(f"Recalculated position for {symbol}: {position}")

    def _calculate_realized_pnl(
        self, symbol: str, removed: List[Tuple], order: Order
    ) -> Decimal:
        realized_pnl = Decimal("0")
        for removed_order in removed:
            if order.side == OrderSide.BUY:
                realized_pnl += removed_order[1] * (
                    removed_order[2] - Decimal(str(order.price))
                )
            else:  # SELL
                realized_pnl += removed_order[1] * (
                    Decimal(str(order.price)) - removed_order[2]
                )
        return realized_pnl

    async def is_position_size_valid(self, symbol: str, quantity: Decimal) -> bool:
        try:
            current_price = await self.get_current_price(symbol)
            minimum_lot_size = CONFIG["minimum_lot_size"]

            validation_result, _, _ = await validate_order(
                quantity=quantity,
                price=current_price,
                minimum_lot_size=minimum_lot_size,
                is_buy=True,
            )

            if not validation_result.is_valid:
                logger.warning(
                    f"Invalid position size for {symbol}: {validation_result.error_message}"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating position size for {symbol}: {str(e)}")
            return False

    async def place_take_profit_order(
        self, symbol: str, quantity: Decimal, entry_price: Decimal, take_profit: Decimal
    ):
        try:
            tp_order = await place_order(
                self.client,
                symbol_info=self.symbol_info,
                is_buy=False,  # Assuming we're closing a long position
                quantity=quantity,
                price=take_profit,
                minimum_lot_size=CONFIG["minimum_lot_size"],
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                stop_price=None,
            )
            logger.info(f"Take profit order placed for {symbol}: {tp_order}")
        except Exception as e:
            logger.error(
                f"Error placing take profit order for {symbol}: {str(e)}", exc_info=True
            )
