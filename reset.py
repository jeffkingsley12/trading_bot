import asyncio


async def partial_reset_position(self, symbol: str, failed_order_id: int):
    async with self.lock:
        logger.warning(
            f"Partially resetting position for {symbol} due to order placement failure"
        )

        # Remove the failed order from the queue
        if symbol in self.buy_queue:
            self.buy_queue[symbol].remove_by_id(failed_order_id)
        if symbol in self.sell_queue:
            self.sell_queue[symbol].remove_by_id(failed_order_id)

        # Remove the failed order from tracking
        self.orders.pop(failed_order_id, None)

        # Cancel any monitoring task for this order
        if failed_order_id in self.monitoring_tasks:
            self.monitoring_tasks[failed_order_id].cancel()
            del self.monitoring_tasks[failed_order_id]

        # Recalculate the position based on the remaining orders
        await self._recalculate_position(symbol)

        logger.info(f"Partial position reset completed for {symbol}")


async def _recalculate_position(self, symbol: str):
    buy_quantity = self.buy_queue[symbol].total_quantity()
    sell_quantity = self.sell_queue[symbol].total_quantity()
    net_quantity = buy_quantity - sell_quantity

    if symbol not in self.positions:
        self.positions[symbol] = Position(symbol, Decimal("0"), Decimal("0"), None)

    position = self.positions[symbol]
    position.net_quantity = net_quantity
    position.absolute_quantity = abs(net_quantity)

    if position.absolute_quantity != Decimal("0"):
        position.average_price = (
            self.buy_queue[symbol].total_price() - self.sell_queue[symbol].total_price()
        ) / position.absolute_quantity
    else:
        position.average_price = Decimal("0")

    position.side = OrderSide.BUY if position.net_quantity > 0 else OrderSide.SELL

    logger.info(f"Recalculated position for {symbol}: {position}")


async def reset_position(self, symbol: str):
    async with self.lock:
        logger.warning(
            f"Resetting position for {symbol} due to order placement failure"
        )

        # Cancel all open orders for the symbol
        await self.cancel_open_orders(self.client, symbol)

        # Clear the buy and sell queues
        if symbol in self.buy_queue:
            self.buy_queue[symbol] = OrderQueue()
        if symbol in self.sell_queue:
            self.sell_queue[symbol] = OrderQueue()

        # Reset the position
        if symbol in self.positions:
            self.positions[symbol] = Position(symbol, Decimal("0"), Decimal("0"), None)

        # Remove all orders for this symbol
        self.orders = {k: v for k, v in self.orders.items() if v.symbol != symbol}

        # Cancel any monitoring tasks for this symbol
        tasks_to_cancel = [
            task
            for order_id, task in self.monitoring_tasks.items()
            if self.orders.get(order_id, Order(symbol="")).symbol == symbol
        ]
        for task in tasks_to_cancel:
            task.cancel()

        logger.info(f"Position reset completed for {symbol}")
