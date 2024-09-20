import asyncio
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from enums import (
    Order,
    OrderStatus,
    OrderSide,
    OrderType,
    TimeInForce,
    Position,
    SymbolInfo,
    ValidationResult,
)
from logs import logger


class OrderMatchingEngine:
    def __init__(self):
        self.buy_orders: Dict[str, List[Order]] = {}
        self.sell_orders: Dict[str, List[Order]] = {}

    def add_order(self, order: Order):
        orders_dict = (
            self.buy_orders if order.side == OrderSide.BUY else self.sell_orders
        )
        if order.symbol not in orders_dict:
            orders_dict[order.symbol] = []
        orders_dict[order.symbol].append(order)
        orders_dict[order.symbol].sort(
            key=lambda x: x.price, reverse=(order.side == OrderSide.BUY)
        )

    def match_orders(self, symbol: str) -> List[Tuple[Order, Order, Decimal]]:
        matches = []
        buy_orders = self.buy_orders.get(symbol, [])
        sell_orders = self.sell_orders.get(symbol, [])

        while buy_orders and sell_orders:
            buy_order = buy_orders[0]
            sell_order = sell_orders[0]

            if buy_order.price >= sell_order.price:
                match_quantity = min(
                    buy_order.quantity - buy_order.executed_quantity,
                    sell_order.quantity - sell_order.executed_quantity,
                )
                match_price = (
                    sell_order.price
                )  # Assume the sell order's price is the execution price

                matches.append((buy_order, sell_order, match_quantity))

                buy_order.executed_quantity += match_quantity
                sell_order.executed_quantity += match_quantity

                if buy_order.executed_quantity == buy_order.quantity:
                    buy_orders.pop(0)
                if sell_order.executed_quantity == sell_order.quantity:
                    sell_orders.pop(0)
            else:
                break

        return matches


class OrderManager:
    def __init__(self):
        self.orders: Dict[int, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.symbol_info: Dict[str, SymbolInfo] = {}
        self.matching_engine = OrderMatchingEngine()

    def set_symbol_info(self, symbol: str, info: SymbolInfo):
        self.symbol_info[symbol] = info

    def validate_order(self, order: Order) -> ValidationResult:
        symbol_info = self.symbol_info.get(order.symbol)
        if not symbol_info:
            return ValidationResult(
                False,
                False,
                order.symbol,
                order.quantity,
                order.price,
                Decimal("0"),
                "Symbol not found",
            )

        if order.quantity % symbol_info.step_size != 0:
            return ValidationResult(
                False,
                False,
                order.symbol,
                order.quantity,
                order.price,
                symbol_info.step_size,
                "Invalid quantity step size",
            )

        if order.price < symbol_info.min_price or order.price > symbol_info.max_price:
            return ValidationResult(
                False,
                False,
                order.symbol,
                order.quantity,
                order.price,
                Decimal("0"),
                "Invalid price",
            )

        notional = order.quantity * order.price
        if notional < symbol_info.min_notional:
            return ValidationResult(
                False,
                False,
                order.symbol,
                order.quantity,
                order.price,
                Decimal("0"),
                "Order value below minimum notional",
            )

        if symbol_info.apply_max_to_market and notional > symbol_info.max_notional:
            return ValidationResult(
                False,
                False,
                order.symbol,
                order.quantity,
                order.price,
                Decimal("0"),
                "Order value above maximum notional",
            )

        return ValidationResult(
            order.side == OrderSide.BUY,
            True,
            order.symbol,
            order.quantity,
            order.price,
            symbol_info.step_size,
        )

    async def place_order(self, order: Order) -> bool:
        validation_result = self.validate_order(order)
        if not validation_result.is_valid:
            logger.warning(
                f"Order validation failed: {validation_result.error_message}"
            )
            return False

        self.orders[order.order_id] = order
        self.matching_engine.add_order(order)

        matches = self.matching_engine.match_orders(order.symbol)
        for buy_order, sell_order, match_quantity in matches:
            await self.process_match(buy_order, sell_order, match_quantity)

        if order.executed_quantity > 0:
            await self.update_position(order)

        if order.executed_quantity == order.quantity:
            order.status = OrderStatus.FILLED
        elif order.executed_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.NEW

        logger.info(f"Order placed: {order}")
        return True

    async def process_match(
        self, buy_order: Order, sell_order: Order, match_quantity: Decimal
    ):
        execution_price = sell_order.price

        buy_order.executed_quantity += match_quantity
        buy_order.cumulative_quote_quantity += match_quantity * execution_price

        sell_order.executed_quantity += match_quantity
        sell_order.cumulative_quote_quantity += match_quantity * execution_price

        await self.update_position_from_match(
            buy_order, match_quantity, execution_price
        )
        await self.update_position_from_match(
            sell_order, match_quantity, execution_price
        )

        logger.info(
            f"Match processed: Buy Order {buy_order.order_id}, Sell Order {sell_order.order_id}, Quantity {match_quantity}, Price {execution_price}"
        )

    async def update_position_from_match(
        self, order: Order, match_quantity: Decimal, execution_price: Decimal
    ):
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol, Decimal("0"), Decimal("0"), None, Decimal("0")
            )

        position = self.positions[symbol]
        quantity_change = (
            match_quantity if order.side == OrderSide.BUY else -match_quantity
        )
        position.update(quantity_change, execution_price)

        logger.info(f"Position updated for {symbol}: {position}")

    async def update_position(self, order: Order) -> None:
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol, Decimal("0"), Decimal("0"), None, Decimal("0")
            )

        position = self.positions[symbol]
        quantity_change = (
            order.executed_quantity
            if order.side == OrderSide.BUY
            else -order.executed_quantity
        )
        average_execution_price = (
            order.cumulative_quote_quantity / order.executed_quantity
            if order.executed_quantity > 0
            else Decimal("0")
        )
        position.update(quantity_change, average_execution_price)

        logger.info(f"Position updated for {symbol}: {position}")

    async def cancel_order(self, order_id: int) -> bool:
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                order.status = OrderStatus.CANCELED
                # Remove the order from the matching engine
                if order.side == OrderSide.BUY:
                    self.matching_engine.buy_orders[order.symbol] = [
                        o
                        for o in self.matching_engine.buy_orders[order.symbol]
                        if o.order_id != order_id
                    ]
                else:
                    self.matching_engine.sell_orders[order.symbol] = [
                        o
                        for o in self.matching_engine.sell_orders[order.symbol]
                        if o.order_id != order_id
                    ]
                logger.info(f"Order canceled: {order}")
                return True
            else:
                logger.warning(
                    f"Cannot cancel order {order_id}: Order is in {order.status} status"
                )
        else:
            logger.warning(f"Order {order_id} not found")
        return False

    def get_open_positions(self) -> List[Position]:
        return [pos for pos in self.positions.values() if pos.is_open]

    def calculate_unrealized_pnl(
        self, symbol: str, current_price: Decimal
    ) -> Optional[Decimal]:
        if symbol in self.positions:
            position = self.positions[symbol]
            return (current_price - position.average_price) * position.net_quantity
        return None

    def get_order_book(self, symbol: str) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        buy_orders = self.matching_engine.buy_orders.get(symbol, [])
        sell_orders = self.matching_engine.sell_orders.get(symbol, [])

        buy_book = [
            (order.price, order.quantity - order.executed_quantity)
            for order in buy_orders
        ]
        sell_book = [
            (order.price, order.quantity - order.executed_quantity)
            for order in sell_orders
        ]

        return {"bids": buy_book, "asks": sell_book}


# Usage example
async def main():
    order_manager = OrderManager()

    # Set symbol info
    btc_info = SymbolInfo(
        "BTCUSD",
        Decimal("0.00001"),
        Decimal("10000"),
        Decimal("100000"),
        Decimal("10"),
        Decimal("0.00001"),
        Decimal("1.1"),
        Decimal("0.9"),
        Decimal("1.1"),
        Decimal("0.9"),
        5,
        True,
        Decimal("1000000"),
        True,
        100,
        50,
    )
    order_manager.set_symbol_info("BTCUSD", btc_info)

    # Place buy order
    buy_order = Order(
        order_id=1,
        status=OrderStatus.NEW,
        symbol="BTCUSD",
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        price=Decimal("50000"),
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
    )
    await order_manager.place_order(buy_order)

    # Place sell order
    sell_order = Order(
        order_id=2,
        status=OrderStatus.NEW,
        symbol="BTCUSD",
        side=OrderSide.SELL,
        quantity=Decimal("0.5"),
        price=Decimal("50000"),
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
    )
    await order_manager.place_order(sell_order)

    # Get open positions
    open_positions = order_manager.get_open_positions()
    print("Open positions:", open_positions)

    # Calculate unrealized PNL
    unrealized_pnl = order_manager.calculate_unrealized_pnl("BTCUSD", Decimal("55000"))
    print("Unrealized PNL:", unrealized_pnl)

    # Get order book
    order_book = order_manager.get_order_book("BTCUSD")
    print("Order book:", order_book)


if __name__ == "__main__":
    asyncio.run(main())
