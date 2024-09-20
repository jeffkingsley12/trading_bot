import asyncio
from typing import Dict, List, Tuple, Optional
from decimal import Decimal
from sortedcontainers import SortedDict
from collections import deque
from enums import Order, OrderStatus, OrderSide, OrderType, TimeInForce, SymbolInfo, ValidationResult, Position

class PriceLevel:
    def __init__(self):
        self.orders = deque()
        self.total_quantity = Decimal(0)

    def add_order(self, order: Order):
        self.orders.append(order)
        self.total_quantity += order.quantity

    def remove_order(self, order: Order):
        self.orders.remove(order)
        self.total_quantity -= order.quantity

class OrderBook:
    def __init__(self):
        self.bids = SortedDict(reverse=True)  # Highest price first
        self.asks = SortedDict()  # Lowest price first
        self.orders: Dict[int, Order] = {}

    def add_order(self, order: Order):
        book = self.bids if order.side == OrderSide.BUY else self.asks
        if order.price not in book:
            book[order.price] = PriceLevel()
        book[order.price].add_order(order)
        self.orders[order.order_id] = order

    def remove_order(self, order_id: int) -> Optional[Order]:
        if order_id in self.orders:
            order = self.orders[order_id]
            book = self.bids if order.side == OrderSide.BUY else self.asks
            price_level = book[order.price]
            price_level.remove_order(order)
            if price_level.total_quantity == 0:
                del book[order.price]
            del self.orders[order_id]
            return order
        return None

    def get_best_price(self, side: OrderSide) -> Optional[Decimal]:
        book = self.bids if side == OrderSide.BUY else self.asks
        return next(iter(book.keys())) if book else None

class OptimizedOrderMatchingEngine:
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self.symbol_info: Dict[str, SymbolInfo] = {}
        self.positions: Dict[str, Position] = {}

    def get_order_book(self, symbol: str) -> OrderBook:
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook()
        return self.order_books[symbol]

    def set_symbol_info(self, symbol_info: SymbolInfo):
        self.symbol_info[symbol_info.symbol] = symbol_info

    def validate_order(self, order: Order) -> ValidationResult:
        symbol_info = self.symbol_info.get(order.symbol)
        if not symbol_info:
            return ValidationResult(False, False, order.symbol, order.quantity, order.price, Decimal("0"), "Symbol not found")

        if order.quantity % symbol_info.step_size != 0:
            return ValidationResult(False, False, order.symbol, order.quantity, order.price, symbol_info.step_size, "Invalid quantity step size")

        if order.price < symbol_info.min_price or order.price > symbol_info.max_price:
            return ValidationResult(False, False, order.symbol, order.quantity, order.price, Decimal("0"), "Invalid price")

        notional = order.quantity * order.price
        if notional < symbol_info.min_notional:
            return ValidationResult(False, False, order.symbol, order.quantity, order.price, Decimal("0"), "Order value below minimum notional")

        if symbol_info.apply_max_to_market and notional > symbol_info.max_notional:
            return ValidationResult(False, False, order.symbol, order.quantity, order.price, Decimal("0"), "Order value above maximum notional")

        return ValidationResult(order.side == OrderSide.BUY, True, order.symbol, order.quantity, order.price, symbol_info.step_size)

    async def add_order(self, order: Order) -> bool:
        validation_result = self.validate_order(order)
        if not validation_result.is_valid:
            order.status = OrderStatus.REJECTED
            return False

        order_book = self.get_order_book(order.symbol)
        order_book.add_order(order)
        
        if order.type == OrderType.LIMIT:
            await self.match_orders(order.symbol)
        elif order.type == OrderType.MARKET:
            await self.execute_market_order(order)

        return True

    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        order_book = self.get_order_book(symbol)
        order = order_book.remove_order(order_id)
        if order:
            order.status = OrderStatus.CANCELED
            return True
        return False

    async def match_orders(self, symbol: str) -> List[Tuple[Order, Order, Decimal]]:
        order_book = self.get_order_book(symbol)
        matches = []

        while True:
            best_bid = order_book.get_best_price(OrderSide.BUY)
            best_ask = order_book.get_best_price(OrderSide.SELL)

            if best_bid is None or best_ask is None or best_bid < best_ask:
                break

            bid_orders = order_book.bids[best_bid].orders
            ask_orders = order_book.asks[best_ask].orders

            while bid_orders and ask_orders:
                buy_order = bid_orders[0]
                sell_order = ask_orders[0]

                match_quantity = min(buy_order.quantity - buy_order.executed_quantity,
                                     sell_order.quantity - sell_order.executed_quantity)
                match_price = sell_order.price

                matches.append((buy_order, sell_order, match_quantity))

                await self.update_order(buy_order, match_quantity, match_price)
                await self.update_order(sell_order, match_quantity, match_price)

                if buy_order.status == OrderStatus.FILLED:
                    bid_orders.popleft()
                    order_book.remove_order(buy_order.order_id)

                if sell_order.status == OrderStatus.FILLED:
                    ask_orders.popleft()
                    order_book.remove_order(sell_order.order_id)

            if not bid_orders:
                del order_book.bids[best_bid]
            if not ask_orders:
                del order_book.asks[best_ask]

        return matches

    async def update_order(self, order: Order, executed_quantity: Decimal, executed_price: Decimal):
        order.executed_quantity += executed_quantity
        order.cumulative_quote_quantity += executed_quantity * executed_price

        if order.executed_quantity == order.quantity:
            order.status = OrderStatus.FILLED
        elif order.executed_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED

        await self.update_position(order.symbol, order.side, executed_quantity, executed_price)

    async def update_position(self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal):
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, Decimal(0), Decimal(0), None, Decimal(0))

        position = self.positions[symbol]
        quantity_change = quantity if side == OrderSide.BUY else -quantity
        position.update(quantity_change, price)

    async def execute_market_order(self, order: Order):
        order_book = self.get_order_book(order.symbol)
        opposite_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        remaining_quantity = order.quantity

        while remaining_quantity > 0:
            best_price = order_book.get_best_price(opposite_side)
            if best_price is None:
                break  # No more orders to match

            price_level = order_book.asks[best_price] if order.side == OrderSide.BUY else order_book.bids[best_price]
            for matched_order in price_level.orders:
                match_quantity = min(remaining_quantity, matched_order.quantity - matched_order.executed_quantity)
                await self.update_order(order, match_quantity, best_price)
                await self.update_order(matched_order, match_quantity, best_price)
                remaining_quantity -= match_quantity

                if matched_order.status == OrderStatus.FILLED:
                    price_level.orders.remove(matched_order)
                    order_book.remove_order(matched_order.order_id)

                if remaining_quantity == 0:
                    break

            if not price_level.orders:
                del (order_book.asks if order.side == OrderSide.BUY else order_book.bids)[best_price]

        if remaining_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.FILLED

    def get_order_book_snapshot(self, symbol: str, depth: int = 10) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        order_book = self.get_order_book(symbol)
        bids = [(price, level.total_quantity) for price, level in list(order_book.bids.items())[:depth]]
        asks = [(price, level.total_quantity) for price, level in list(order_book.asks.items())[:depth]]
        return {"bids": bids, "asks": asks}

# Usage example
async def main():
    engine = OptimizedOrderMatchingEngine()
    
    # Set symbol info
    btc_info = SymbolInfo("BTCUSD", Decimal("0.00001"), Decimal("10000"), Decimal("100000"), Decimal("10"), Decimal("0.00001"), 
                          Decimal("1.1"), Decimal("0.9"), Decimal("1.1"), Decimal("0.9"), 5, True, Decimal("1000000"), True, 100, 50)
    engine.set_symbol_info(btc_info)

    # Add some orders
    order1 = Order(1, OrderStatus.NEW, "BTCUSD", OrderSide.BUY, Decimal("1"), Decimal("50000"), OrderType.LIMIT, TimeInForce.GTC)
    order2 = Order(2, OrderStatus.NEW, "BTCUSD", OrderSide.BUY, Decimal("0.5"), Decimal("50001"), OrderType.LIMIT, TimeInForce.GTC)
    order3 = Order(3, OrderStatus.NEW, "BTCUSD", OrderSide.SELL, Decimal("0.7"), Decimal("50000"), OrderType.LIMIT, TimeInForce.GTC)

    await engine.add_order(order1)
    await engine.add_order(order2)
    await engine.add_order(order3)

    # Print the order book
    snapshot = engine.get_order_book_snapshot("BTCUSD")
    print("Order Book:")
    print("Bids:", snapshot["bids"])
    print("Asks:", snapshot["asks"])

    # Print matches
    matches = await engine.match_orders("BTCUSD")
    for buy_order, sell_order, quantity in matches:
        print(f"Match: Buy Order {buy_order.order_id} with Sell Order {sell_order.order_id}, Quantity: {quantity}")

    # Print positions
    for symbol, position in engine.positions.items():
        print(f"Position for {symbol}: {position}")

if __name__ == "__main__":
    asyncio.run(main())
