import requests
import asyncio

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from utils import round_decimal
from typing import Tuple
from config import CONFIG
from functools import lru_cache

from enums import ValidationResult, SymbolInfo


@lru_cache(maxsize=100)
def get_symbol_info(symbol):
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for item in data["symbols"]:
            if item["symbol"] == symbol:
                return SymbolInfo(
                    symbol=item["symbol"],
                    step_size=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "LOT_SIZE"
                            ),
                            {},
                        ).get("stepSize", 0)
                    ),
                    min_price=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "PRICE_FILTER"
                            ),
                            {},
                        ).get("minPrice", 0)
                    ),
                    max_price=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "PRICE_FILTER"
                            ),
                            {},
                        ).get("maxPrice", 0)
                    ),
                    min_notional=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "NOTIONAL"
                            ),
                            {},
                        ).get("minNotional", 0)
                    ),
                    market_step_size=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "MARKET_LOT_SIZE"
                            ),
                            {},
                        ).get("stepSize", 0)
                    ),
                    bid_multiplier_up=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "PERCENT_PRICE_BY_SIDE"
                            ),
                            {},
                        ).get("bidMultiplierUp", 0)
                    ),
                    bid_multiplier_down=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "PERCENT_PRICE_BY_SIDE"
                            ),
                            {},
                        ).get("bidMultiplierDown", 0)
                    ),
                    ask_multiplier_up=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "PERCENT_PRICE_BY_SIDE"
                            ),
                            {},
                        ).get("askMultiplierUp", 0)
                    ),
                    ask_multiplier_down=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "PERCENT_PRICE_BY_SIDE"
                            ),
                            {},
                        ).get("askMultiplierDown", 0)
                    ),
                    avg_price_mins=int(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "PERCENT_PRICE_BY_SIDE"
                            ),
                            {},
                        ).get("avgPriceMins", 0)
                    ),
                    apply_min_to_market=bool(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "NOTIONAL"
                            ),
                            {},
                        ).get("applyMinToMarket", False)
                    ),
                    max_notional=float(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "NOTIONAL"
                            ),
                            {},
                        ).get("maxNotional", 0)
                    ),
                    apply_max_to_market=bool(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "NOTIONAL"
                            ),
                            {},
                        ).get("applyMaxToMarket", False)
                    ),
                    max_num_orders=int(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "MAX_NUM_ORDERS"
                            ),
                            {},
                        ).get("maxNumOrders", 0)
                    ),
                    max_num_algo_orders=int(
                        next(
                            (
                                f
                                for f in item["filters"]
                                if f["filterType"] == "MAX_NUM_ALGO_ORDERS"
                            ),
                            {},
                        ).get("maxNumAlgoOrders", 0)
                    ),
                )
    return None


def get_tick_size_precision(tick_size: Decimal) -> int:
    return abs(tick_size.as_tuple().exponent)


async def validate_order(
    quantity: Decimal = 0.0,
    price: Decimal = 0.0,
    minimum_lot_size: Decimal = 0.0,
    is_buy: bool = True,
) -> Tuple[ValidationResult, Decimal, Decimal]:
    symbol_info: SymbolInfo = get_symbol_info(CONFIG["symbol"])
    if not symbol_info:
        return (
            ValidationResult(
                is_buy,
                False,
                "Unknown",
                quantity,
                price,
                minimum_lot_size,
                "Missing or invalid symbol_info",
            ),
            quantity,
            price,
        )

    # Ensure all inputs are Decimal
    quantity = Decimal(str(quantity))
    price = round_decimal(
        str(price), 8
    )  # Assuming 8 decimal precision, adjust if needed
    minimum_lot_size = Decimal(str(minimum_lot_size))
    step_size = Decimal(str(symbol_info.step_size))

    # Adjust quantity to be a multiple of step_size and not less than minimum_lot_size
    quantity = max(quantity, minimum_lot_size)
    quantity = (quantity / step_size).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ) * step_size

    # Validate and adjust price precision
    price = round_decimal(price, 8)  # Assuming 8 decimal precision, adjust if needed

    # Validate price
    if price < symbol_info.min_price or price > symbol_info.max_price:
        if price < symbol_info.min_price:
            close_price = symbol_info.min_price
        else:
            close_price = symbol_info.max_price

        adjusted_price = round_decimal(close_price, 8)  # Assuming 8 decimal precision
        price = adjusted_price

    # Check notional value
    notional_value = quantity * price
    min_notional_value = Decimal(str(symbol_info.min_notional))
    if notional_value < min_notional_value:
        if symbol_info.apply_min_to_market:
            new_quantity = (min_notional_value / price).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            quantity = max(new_quantity, minimum_lot_size)
            quantity = (quantity / step_size).quantize(
                Decimal("1"), rounding=ROUND_DOWN
            ) * step_size
        else:
            return (
                ValidationResult(
                    is_buy,
                    False,
                    symbol_info.symbol,
                    quantity,
                    price,
                    minimum_lot_size,
                    "Order notional value is too low",
                ),
                quantity,
                price,
            )

    # Check maximum notional value if applicable
    if symbol_info.max_notional and symbol_info.apply_max_to_market:
        max_notional_value = Decimal(str(symbol_info.max_notional))
        if notional_value > max_notional_value:
            new_quantity = (max_notional_value / price).quantize(
                Decimal("1"), rounding=ROUND_DOWN
            )
            quantity = (new_quantity / step_size).quantize(
                Decimal("1"), rounding=ROUND_DOWN
            ) * step_size

    return (
        ValidationResult(
            is_buy, True, symbol_info.symbol, quantity, price, minimum_lot_size
        ),
        quantity,
        price,
    )
