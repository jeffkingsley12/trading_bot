import time as time_module  # Make sure to import the time module
from enum import Enum, auto

from dataclasses import dataclass, field
import time as time_module
from logs import logger
from typing import Dict, Any, Optional, List, NamedTuple
from decimal import Decimal
from datetime import datetime


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXPIRED_IN_MATCH = "EXPIRED_IN_MATCH"  # New status added

    @classmethod
    def from_string(cls, status_string):
        try:
            return cls(status_string)
        except ValueError:
            print(
                f"Warning: Unknown order status '{status_string}'. Treating as EXPIRED."
            )
            return cls.EXPIRED


@dataclass
class OrderEntry:
    order_id: int
    quantity: Decimal
    price: Decimal


class OrderState(Enum):
    NEW = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELED = auto()
    REJECTED = auto()


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


@dataclass
class SymbolInfo:
    symbol: str
    step_size: Decimal
    min_price: Decimal
    max_price: Decimal
    min_notional: Decimal
    market_step_size: Decimal
    bid_multiplier_up: Decimal
    bid_multiplier_down: Decimal
    ask_multiplier_up: Decimal
    ask_multiplier_down: Decimal
    avg_price_mins: int
    apply_min_to_market: bool
    max_notional: Decimal
    apply_max_to_market: bool
    max_num_orders: int
    max_num_algo_orders: int


class ValidationResult(NamedTuple):
    is_buy: bool
    is_valid: bool
    symbol: str
    quantity: Decimal
    price: Decimal
    minimum_lot_size: Decimal
    error_message: Optional[str] = None


@dataclass
class Order:
    order_id: int
    status: OrderStatus
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    type: OrderType
    time_in_force: TimeInForce
    executed_quantity: Decimal = Decimal("0")
    cumulative_quote_quantity: Decimal = Decimal("0")
    client_order_id: str = ""
    order_list_id: int = -1
    stop_price: Decimal = Decimal("0")
    iceberg_quantity: Decimal = Decimal("0")
    time: int = field(default_factory=lambda: int(time_module.time() * 1000))
    update_time: int = field(default_factory=lambda: int(time_module.time() * 1000))
    is_working: bool = True
    working_time: int = field(default_factory=lambda: int(time_module.time() * 1000))
    orig_quote_order_quantity: Decimal = Decimal("0")
    self_trade_prevention_mode: str = "NONE"
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.order_id, self.symbol))

    def __eq__(self, other):
        if isinstance(other, Order):
            return self.order_id == other.order_id and self.symbol == other.symbol
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orderId": self.order_id,
            "status": self.status.value,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "type": self.type,
            "time_in_force": self.TimeInForce,
            "executed_quantity": self.executed_quantity,
            "cumulative_quote_quantity": self.executed_quantity,
            "client_order_id": self.client_order_id,
            "order_list_id": self.order_list_id,
            "stop_price": self.stop_price,
            "iceberg_quantity": self.iceberg_quantity,
            "time": self.time,
            "update_time": self.update_time,
            "is_working": self.is_working,
            "working_time": self.working_time,
            "orig_quote_order_quantity": self.orig_quote_order_quantity,
            "self_trade_prevention_mode": self.self_trade_prevention_mode,
            "additional_info": self.additional_info,
        }

    @classmethod
    def dummy(cls, symbol: str = ""):
        return cls(
            order_id=0,
            status=OrderStatus.NEW,
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=Decimal("0"),
            price=Decimal("0"),
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        logger.debug(f"Creating Order from dict: {data}")
        mapped_data = {
            "symbol": data["symbol"],
            "order_id": int(data["orderId"]),  # Make sure this line is correct
            "status": OrderStatus(data["status"]),
            "side": OrderSide(data["side"]),
            "type": OrderType(data["type"]),
            "time_in_force": TimeInForce(data.get("timeInForce", "GTC")),
            "quantity": Decimal(data["origQty"]),
            "price": Decimal(data["price"]),
            "executed_quantity": Decimal(data["executedQty"]),
            "cumulative_quote_quantity": Decimal(data["cummulativeQuoteQty"]),
            "client_order_id": data["clientOrderId"],
            "order_list_id": int(data.get("orderListId", -1)),
            "stop_price": Decimal(data.get("stopPrice", "0")),
            "iceberg_quantity": Decimal(data.get("icebergQty", "0")),
            "time": int(data.get("time", int(time_module.time() * 1000))),
            "update_time": int(data.get("updateTime", int(time_module.time() * 1000))),
            "is_working": bool(data.get("isWorking", True)),
            "working_time": int(
                data.get("workingTime", int(time_module.time() * 1000))
            ),
            "orig_quote_order_quantity": Decimal(data.get("origQuoteOrderQty", "0")),
            "self_trade_prevention_mode": data.get("selfTradePreventionMode", "NONE"),
        }

        known_fields = set(mapped_data.keys()) | {
            "symbol",
            "orderId",
            "status",
            "side",
            "type",
            "timeInForce",
            "origQty",
            "price",
            "executedQty",
            "cummulativeQuoteQty",
            "clientOrderId",
            "orderListId",
            "stopPrice",
            "icebergQty",
            "time",
            "updateTime",
            "isWorking",
            "workingTime",
            "origQuoteOrderQty",
            "selfTradePreventionMode",
        }
        additional_info = {k: v for k, v in data.items() if k not in known_fields}

        logger.debug(f"Raw order data: {data}")
        logger.debug(f"Mapped order data: {mapped_data}")
        order = cls(**mapped_data, additional_info=additional_info)
        logger.debug(f"Created Order object: {order}")
        return order


# @dataclass
# class Position:
#     symbol: str
#     net_quantity: Decimal
#     average_price: Decimal


@dataclass
class Position:
    symbol: str
    net_quantity: Decimal
    average_price: Decimal
    side: OrderSide
    entry_price: Decimal
    absolute_quantity: Decimal = Decimal("0")
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.absolute_quantity = abs(self.net_quantity)

    @property
    def is_open(self):
        return self.net_quantity != Decimal("0")

    def update(self, quantity: Decimal, price: Decimal):
        if self.net_quantity == Decimal("0"):
            self.entry_price = price
            self.side = OrderSide.BUY if quantity > 0 else OrderSide.SELL

        new_total = self.net_quantity * self.average_price + quantity * price
        self.net_quantity += quantity
        self.absolute_quantity = abs(self.net_quantity)

        if self.absolute_quantity != Decimal("0"):
            self.average_price = new_total / self.absolute_quantity
        else:
            self.average_price = Decimal("0")
            self.side = None

        if self.net_quantity > 0:
            self.side = OrderSide.BUY
        elif self.net_quantity < 0:
            self.side = OrderSide.SELL

    # s#     return f"Position(symbol={self.symbol}, net_quantity={self.net_quantity}, absolute_quantity={self.absolute_quantity}, average_price={self.average_price}, side={self.side})"


class ExitAttempt:
    def __init__(self):
        self.last_attempt = 0
        self.cooldown = 60  # 60 seconds cooldown

    def can_attempt(self):
        current_time = time_module.time()
        if current_time - self.last_attempt > self.cooldown:
            self.last_attempt = current_time
            return True
        return False
