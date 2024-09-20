import asyncio
import time
from collections import deque
from binance.error import ClientError
from typing import List, Union, Optional, Callable
from tenacity import (
    retry,
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from logs import logger
from parse import parse_rate_limit_headers
from functools import wraps
from config import CONFIG
from decimal import Decimal


# Improved rate limiter for Binance API.


class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_time: float):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.monotonic()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def is_open(self):
        if (
            self.state == "OPEN"
            and time.monotonic() - self.last_failure_time > self.recovery_time
        ):
            self.state = "HALF-OPEN"
        return self.state == "OPEN"


class RateLimiter:
    def __init__(self, initial_rate: int, max_rate: int):
        self.current_rate = Decimal(str(initial_rate))
        self.max_rate = Decimal(str(max_rate))
        self.last_call_time = Decimal(str(time.monotonic()))
        self.call_times: deque[Decimal] = deque(maxlen=100)
        self.min_interval = Decimal("1") / self.max_rate
        self.lock = asyncio.Lock()
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time=60)

    async def wait(self, response: Optional[Union[Exception, dict]] = None):
        async with self.lock:
            current_time = Decimal(str(time.monotonic()))
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                await asyncio.sleep(float(sleep_time))

            self.last_call_time = Decimal(str(time.monotonic()))
            self.call_times.append(self.last_call_time)

            if len(self.call_times) == 100:
                self._adjust_rate()

            if response:
                self._adjust_rate_based_on_response(response)

    def _adjust_rate(self):
        time_diff = self.call_times[-1] - self.call_times[0]
        actual_rate = Decimal("99") / time_diff if time_diff > 0 else self.max_rate
        self.current_rate = min(actual_rate * Decimal("0.95"), self.max_rate)
        self.min_interval = Decimal("1") / self.current_rate

    def _adjust_rate_based_on_response(self, response: Union[Exception, dict]):
        rate_limit_info = parse_rate_limit_headers(response)
        if not rate_limit_info.get("error"):
            for interval, used_weight in rate_limit_info.get("weight", {}).items():
                limit = rate_limit_info.get("order_count", {}).get(interval, 0)
                if limit > 0:
                    usage_percentage = Decimal(str(used_weight)) / Decimal(str(limit))
                    if usage_percentage < Decimal("0.5"):
                        self.current_rate = min(
                            self.current_rate * Decimal("1.05"), self.max_rate
                        )
                    elif usage_percentage > Decimal("0.8"):
                        self.current_rate = max(
                            self.current_rate * Decimal("0.95"), Decimal("1")
                        )
                    self.min_interval = Decimal("1") / self.current_rate


rate_limiter = RateLimiter(CONFIG["initial_rate"], CONFIG["max_rate"])


def sophisticated_retry(func: Callable):
    @retry(
        retry=retry_if_exception_type((ClientError, ConnectionError, TimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying {func.__name__} in {retry_state.next_action.sleep} seconds"
        ),
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


def rate_limited(func):
    @wraps(func)
    @sophisticated_retry
    async def wrapper(self, *args, **kwargs):
        if rate_limiter.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker is open. Skipping call to {func.__name__}")
            raise Exception("Circuit breaker is open")

        await rate_limiter.wait()
        try:
            response = await func(self, *args, **kwargs)
            await rate_limiter.wait(response)
            rate_limiter.circuit_breaker.record_success()
            return response
        except ClientError as e:
            logger.error(f"Binance API error in {func.__name__}: {str(e)}")
            rate_limiter.circuit_breaker.record_failure()
            if "Insufficient balance" in str(e):
                raise InsufficientBalance(str(e))
            raise
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"{type(e).__name__} in {func.__name__}: {str(e)}")
            rate_limiter.circuit_breaker.record_failure()
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
            rate_limiter.circuit_breaker.record_failure()
            raise

    return wrapper


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")
        self.retry_after = retry_after


class IPBanned(Exception):
    def __init__(self, retry_after: int):
        super().__init__(f"IP banned. Retry after {retry_after} seconds.")
        self.retry_after = retry_after


class InsufficientBalance(Exception):
    def __init__(self, message: str):
        super().__init__(message)


# class InsufficientBalance(Exception):
#     def __init__(self, code: int, message: str):
#         self.code = code
#         self.msg = message
#         super().__init__(f"Code {code}: {self.msg}")

#     @classmethod
#     def from_binance_error(cls, error_dict: dict):
#         return cls(error_dict.get("code", -1), error_dict.get("msg", "Unknown error"))


class OrderPlacementError(Exception):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.original_error = original_error
        self.message = message
        super().__init__(f"{message} : {original_error}")


class WeightTracker:
    def __init__(self):
        self.weights = {}

    def update(self, interval: str, weight: int):
        self.weights[interval] = weight

    def get_weight(self, interval: str) -> int:
        return self.weights.get(interval, 0)


class OrderCountTracker:
    def __init__(self):
        self.order_counts = {}

    def update(self, interval: str, count: int):
        self.order_counts[interval] = count

    def get_count(self, interval: str) -> int:
        return self.order_counts.get(interval, 0)


weight_tracker = WeightTracker()
order_count_tracker = OrderCountTracker()


RetryableError = Union[RateLimitExceeded, IPBanned, ClientError]


async def check_balance(client, asset, amount):
    account_info = await client.account()
    balances = {
        balance["asset"]: Decimal(balance["free"])
        for balance in account_info["balances"]
    }
    if asset not in balances or balances[asset] < amount:
        raise InsufficientBalance(
            f"Insufficient balance. Required: {amount} {asset}, Available: {balances.get(asset, 0)} {asset}"
        )


def is_insufficient_balance_error(e: ClientError) -> bool:
    return (
        e.error_code == -2010 and "insufficient balance" in e.error_message.lower()
    ) or (e.error_code == -1013 and "could not be found" in e.error_message.lower())


async def create_order_with_retry(client, order_params: dict) -> dict:
    async def handle_client_error(e: ClientError):
        error_code = e.error_code
        error_message = e.error_message

        if "Stop price would trigger immediately" in error_message:
            order_params["type"] = "LIMIT"
            order_params.pop("stopPrice", None)
            order_params.pop("price", None)
            logger.info(
                "Modified order to limit order due to immediate stop price trigger"
            )
            return True  # Retry with modified params

        if error_code in [-1003, -1006]:
            raise RateLimitExceeded("Rate limit exceeded. Please try again later.")
        if error_code == -1007:
            raise IPBanned("IP banned. Please try again later.")
        if is_insufficient_balance_error(e):
            raise InsufficientBalance("Insufficient balance for the requested action.")

        logger.error(f"Unhandled error: {error_message}")
        return False  # Don't retry for unhandled errors

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((RateLimitExceeded, IPBanned, ClientError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying in {retry_state.next_action.sleep} seconds..."
        ),
    ):
        with attempt:
            try:
                await rate_limiter.wait()
                response = client.new_order(**order_params)
                await rate_limiter.wait(response)

                if response is None:
                    raise OrderPlacementError("No response received from the exchange")

                return response

            except ClientError as e:
                if not await handle_client_error(e):
                    raise OrderPlacementError(f"Order placement failed: {str(e)}")

    raise OrderPlacementError("Max retries reached. Order placement failed.")


# Helper function
