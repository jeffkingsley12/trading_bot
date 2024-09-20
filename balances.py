import asyncio
from decimal import Decimal
from typing import Dict
from logs import logger
from config import CONFIG
from binance.error import ClientError
from utils import round_decimal
import time
from rate_limiters import InsufficientBalance


class LocalBalanceTracker:
    def __init__(self, client):
        self.client = client
        self.balances: Dict[str, Dict[str, Decimal]] = {}
        self.lock = asyncio.Lock()
        self.last_update_time = 0
        self.round_decimal = round_decimal

    async def fetch_all_balances(self):
        async with self.lock:
            try:
                account_info = await self.client.account(recvWindow=6000)
                for balance in account_info["balances"]:
                    asset = balance["asset"]
                    free = self.round_decimal(
                        Decimal(balance["free"]), CONFIG["precision"]
                    )
                    locked = self.round_decimal(
                        Decimal(balance["locked"]), CONFIG["precision"]
                    )
                    total = free + locked
                    self.balances[asset] = {
                        "free": free,
                        "locked": locked,
                        "total": total,
                    }
                self.last_update_time = time.time()
                logger.info("Successfully fetched and updated all balances")
            except ClientError as e:
                logger.error(f"Binance API error fetching wallet information: {str(e)}")
                raise
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching wallet information: {str(e)}",
                    exc_info=True,
                )
                raise

    async def update_balance(
        self, asset: str, amount: Decimal, balance_type: str = "free"
    ):
        async with self.lock:
            if asset in self.balances:
                if balance_type in self.balances[asset]:
                    self.balances[asset][balance_type] = self.round_decimal(
                        self.balances[asset][balance_type] + amount, CONFIG["precision"]
                    )
                    self.balances[asset]["total"] = self.round_decimal(
                        self.balances[asset]["free"] + self.balances[asset]["locked"],
                        CONFIG["precision"],
                    )
                    logger.info(
                        f"Updated local {balance_type} balance of {asset}: {self.balances[asset][balance_type]}"
                    )
                else:
                    logger.warning(
                        f"Invalid balance type '{balance_type}' for asset {asset}"
                    )
            else:
                logger.warning(f"Asset {asset} not found in local balance tracker")

    async def check_balance(
        self, asset: str, required_amount: Decimal, balance_type: str = "free"
    ) -> bool:
        async with self.lock:
            if time.time() - self.last_update_time > CONFIG["balance_refresh_interval"]:
                await self.fetch_all_balances()

            if asset in self.balances and balance_type in self.balances[asset]:
                balance = self.balances[asset][balance_type]
                if balance >= required_amount:
                    logger.info(
                        f"Sufficient {balance_type} balance of {asset}: {balance} (required: {required_amount})"
                    )
                    return True
                else:
                    logger.warning(
                        f"Insufficient {balance_type} balance of {asset}: {balance} (required: {required_amount})"
                    )
                    raise InsufficientBalance(
                        f"Insufficient {balance_type} balance of {asset}: {balance} (required: {required_amount})"
                    )
            else:
                logger.warning(
                    f"Asset {asset} or balance type '{balance_type}' not found in local balance tracker"
                )
                raise InsufficientBalance(
                    f"Asset {asset} or balance type '{balance_type}' not found in local balance tracker"
                )


async def execute_order(tracker: LocalBalanceTracker, asset: str, amount: Decimal):
    try:
        await tracker.check_balance(asset, amount, "free")
        # Your order execution logic here
        # After successful order execution:
        await tracker.update_balance(asset, -amount, "free")  # Decrease free balance
        await tracker.update_balance(asset, amount, "locked")  # Increase locked balance
        logger.info(f"Order executed successfully for {amount} {asset}")
    except InsufficientBalance as e:
        logger.error(f"Insufficient balance for order execution: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error executing order: {str(e)}", exc_info=True)
        raise
