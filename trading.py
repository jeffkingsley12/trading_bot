import asyncio
import timeit

from history_data import get_historical_data
from validate import get_symbol_info
from binance.spot import Spot
from orders import get_orders, my_trades, check_time_sync
from transactions import TradingExecutor
from prepare_env import get_api_key
from rate_limiters import RateLimiter
from utils import round_decimal
from config import CONFIG

from binance.error import ClientError
from logs import logger
import app
import tracemalloc
from balances import LocalBalanceTracker, execute_order

tracemalloc.start()


api_key, api_secret = get_api_key()
symbol = CONFIG["symbol"]


def place_order(quantity, symbol):
    print(f"Placing order for {quantity} of {symbol}")


async def main():
    client = Spot(api_key, api_secret)

    # client = Spot(api_key, api_secret, base_url="https://testnet.binance.vision")
    logger.info("Successfully connected to Binance API")

    tracker = LocalBalanceTracker(client)
    # target_asset = symbol[-5:]  # Extract the last 5 characters from the symbol
    while True:
        start_time = timeit.default_timer()
        await check_time_sync(client)

        try:
            df = await get_historical_data(client, symbol)

            # Extract various indicators and signals from the historical data
            rsi = int(df["rsi"].iloc[-1])
            lo_rsi = int(df["rsi_low"].iloc[-1])
            hi_rsi = int(df["rsi_high"].iloc[-1])
            long_entry = bool(df["long_entry"].iloc[-1])
            short_entry = bool(df["short_entry"].iloc[-1])
            exit_long = bool(df["exit_long"].iloc[-1])
            exit_short = bool(df["exit_short"].iloc[-1])
            current_price = round_decimal(df["close"].iloc[-1], CONFIG["precision"])
            buy_price = round_decimal(df["Buy_Price"].iloc[-1], CONFIG["precision"])
            sell_price = round_decimal(df["Sell_Price"].iloc[-1], CONFIG["precision"])
            stop_loss = round_decimal(df["Stop_Loss"].iloc[-1], CONFIG["precision"])
            risk = round_decimal(df["Risk_Amount"].iloc[-1], CONFIG["precision"])
            quantity = round_decimal(df["Quantity"].iloc[-1], CONFIG["precision"])
            price_change = round_decimal(
                df["Price_Change"].iloc[-1], CONFIG["precision"]
            )
            sell = bool(df["Sell"].iloc[-1])
            buy = bool(df["Buy"].iloc[-1])
            uptrend = df["UPTREND"].iloc[-1]
            bullish_engulfing = df["is_bullish_engulfing"].iloc[-1]
            bearish_engulfing = df["is_bearish_engulfing"].iloc[-1]
            profit_target = round_decimal(
                df["Profit_Target"].iloc[-1], CONFIG["precision"]
            )
            ML_Prediction = df["ML_Prediction"].iloc[-1]
            reversal_pattern = bool(bullish_engulfing or bearish_engulfing)

            orders = (await get_orders(client, symbol), await my_trades(client, symbol))

            number_orders = len(orders[0]) if orders is not None else 0

            # Print trading signals and current market conditions
            print(f""" 
            \033[1;35;40m Buy Signal: {long_entry} 
            \033[1;32;40m Sell Signal: {short_entry}
            \033[1;32;40m Exit Long Position (Sell): {exit_long} 
            \033[1;35;40m Exit Short Position (Buy):  {exit_short}
            \033[1;35;40m RSI : {rsi}
            \033[32m Both RSI: lo {lo_rsi} - hi {hi_rsi}
            \033[1;33;40m { symbol } Price : {current_price}
            \033[1;36;40m Price Change  : {price_change}
            \033[1;32;40m Up Trend: {uptrend}
            \033[1;37;40m Quantity     : {quantity} 
            \033[1;32;40m Selling Price: {sell_price}
            \033[1;37;40m Buying Price : {buy_price}
            \033[1;36;40m Risk Amount  : {risk}
            \033[1;35;40m Profit Target: {profit_target}
            \033[1;31;40m Stop Loss : {stop_loss}
            \033[1;32;40m ML Prediction: {ML_Prediction}
            \033[36m {"You have " + str(number_orders) + " orders" if number_orders > 0 else ""}
            \033[1;37;40m *******************************
            \033[0;37;40m {
                f"💰💰💰 We are buying at {buy_price} risk of {risk} USD 💰💰💰"
                if buy
                else (
                   f"💰💰💰 We are selling at {sell_price} risk of {risk} USD 💰💰💰"
                    if sell
                    else "RSI is within the range"
                )
            }
            \033[1;37;40m *******************************""")

            # print(f"main{orders}")
            # Execute trading strategy based on signals
            symbol_info = get_symbol_info(symbol)
            rate_limiter = RateLimiter(CONFIG["initial_rate"], CONFIG["max_rate"])
            try:
                async with TradingExecutor(
                    symbol,
                    client,
                    tracker,
                    execute_order,
                    symbol_info,
                    current_price,
                    sell_price,
                    buy_price,
                    stop_loss,
                    profit_target,
                    orders,
                    rate_limiter,
                ) as executor:
                    await executor.execute_trading_strategy(
                        symbol,
                        quantity,
                        buy_price,
                        sell_price,
                        long_entry,
                        short_entry,
                        exit_long,
                        exit_short,
                        current_price,
                        stop_loss,
                        profit_target,
                        reversal_pattern,
                        bullish_engulfing,
                        bearish_engulfing,
                    )
                await asyncio.sleep(0)  # Wait for the next iteration
            except ClientError as e:
                logger.error(f"Unhandled exception in main: {str(e)}")

        except ClientError as error:
            logger.error(
                "Found main error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )
        # Calculate elapsed time
        end_time = timeit.default_timer()
        elapsed_time = round_decimal(end_time - start_time, CONFIG["precision"])
        print(f"Elapsed Time: {elapsed_time} seconds")


if __name__ == "__main__":
    asyncio.run(main())
