

def cancel_order(symbol, orderId):
    try:
        response = client.cancel_order(symbol=symbol, orderId=orderId)
        logger.info(response)
    except BinanceAPIException as error:
        logger.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

