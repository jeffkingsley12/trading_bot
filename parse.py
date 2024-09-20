import requests
import json
from typing import Union, Dict, Any
from decimal import Decimal


def parse_rate_limit_headers(response: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    result = {"weight": {}, "order_count": {}, "error": None}

    try:
        # If response is a string, try to parse it as JSON
        if isinstance(response, str):
            response = json.loads(response)

        # Check if we're dealing with an error response
        if isinstance(response, dict) and "code" in response:
            result["error"] = {
                "code": response["code"],
                "message": response.get("msg", "Unknown error"),
            }
            return result

        # If we have headers, process them
        headers = (
            response.get("headers", {}) if isinstance(response, dict) else response
        )

        for key, value in headers.items():
            if isinstance(
                key, str
            ):  # Ensure key is a string before using string methods
                if key.startswith("X-MBX-USED-WEIGHT-"):
                    interval = key.split("-")[-1]
                    result["weight"][interval] = int(value)
                elif key.startswith("X-MBX-ORDER-COUNT-"):
                    interval = key.split("-")[-1]
                    result["order_count"][interval] = int(value)

    except json.JSONDecodeError:
        result["error"] = {
            "code": "JSON_PARSE_ERROR",
            "message": "Failed to parse response as JSON",
        }
    except Exception as e:
        result["error"] = {"code": "UNKNOWN_ERROR", "message": str(e)}

    return result


async def get_price_symbol(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return Decimal(data["price"])

    return None
