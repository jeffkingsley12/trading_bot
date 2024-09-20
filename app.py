from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/trading-data")
async def get_trading_data():
    # Replace with the actual URL of your data source or logic to fetch the data
    data = {
        'longEntry': True,
        'shortEntry': False,
        'exitLong': True,
        'exitShort': False,
        'rsi': 50,
        'loRsi': 30,
        'hiRsi': 70,
        'symbol': 'BTCUSDT',
        'currentPrice': 45000,
        'priceChange': 200,
        'uptrend': True,
        'quantity': 1.5,
        'sellPrice': 45500,
        'buyPrice': 44500,
        'risk': 1000,
        'profitTarget': 46000,
        'stopLoss': 44000,
        'numberOrders': 2,
        'buy': True,
        'sell': False,
        'historicalData': [
            {'date': '2023-07-01', 'close': 44000, 'rsi': 60},
            {'date': '2023-07-02', 'close': 44500, 'rsi': 65},
            # Add more historical data points here
        ]
    }
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
