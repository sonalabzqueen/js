import requests
import time
import pandas as pd
from datetime import datetime

# Define start and end dates
DAYS = 100 # Number of days to fetch
end_date = datetime.utcnow().strftime("%Y-%m-%d %H%M%S")  # Current UTC date
start_date = datetime.utcnow() - pd.Timedelta(days=DAYS)  # 100 days ago
start_date = start_date.strftime("%Y-%m-%d %H%M%S")

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

# Define trading pair and interval
symbol = "SOLUSDT"
#symbol = "BTCUSDT"
interval = "4h"  # Hourly candles

# Convert start and end dates to timestamps (milliseconds)
start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d %H%M%S").timestamp() * 1000)
end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d %H%M%S").timestamp() * 1000)

# Function to fetch historical data
def fetch_historical_klines(symbol, interval, start_time, end_time):
    all_data = []
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # Max limit
        }
        print(f'start_time={datetime.utcfromtimestamp(start_time / 1000)}')
        print(f'end_time={datetime.utcfromtimestamp(end_time / 1000)}')
        response = requests.get(BINANCE_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data:
                break  # No more data available
            all_data.extend(data)

            # Update start_time for the next request
            start_time = data[-1][0] + 1  # Move past last returned candle
            time.sleep(0.5)  # Prevent hitting API rate limits
        else:
            print(f"Error {response.status_code}: {response.text}")
            break
    return all_data

# Fetch data
historical_klines = fetch_historical_klines(symbol, interval, start_timestamp, end_timestamp)

# Convert to DataFrame
columns = ["datetime", "open_price", "high_price", "low_price", "close_price", "volume", "close_time",
           "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
df = pd.DataFrame(historical_klines, columns=columns)

# Convert numeric columns to proper types
df[["open_price", "high_price", "low_price", "close_price", "volume", "quote_asset_volume",
    "taker_buy_base", "taker_buy_quote"]] = df[[
    "open_price", "high_price", "low_price", "close_price", "volume", "quote_asset_volume",
    "taker_buy_base", "taker_buy_quote"
]].astype(float)
df['datetime'] = pd.to_datetime(df["datetime"], unit="ms")  # Convert to datetime

df.head(n=1)

timestamp = df["datetime"].iloc[-1].strftime("%Y%m%d_%H%M%S")
filename = f"solana_{DAYS}d_4h_{timestamp}.csv"
df.to_csv(filename, index=False)
print(f"Data saved to {filename}")