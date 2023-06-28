import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA

from helper import evaluate_preds

df = pd.read_csv("dataset/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
                 parse_dates=["Date"],
                 index_col=["Date"]) # parse the date column (tell pandas column 1 is a datetime)
print(df.head())
print(df.info())

# Only want closing price for each day
bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
print(bitcoin_prices.head())

# Get bitcoin date array
timesteps = bitcoin_prices.index.to_numpy()
prices = bitcoin_prices["Price"].to_numpy()
print(timesteps[:10], prices[:10])

# Create train and test splits the right way for time series data
split_size = int(0.8 * len(prices)) # 80% train, 20% test
X_train, y_train = timesteps[:split_size], prices[:split_size]
X_test, y_test = timesteps[split_size:], prices[split_size:]
print(len(X_train), len(X_test), len(y_train), len(y_test))


# Create a naïve forecast
naive_forecast = y_test[:-1] # Naïve forecast equals every value excluding the last value
print(naive_forecast[:10], naive_forecast[-10:])  # View frist 10 and last 10

naive_results = evaluate_preds(y_true=y_test[1:],
                               y_pred=naive_forecast)
print(naive_results)

# 1,1,2 ARIMA Model
model = ARIMA(bitcoin_prices.values, order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary())