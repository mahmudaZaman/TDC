import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from callback import create_callbacks, checkpoint_path
from helper import evaluate_preds, make_windows, make_train_test_splits, make_preds

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

HORIZON = 7
WINDOW_SIZE = 30

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)

# View the first 3 windows/labels
for i in range(3):
  print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)
print(train_windows[:3], type(train_windows))
print(train_labels[:3], type(train_labels))


tf.random.set_seed(42)

# Construct model
model = tf.keras.Sequential([
  layers.Dense(128, activation="relu"),
  layers.Dense(HORIZON, activation="linear")
])

# Compile model
model.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"])
callbacks = create_callbacks()

# Fit model
model.fit(x=train_windows, # train windows of 7 timesteps of Bitcoin prices
            y=train_labels, # horizon value of 1 (using the previous 7 timesteps to predict next day)
            epochs=100,
            verbose=1,
            batch_size=128,
            validation_data=(test_windows, test_labels),
          callbacks = callbacks)


# Evaluate model on test data
model.evaluate(test_windows, test_labels)
model.save("out/forcasting_model.h5")

model.load_weights(checkpoint_path)
loaded_weights_model_results = model.evaluate(test_windows, test_labels)
print("loaded_weights_model_results: ", loaded_weights_model_results)


# Make predictions using model_1 on the test dataset and view the results
model_preds = make_preds(model, test_windows)

# Evaluate preds
model_results = evaluate_preds(y_true=tf.squeeze(test_labels), # reduce to right shape
                                 y_pred=model_preds)
print(model_results)
