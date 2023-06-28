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

HORIZON = 1 # predict 1 step at a time
WINDOW_SIZE = 7 # use a week worth of timesteps to predict the horizon

full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)

# View the first 3 windows/labels
for i in range(3):
  print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)
print(train_windows[:3], type(train_windows))
print(train_labels[:3], type(train_labels))

# Check data sample shapes
print(train_windows[0].shape)
# Before we pass our data to the Conv1D layer, we have to reshape it in order to make sure it works
x = tf.constant(train_windows[0])
expand_dims_layer = layers.Lambda(lambda x: tf.expand_dims(x, axis=1)) # add an extra dimension for timesteps
print(f"Original shape: {x.shape}") # (WINDOW_SIZE)
print(f"Expanded shape: {expand_dims_layer(x).shape}") # (WINDOW_SIZE, input_dim)
print(f"Original values with expanded shape:\n {expand_dims_layer(x)}")

tf.random.set_seed(42)

# Construct model
model = tf.keras.Sequential([
    layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
    # resize the inputs to adjust for window size / Conv1D 3D input requirements
    layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
    # we can add LSTM layes as well
    # layers.LSTM(128, activation="relu", return_sequences=True), # this layer will error if the inputs are not the right shape
    # layers.LSTM(128, activation="relu"), # using the tanh loss function results in a massive error
    layers.Dense(HORIZON)
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
