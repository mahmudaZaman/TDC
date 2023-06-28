import tensorflow as tf
from sklearn.metrics import r2_score
import pandas as pd
from callback import create_callbacks

# Read in the boston_housing dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path='boston_housing.npz', test_split=0.2, seed=113
)
print(x_train[:5])
print(y_train[:5])
print("X shape: ",x_train.shape, x_test.shape)
print("y shape: ",y_train.shape, y_test.shape)
print("Type of X and y: ", type(x_train), type(y_train))

# DNN model
# Set random seed
tf.random.set_seed(42)

# Build the model
model = tf.keras.Sequential([
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(256),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(256),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(256),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

callbacks = create_callbacks()

# Fit the model for 200 epochs
model_history = model.fit(x_train, y_train,validation_data = (x_test, y_test),batch_size=16,
                          callbacks=[callbacks], epochs=50, verbose=1)

# Evaulate  model
model_loss, model_mae = model.evaluate(x_test, y_test)
print("model_loss, model_mae", model_loss, model_mae )

y_pred = model.predict(x_test)
print("r2_score", r2_score(y_test, y_pred))

print("y_pred before", y_pred.shape, y_pred[:5])
# Since the shape of y_pred is 2d (102, 1), we have to squeez y_pred to make a comparison with y_test
y_pred = tf.squeeze(y_pred)
print("y_pred after", y_pred[:5])
print("y_test", y_test[:5])
print("r2_score", r2_score(y_test, y_pred))

# View real and predicted data
comparison = pd.DataFrame({"Y_value": y_test,
                            "pred_value": y_pred})
print(comparison)

model.save("out/boston_housing_model.h5")