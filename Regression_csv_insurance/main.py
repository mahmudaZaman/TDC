import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score

# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(insurance.head())
print(insurance.info())
print(insurance.shape)
print(insurance.columns)

# Preprocess data for the model
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)
# Separate features and labels from data
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ct.fit(X_train)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

print(X_train.loc[0])
print(X_train_normal[0], type(X_train_normal))
print(y_train[:5], type(y_train))

y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
print(y_train[:5], type(y_train))
print("y shape",y_train.shape, y_test.shape)

# DNN model
# Set random seed
tf.random.set_seed(42)

# Create callback
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3)

# Build the model
model = tf.keras.Sequential([
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(512, activation="relu"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(256),
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

# Fit the model
model.fit(X_train_normal, y_train,validation_data = (X_test_normal, y_test),batch_size=16,
                          callbacks=[callback], epochs=500, verbose=1)

# Evaluate  model
model_loss, model_mae = model.evaluate(X_test_normal, y_test)

# Predict with the model
y_pred = model.predict(X_test_normal)

print("y_pred before", y_pred.shape, y_pred[:5])
# Since the shape of y_pred is 2d (268, 1), we have to squeez y_pred to make a comparison with y_test
y_pred = tf.squeeze(y_pred)
print("y_pred after", y_pred[:5])
print("y_test", y_test[:5])
print("r2_score", r2_score(y_test, y_pred))

# View real and predicted data
comparison = pd.DataFrame({"Y_value": y_test,
                            "pred_value": y_pred})
print(comparison)

model.save("out/insurance_model.h5")