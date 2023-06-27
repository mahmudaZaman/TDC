import tensorflow as tf
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Read in the usa_housing dataset
path = "dataset/USA_Housing.csv"
data = pd.read_csv(path)
print(data.head())
print(data.loc[0])
print(data.info())

# Data preprocessing
data.drop('Address', axis = 1, inplace = True)
print(data.head())
print(data.info())
print(data.shape)
print(data.columns)
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train[:5])
print(y_train[:5])
print("X shape: ",X_train.shape, X_test.shape)
print("y shape: ",y_train.shape, y_test.shape)
print("Type of X and y: ", type(X_train), type(y_train))
X_train, X_test, y_train, y_test = X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # DNN model
# # Set random seed
tf.random.set_seed(42)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(28),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.SGD(),
                          metrics=['mae'])

# Fit the model for 200 epochs
model_history = model.fit(X_train, y_train,validation_data = (X_test, y_test),batch_size=32,
                          callbacks=[callback], epochs=500, verbose=1)

# # Evaulate  model
model_loss, model_mae = model.evaluate(X_test, y_test)
print("model_loss, model_mae", model_loss, model_mae )

y_pred = model.predict(X_test)
print("y_pred before", y_pred.shape, y_pred[:5])
y_pred = tf.squeeze(y_pred)
print("y_pred after", y_pred[:5])
print("y_test", y_test[:5])
print("r2_score", r2_score(y_test, y_pred))

# View real and predicted data
comparison = pd.DataFrame({"Y_value": y_test,
                            "pred_value": y_pred})
print(comparison)

model.save("out/usa_housing_model.h5")