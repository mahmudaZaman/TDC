import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def calculate_results(y_true, y_pred):
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

# Read in the usa_housing dataset
path = "dataset/fetal_health.csv"
data = pd.read_csv(path)
print(data.head())
print(data.loc[0])
print(data.info())
#
# # Data preprocessing
print(data.shape)
print(data.columns)

# # Get the label values
print("labels are: ",data["fetal_health"].unique())
print("label value counts: ",data["fetal_health"].value_counts())
num_of_classes = len(data['fetal_health'].unique())
print("num of classes",num_of_classes)

X = data.drop("fetal_health", axis=1)
y = data["fetal_health"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#  Label encoded
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

print(X_train[:5])
print(y_train[:5])
print("X shape: ",X_train.shape, X_test.shape)
print("y shape: ",y_train.shape, y_test.shape)
print("Type of X and y: ", type(X_train), type(y_train))

# DNN model
# Set random seed
tf.random.set_seed(42)

callback = tf.keras.callbacks.EarlyStopping(min_delta=0.001,
    patience=10,
    restore_best_weights=True,)

#  Build the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(512, activation="relu"),
  tf.keras.layers.Dropout(0.50),
  tf.keras.layers.Dense(256, activation="relu"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.50),
  tf.keras.layers.Dense(56, activation="relu"),
  tf.keras.layers.Dense(56, activation="relu"),
  tf.keras.layers.Dense(num_of_classes, activation="softmax")
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

# Fit the model
model_history = model.fit(X_train, y_train,validation_data = (X_test, y_test),batch_size=32,
                          callbacks=[callback], epochs=500, verbose=1)
# Evaluate  model
model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
print("y pred before: ",y_pred[:5])
y_pred = y_pred.argmax(axis=1)
print("y pred after: ",y_pred[:5])

results = calculate_results(y_true=y_test,
                                     y_pred=y_pred)
print("results: ", results)

model.save("out/fatal_health_model.h5")