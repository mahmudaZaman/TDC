import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from callback import create_callbacks, checkpoint_path


def calculate_results(y_true, y_pred):
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


# The data has already been sorted into training and test sets for us
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# another way to load data
# fmnist = tf.keras.datasets.fashion_mnist
# (train_data, train_labels), (test_data, test_labels) = fmnist.load_data()

print(train_data[0], train_labels[:5])
print("data type: ",type(train_data), type(train_labels))
print("shapes: ",train_data.shape, train_labels.shape)
print("one image shape: ",train_data[0].shape, train_labels[0].shape)
num_of_classes = len(np.unique(train_labels))
print(num_of_classes)

# Data preprocessing
train_data = train_data / 255.0
test_data = test_data / 255.0
print("After normalizing: ",train_data[0])

# DNN model
# Set random seed
tf.random.set_seed(42)

callback = tf.keras.callbacks.EarlyStopping(min_delta=0.001,
    patience=10,
    restore_best_weights=True,)

# Create the model
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
  tf.keras.layers.MaxPool2D(2,2),
  # tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
  # tf.keras.layers.MaxPool2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation="relu"),
  tf.keras.layers.Dense(128, activation="relu"),
  tf.keras.layers.Dense(num_of_classes, activation="softmax") # output shape is 10, activation is softmax
])

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

callbacks = create_callbacks()

# Fit the model
history = model.fit(train_data,
                    train_labels,
                    epochs=500,
                    validation_data=(test_data, test_labels),
                    callbacks=callbacks,verbose=1)

# Evaluate  model
model.evaluate(test_data, test_labels)

y_pred = model.predict(test_data)
print("y pred before: ",y_pred[:5])
y_pred = y_pred.argmax(axis=1)
print("y pred after: ",y_pred[:5])

results = calculate_results(y_true=test_labels,
                                     y_pred=y_pred)
print("results: ", results)

print(confusion_matrix(y_true=test_labels,
                 y_pred=y_pred))

model.save("out/fashion_mnist_model.h5")

# Load in saved model weights and evaluate model
model.load_weights(checkpoint_path)
loaded_weights_model_results = model.evaluate(test_data, test_labels)