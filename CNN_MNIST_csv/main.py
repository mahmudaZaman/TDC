import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('dataset/mnist_train.csv')
test_df = pd.read_csv('dataset/mnist_test.csv')

train_labels = train_df.iloc[:, 0]
train_features = train_df.iloc[:, 1:]

test_labels = test_df.iloc[:, 0]
test_features = test_df.iloc[:, 1:]

# Preprocess the data
train_features = train_features / 255.0  # Normalize pixel values to [0, 1]
test_features = test_features / 255.0

# Reshape the features to images (assuming they are flattened)
train_features = train_features.values.reshape(-1, 28, 28, 1)
test_features = test_features.values.reshape(-1, 28, 28, 1)

# Split data into training and testing sets
x_train, y_train = train_features, train_labels.values
x_test, y_test = test_features, test_labels.values

# If read from single csv, then need to split
# x_train, x_test, y_train, y_test = train_test_split(features, labels.values, test_size=0.2, random_state=42)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])

# Create the neural network model
model = tf.keras.Sequential([
    data_augmentation,  # Data augmentation as the first layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)
