import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Load Fashion MNIST dataset
dataset_name = "fashion_mnist"
data, info = tfds.load(dataset_name, split=tfds.Split.TRAIN, with_info=True)

# Preprocess the data
def preprocess_data(entry):
    image = tf.cast(entry['image'], tf.float32) / 255.0  # Normalize pixel values to [0, 1]
    label = tf.one_hot(entry['label'], depth=10)        # One-hot encode labels
    return image, label

# Apply preprocessing and split the data
train_data = data.map(preprocess_data).batch(32).shuffle(buffer_size=1024)

# Create the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# Create the improved model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=5)

# Evaluate the model on test data
test_data, test_info = tfds.load(dataset_name, split=tfds.Split.TEST, with_info=True)
test_data = test_data.map(preprocess_data).batch(32)
test_loss, test_accuracy = model.evaluate(test_data)
print("Test accuracy:", test_accuracy)