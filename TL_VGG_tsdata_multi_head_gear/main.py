import tensorflow as tf
import numpy as np
import pathlib
import os
import random
import matplotlib.image as mpimg
from tensorflow.keras import layers
from callback import create_callbacks, checkpoint_path


def view_random_image(target_dir, target_class):
  target_folder = target_dir+target_class
  random_image = random.sample(os.listdir(target_folder), 1)
  img = mpimg.imread(target_folder + "/" + random_image[0])
  print(f"Image shape: {img.shape}")
  return img

# Setup the train and test directories
train_dir = "dataset/train/"
test_dir = "dataset/test/"

train_dir = pathlib.Path("dataset/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in train_dir.glob('*')])) # created a list of class_names from the subdirectories
num_of_classes = len(class_names)
print(class_names, num_of_classes)

num_images_train = len(os.listdir("dataset/train/FEZ"))
print(num_images_train)

img = view_random_image(target_dir="dataset/train/",
                        target_class="FEZ")

print("img", img)
print("img shape", img.shape)

# CNN model
# Set the seed
tf.random.set_seed(42)

# Import data from directories and turn it into batches
train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                               seed=123,
                                               batch_size=32,
                                               image_size=(224, 224),
                                               label_mode="categorical")

valid_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                               seed=123,
                                               batch_size=32,
                                               image_size=(224, 224),
                                               label_mode="categorical")


# Check out the class names of our dataset
print(train_data.class_names)
# See an example batch of data
for images, labels in train_data.take(1):
  print(images, labels)


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomHeight(0.2),
  layers.RandomWidth(0.2),
  layers.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
], name ="data_augmentation")


# Download the pretrained model and save it as a Keras layer
base_model = tf.keras.applications.vgg16.VGG16(weights="imagenet",
                       include_top=False,
                       input_shape=(224, 224, 3))


base_model.trainable = False

# Create our own model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = data_augmentation(inputs)
x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
outputs = tf.keras.layers.Dense(num_of_classes, activation="softmax", name="output_layer")(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

callbacks = create_callbacks()

# Fit the model
history = model.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data),batch_size=32,
                        callbacks=callbacks,
                        verbose=1)

# Evaluate  model
model.evaluate(valid_data)

y_pred = model.predict(valid_data)
print("y pred before: ",y_pred[:5])
y_pred = y_pred.argmax(axis=1)
print("y pred after: ",y_pred[:5])

model.save("out/head_gear_model.h5")

# Load in saved model weights and evaluate model
model.load_weights(checkpoint_path)
loaded_weights_model_results = model.evaluate(valid_data)