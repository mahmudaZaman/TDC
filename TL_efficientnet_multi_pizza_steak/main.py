import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import os
import random
import matplotlib.image as mpimg
import tensorflow_hub as hub
from tensorflow.keras import layers
from callback import create_callbacks, checkpoint_path

efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

def view_random_image(target_dir, target_class):
  target_folder = target_dir+target_class
  random_image = random.sample(os.listdir(target_folder), 1)
  img = mpimg.imread(target_folder + "/" + random_image[0])
  print(f"Image shape: {img.shape}")
  return img

# Setup the train and test directories
train_dir = "dataset/10_food_classes_all_data/train/"
test_dir = "dataset/10_food_classes_all_data/test/"

data_dir = pathlib.Path("dataset/10_food_classes_all_data/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
num_of_classes = len(class_names)
print(class_names, num_of_classes)

num_steak_images_train = len(os.listdir("dataset/10_food_classes_all_data/train/steak"))
print(num_steak_images_train)

img = view_random_image(target_dir="dataset/10_food_classes_all_data/train/",
                        target_class="steak")

print("img", img)
print("img shape", img.shape)


# CNN model
# Set the seed
tf.random.set_seed(42)

# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True
                                   )
valid_datagen = ImageDataGenerator(rescale=1./255)

# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               seed=42)

images, labels = train_data.next()
print(len(images), len(labels))
print("labels: ", labels)

# Download the pretrained model and save it as a Keras layer
feature_extractor_layer = hub.KerasLayer(efficientnet_url,
                                         trainable=False,
                                         name='feature_extraction_layer',
                                         input_shape=(224, 224, 3))


# Create our own model
model = tf.keras.Sequential([
    feature_extractor_layer,
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    layers.Dense(num_of_classes, activation='softmax', name='output_layer')
])

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
                        verbose=1,callbacks=callbacks)

# Evaluate  model
model.evaluate(valid_data)

y_pred = model.predict(valid_data)
print("y pred before: ",y_pred[:5])
y_pred = y_pred.argmax(axis=1)
print("y pred after: ",y_pred[:5])

model.save("out/pizza_model.h5")

# Load in saved model weights and evaluate model
model.load_weights(checkpoint_path)
loaded_weights_model_results = model.evaluate(valid_data)