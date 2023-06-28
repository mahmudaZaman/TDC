import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import os
import random
import matplotlib.image as mpimg
from callback import create_callbacks, checkpoint_path

cpk_path = './happy_sad_best_model.h5'

def view_random_image(target_dir, target_class):
  target_folder = target_dir+target_class
  random_image = random.sample(os.listdir(target_folder), 1)
  img = mpimg.imread(target_folder + "/" + random_image[0])
  print(f"Image shape: {img.shape}")
  return img

# Setup the train and test directories
data_dir = "dataset/happy_sad/"

data_dir = pathlib.Path("dataset/happy_sad/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)

num_steak_images_train = len(os.listdir("dataset/happy_sad/"))
print(num_steak_images_train)

img = view_random_image(target_dir="dataset/happy_sad/",
                        target_class="happy")

print("img", img)
print("img shape", img.shape)


# CNN model
# Set the seed
tf.random.set_seed(42)
# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   rotation_range=20,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True
                                   )


# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(data_dir,
                                               subset='training',
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

valid_data = train_datagen.flow_from_directory(data_dir,
                                               subset='validation',
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

images, labels = train_data.next()
print(len(images), len(labels))
print("labels: ", labels)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=32,
                         kernel_size=(3, 3),
                         padding='same',
                         activation="relu",
                         input_shape=(224, 224, 3)),
  tf.keras.layers.Conv2D(32, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                            padding="same"),
  tf.keras.layers.Conv2D(64, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(2,padding="same"),
  tf.keras.layers.Conv2D(128, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(2,padding="same"),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

callbacks = create_callbacks()

# Fit the model
history = model.fit(train_data,
                        epochs=50,
                        batch_size=16,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data),
                        verbose=1,callbacks=callbacks)

# Evaluate  model
model.evaluate(valid_data)

y_pred = model.predict(valid_data)
print("y pred before: ",y_pred[:5])
y_pred = tf.round(y_pred)
print("y pred after: ",y_pred[:5])

model.save("out/happy_sad_model.h5")

# Load in saved model weights and evaluate model
model.load_weights(checkpoint_path)
loaded_weights_model_results = model.evaluate(valid_data)
print("loaded_weights_model_results: ", loaded_weights_model_results)