import zipfile
import requests
import pathlib
import numpy as np
import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # URL of the zip file
# zip_url = "https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip"
# path_to_zip = tf.keras.utils.get_file('pizza_steak.zip', origin=zip_url, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'pizza_steak')
#
# # Download the zip file
# response = requests.get(zip_url)
# with open("pizza_steak.zip", "wb") as file:
#     file.write(response.content)
#
# # Unzip the downloaded file
# with zipfile.ZipFile("pizza_steak.zip", "r") as zip_ref:
#     zip_ref.extractall()
#
# data_dir = pathlib.Path("./pizza_steak") # turn our training path into a Python path
# class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
# print(class_names)

# cats and dogs image load

# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
# print("PATH", PATH)
#
# train_dir = os.path.join(PATH, 'train')
# validation_dir = os.path.join(PATH, 'validation')
# print(train_dir, validation_dir)

tf.random.set_seed(42)

# Preprocess data (get all of the pixel values between 1 and 0, also called scaling/normalization)
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    rotation_range=20,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    horizontal_flip=True
#                                    )
# valid_datagen = ImageDataGenerator(rescale=1./255)
#
# # Import data from directories and turn it into batches
# train_data = train_datagen.flow_from_directory(train_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="binary",
#                                                seed=42)
#
# valid_data = valid_datagen.flow_from_directory(validation_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="binary",
#                                                seed=42)
#
# images, labels = train_data.next()
# print(len(images), len(labels))
# print("labels: ", labels)

# flowers image load


