import tensorflow as tf
# import numpy as np
import tensorflow_datasets as tfds
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import confusion_matrix
from callback import create_callbacks, checkpoint_path


def preprocess_img(image, label, img_shape=224):
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape to img_shape
    return tf.cast(image, tf.float32), label # return (float32_image, label) tuple

# Get all available datasets in TFDS
datasets_list = tfds.list_builders()

# Set our target dataset and see if it exists
target_dataset = "food101"
print(f"'{target_dataset}' in TensorFlow Datasets: {target_dataset in datasets_list}")


# Load the dataset with the custom download configuration
(train_data, test_data), ds_info = tfds.load(name="food101", # target dataset to get from TFDS
                                             split=["train", "validation"], # what splits of data should we get? note: not all datasets have train, valid, test
                                             shuffle_files=True, # shuffle files on download?
                                             as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                             with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

print(ds_info.features)
class_names = ds_info.features["label"].names
print(class_names)
print("type and shape: ",type(train_data),train_data.shard)
num_of_classes = len(class_names)
print(num_of_classes)

train_one_sample = train_data.take(1)
for image, label in train_one_sample:
  print(f"""
  Image shape: {image.shape}
  Image dtype: {image.dtype}
  Target class from mnist (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
        """)
print(image)
print(tf.reduce_min(image), tf.reduce_max(image))


preprocessed_img = preprocess_img(image, label)[0]
print(f"Image before preprocessing:\n {image[:2]}...,\nShape: {image.shape},\nDatatype: {image.dtype}\n")
print(f"Image after preprocessing:\n {preprocessed_img[:2]}...,\nShape: {preprocessed_img.shape},\nDatatype: {preprocessed_img.dtype}")


# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)
# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

train_one_sample = train_data.take(1)
for image, label in train_one_sample:
  print(f"""
  Image shape: {image.shape}
  Image dtype: {image.dtype}
  Target class from food101 (tensor form): {label}
        """)
# DNN model
# Set random seed
tf.random.set_seed(42)

# Create the model
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False # freeze base model layers

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = base_model(inputs) # set base_model to inference mode only
x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
outputs = tf.keras.layers.Dense(num_of_classes, activation="softmax", name="output_layer")(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=["accuracy"])

callbacks = create_callbacks()

# Fit the model
history = model.fit(train_data,
                               epochs=5,
                               validation_data=test_data,
                               callbacks=callbacks)

# Evaluate  model
model.evaluate(test_data)

y_pred = model.predict(test_data)
print("y pred before: ",y_pred[:5])
y_pred = y_pred.argmax(axis=1)
print("y pred after: ",y_pred[:5])

model.save("out/food101_model.h5")
model.load_weights(checkpoint_path)
loaded_weights_model_results = model.evaluate(test_data)
print("loaded_weights_model_results: ", loaded_weights_model_results)
