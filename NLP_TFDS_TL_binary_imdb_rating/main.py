import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers
from callback import create_callbacks, checkpoint_path

(train_data, test_data), ds_info = tfds.load(name="imdb_reviews", # target dataset to get from TFDS
                                             split=["train", "test"], # what splits of data should we get? note: not all datasets have train, valid, test
                                             shuffle_files=True, # shuffle files on download?
                                             as_supervised=True, # download data in tuple format (sample, label), e.g. (image, label)
                                             with_info=True) # include dataset metadata? if so, tfds.load() returns tuple (data, ds_info)

print(ds_info.features)
class_names = ds_info.features["label"].names
print(class_names)
print("type and shape: ",type(train_data),train_data.shard)
num_of_classes = len(class_names)
print(num_of_classes)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)


# BUFFER_SIZE = 10000
# BATCH_SIZE = 64
# train_dataset = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
sentence_encoder_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

# functional
inputs = layers.Input(shape=[], dtype=tf.string)
pretrained_embedding = sentence_encoder_layer(inputs) # tokenize text and create embedding
x = layers.Dense(128, activation="relu")(pretrained_embedding) # add a fully connected layer on top of the embedding
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x) # create the output layer
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=3,
                    validation_data=test_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

model.save("out/imdb_model.h5")