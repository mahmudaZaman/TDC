import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import tensorflow_hub as hub
from callback import create_callbacks, checkpoint_path
from data_preprocessing import get_lines, text_data_preprocess, calculate_results


train_lines = get_lines('dataset/train.txt')
print(train_lines[:5])

train_samples = text_data_preprocess('dataset/train.txt')
test_samples = text_data_preprocess('dataset/test.txt')
print(len(train_samples), len(test_samples))

train_df = pd.DataFrame(train_samples)
test_df = pd.DataFrame(test_samples)
print(train_df.head())
print(train_df.info())

# extract feature and label from dataframe
train_sentences = train_df["text"].to_numpy()
test_sentences = test_df["text"].to_numpy()
train_labels = train_df["target"].to_numpy()
test_labels = test_df["target"].to_numpy()

print(train_sentences[:5])
print(train_labels[:5])
print(type(train_sentences), type(train_labels))

# get number of classes
num_of_classes = len(train_df["target"].value_counts())
print(num_of_classes)

# convert labels in to one hot encoded
one_hot_encoder = OneHotEncoder(sparse_output=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.fit_transform(test_labels.reshape(-1,1))
print(train_labels_one_hot[:5])

# convert labels in to label encoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
print(train_labels_encoded[:5])

# USE model
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape = [],
                                        dtype=tf.string,
                                        trainable = False,
                                        name = "USE")

# Sequenial API for transfer learning
use_model = tf.keras.Sequential([
    tf_hub_embedding_layer,
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_of_classes, activation="softmax")
    ], name ="USE_model"
)

# Functional API for transfer learning
# inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
# x = tf_hub_embedding_layer(inputs)
# x = tf.keras.layers.Dense(128, activation="relu")(x)
# x = tf.keras.layers.Dense(128, activation="relu")(x)
# outputs = tf.keras.layers.Dense(num_of_classes, activation="softmax")(x)
# use_model_func = tf.keras.Model(inputs,outputs)

use_model.compile(loss="categorical_crossentropy",optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])

# callbacks = create_callbacks()

use_model_history = use_model.fit(train_sentences,train_labels_one_hot ,
                              steps_per_epoch=int(0.1 * len(train_sentences)), # only fit on 10% of batches for faster training time
                              epochs=10,
                              validation_data=(test_sentences,test_labels_one_hot),
                              validation_steps=int(0.1 * len(test_sentences)))

use_model.evaluate(test_sentences, test_labels_one_hot)

use_model_pred_probs = use_model.predict(train_sentences)
print(use_model_pred_probs)

use_model_preds = tf.argmax(use_model_pred_probs, axis=1)
print(use_model_preds)

use_model.save("out/skimlit_model.h5")
# use_model.load_weights(checkpoint_path)
# loaded_weights_model_results = use_model.evaluate(test_sentences, test_labels_one_hot)
# print("loaded_weights_model_results: ", loaded_weights_model_results)