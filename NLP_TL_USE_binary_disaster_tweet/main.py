import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import tensorflow_hub as hub
from tensorflow.keras import layers
from callback import create_callbacks, checkpoint_path

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
print(train_df.head())
print(test_df.head())

print(train_df.loc[0])
print(train_df["text"][0])

print(train_df["target"].value_counts())

# shuffle train data
train_df = train_df.sample(frac=1, random_state=42)
print(train_df.head())

# train test split
X = train_df["text"]
y = train_df["target"]
train_sentences, test_sentences, train_label, test_label = train_test_split(X, y,
                                                                            test_size=0.1,
                                                                            random_state=42)

# check the type of features and labels
print(train_sentences[:5],train_label[:5])
print(type(train_sentences), type(train_label))

# convert to numpy array
train_sentences, test_sentences, train_label, test_label = train_sentences.to_numpy(), test_sentences.to_numpy(), train_label.to_numpy(), test_label.to_numpy()
print(train_sentences[:5],train_label[:5])
print(type(train_sentences), type(train_label))

# Transfer learning (USE)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

print(embeddings)
print(embeddings[0].shape)

sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape = [],
                                        dtype=tf.string,
                                        trainable = False,
                                        name = "USE")

# sequential
# use_model = tf.keras.Sequential([
#     sentence_encoder_layer,
#     layers.Dense(128, activation="relu"),
#     layers.Dense(128, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
#     ], name ="USE_model")

# functional
inputs = layers.Input(shape=[], dtype=tf.string)
pretrained_embedding = sentence_encoder_layer(inputs) # tokenize text and create embedding
x = layers.Dense(128, activation="relu")(pretrained_embedding) # add a fully connected layer on top of the embedding
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x) # create the output layer
use_model = tf.keras.Model(inputs=inputs, outputs=outputs)

use_model.compile(loss="binary_crossentropy",optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])

callbacks = create_callbacks()

use_model_history = use_model.fit(train_sentences,
                              train_label,
                              epochs=5,
                              validation_data=(test_sentences, test_label),
                              callbacks=callbacks)

use_model.evaluate(test_sentences, test_label)
use_model_probs = use_model.predict(test_sentences)
print(use_model_probs[:10])

use_model_pred = tf.squeeze(tf.round(use_model_probs))
print(use_model_pred[:20])

use_model.save("out/tweet_model.h5")
use_model.load_weights(checkpoint_path)
loaded_weights_model_results = use_model.evaluate(test_sentences, test_label)
print("loaded_weights_model_results: ", loaded_weights_model_results)