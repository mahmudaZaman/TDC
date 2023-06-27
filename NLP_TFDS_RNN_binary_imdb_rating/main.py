import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import random
from callback import create_callbacks, checkpoint_path

dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

print(info.features)
class_names = info.features["label"].names
print(class_names)

for example, label in train_dataset.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())

BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# tokenization
VOCAB_SIZE=1500
text_vectorizer = TextVectorization(max_tokens=VOCAB_SIZE)
text_vectorizer.adapt(train_dataset.map(lambda text, label: text))

# embedding
text_vocab = text_vectorizer.get_vocabulary()
token_embed = tf.keras.layers.Embedding(input_dim=len(text_vocab),
                               output_dim=128,
                               mask_zero=True,
                               name="token_embedding")

# RNN (lstm)
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = token_embed(x)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64)(x)
outputs = tf.keras.layers.Dense(1, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs, name="LSTM_model")

# model = tf.keras.Sequential([
#     text_vectorizer,
#     token_embed,
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1,  activation="sigmoid")
# ])

model.compile(loss="binary_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=2,
                    validation_data=test_dataset,
                    validation_steps=4)