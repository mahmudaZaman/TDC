import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import random
from tensorflow.keras import layers
from callback import create_callbacks



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


# Text vectorization (tokenization)
max_vocab_length = 10000
max_length = 15
text_vectorizer = TextVectorization(
    max_tokens=max_vocab_length,
    output_mode='int',
    output_sequence_length=max_length)

text_vectorizer.adapt(train_sentences)
random_sentence = random.choice(train_sentences)
print(random_sentence)
print(text_vectorizer([random_sentence]))

words_in_vocub = text_vectorizer.get_vocabulary()
print(words_in_vocub[:5])
print(words_in_vocub[-5:])

# Embedding
embedding = layers.Embedding(input_dim=max_vocab_length,
                             output_dim=128,
                             input_length=max_length,
                             embeddings_initializer='uniform')
print(embedding(text_vectorizer([random_sentence])))

# RNN(LSTM) model
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64)(x)
outputs = layers.Dense(1, activation = "sigmoid")(x)
lstm_model = tf.keras.Model(inputs, outputs, name="LSTM_model")

lstm_model.compile(loss="binary_crossentropy",optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])

callbacks = create_callbacks()
lstm_model_history = lstm_model.fit(train_sentences,
                              train_label,
                              epochs=5,
                              validation_data=(test_sentences, test_label),
                              callbacks=callbacks)

lstm_model_probs = lstm_model.predict(test_sentences)
print(lstm_model_probs[:10])

lstm_model_pred = tf.squeeze(tf.round(lstm_model_probs))
print(lstm_model_pred[:20])

# RNN(GRU) model
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
# x = layers.GRU(64, return_sequences=True)(x)
x = layers.GRU(64)(x)
# x = layers.GlobalAveragePooling1D()(x)
# x = layers.LSTM(64, return_sequences=True)(x)
# x = layers.GRU(64)(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation = "sigmoid")(x)
gru_model = tf.keras.Model(inputs, outputs, name="GRU_model")

gru_model.compile(loss="binary_crossentropy",optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])
gru_model_history = gru_model.fit(train_sentences,
                              train_label,
                              epochs=5,
                              validation_data=(test_sentences, test_label),
                              callbacks=callbacks)

gru_model_probs = gru_model.predict(test_sentences)
print(gru_model_probs[:10])

gru_model_pred = tf.squeeze(tf.round(gru_model_probs))
print(gru_model_pred[:20])

# Bidirectional RNN
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(64))(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation = "sigmoid")(x)
bidirectional_model = tf.keras.Model(inputs, outputs, name="bidirectional_model")

bidirectional_model.compile(loss="binary_crossentropy",optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])
bidirectional_model_history = bidirectional_model.fit(train_sentences,
                              train_label,
                              epochs=5,
                              validation_data=(test_sentences, test_label),
                              callbacks=callbacks)

bidirectional_model_probs = bidirectional_model.predict(test_sentences)
print(bidirectional_model_probs[:10])

bidirectional_model_pred = tf.squeeze(tf.round(bidirectional_model_probs))
print(bidirectional_model_pred[:20])