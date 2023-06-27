import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import random
from tensorflow.keras import layers

from callback import create_callbacks
from naive_bayes import naive_bayes


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

# Naive bayes
naive_model = naive_bayes()
naive_model.fit(train_sentences, train_label)
baseline_score = naive_model.score(test_sentences, test_label)
print(baseline_score*100)

naive_model_pred = naive_model.predict(test_sentences)
print(naive_model_pred[:10])

# dense model
def dense_model():
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    dense_model = tf.keras.Model(inputs, outputs, name="dense_model")
    return dense_model
dense_model = dense_model()
dense_model.compile(loss="binary_crossentropy",optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])
callbacks = create_callbacks()
dense_model_history = dense_model.fit(train_sentences,
                              train_label,
                              epochs=5,
                              verbose=1,
                              validation_data=(test_sentences, test_label),callbacks=callbacks)

simple_dense_model_probs = dense_model.predict(test_sentences)
print(simple_dense_model_probs[:10])

simple_dense_model_pred = tf.squeeze(tf.round(simple_dense_model_probs))
print(simple_dense_model_pred[:20])

# Conv1D
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Conv1D(filters=64, kernel_size=5, strides=1, activation="relu", padding="valid")(x)
x = layers.GlobalMaxPooling1D()(x)
outputs = layers.Dense(1, activation = "sigmoid")(x)
conv_model = tf.keras.Model(inputs, outputs, name="conv1D_model")

conv_model.compile(loss="binary_crossentropy",optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"])

conv_model_history = conv_model.fit(train_sentences,
                              train_label,
                              epochs=5,
                              validation_data=(test_sentences, test_label),
                              callbacks=callbacks)

conv1d_model_probs = conv_model.predict(test_sentences)
print(conv1d_model_probs[:10])

conv1d_model_pred = tf.squeeze(tf.round(conv1d_model_probs))
print(conv1d_model_pred[:20])