import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import random
import numpy as np
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
print(train_df.head())
print(test_df.head())

print(train_df.loc[0])
print(train_df["Text"][0])

print(train_df["Category"].value_counts())
num_of_classes = len(train_df["Category"].value_counts())
print(num_of_classes)

# shuffle train data
train_df = train_df.sample(frac=1, random_state=42)
print(train_df.head())

# train test split
X = train_df["Text"]
y = train_df["Category"]
train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y,
                                                                            test_size=0.1,
                                                                            random_state=42)

# check the type of features and labels
print(train_sentences[:5],train_labels[:5])
print(type(train_sentences), type(train_labels))

# convert to numpy array
train_sentences, test_sentences, train_labels, test_labels = train_sentences.to_numpy(), test_sentences.to_numpy(), train_labels.to_numpy(), test_labels.to_numpy()
print(train_sentences[:5],train_labels[:5])
print(type(train_sentences), type(train_labels))

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

# some data analysis for tokenize
sent_lens = [len(sentence.split()) for sentence in train_sentences]
print(sent_lens[:10])

avg_len = np.mean(sent_lens)
print(avg_len)

out_len_seq = np.percentile(sent_lens, 95)
print(out_len_seq)

# tokenization
max_tokens = 25000
output_sequence_length = int(out_len_seq)
text_vectorizer = TextVectorization( max_tokens=max_tokens,
                                    output_sequence_length=output_sequence_length)
text_vectorizer.adapt(train_sentences)
random_sen = random.choice(train_sentences)
print(text_vectorizer([random_sen]))

# embedding
bbc_text_vocab = text_vectorizer.get_vocabulary()
token_embed = tf.keras.layers.Embedding(input_dim=len(bbc_text_vocab),
                               output_dim=128,
                               mask_zero=True,
                               name="token_embedding")

# Conv1D Model
inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = token_embed(x)
x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(num_of_classes, activation="softmax")(x)
conv_model = tf.keras.Model(inputs, outputs)

conv_model.compile(loss="categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

conv_model_history =  conv_model.fit(train_sentences,
                               train_labels_one_hot,
                               epochs=5,
                               verbose=1,
                               validation_data=(test_sentences, test_labels_one_hot))

conv_model.evaluate(test_sentences, test_labels_one_hot)
conv_model_pred_probs = conv_model.predict(test_sentences)
print(conv_model_pred_probs[:20])

conv_model_preds = tf.argmax(conv_model_pred_probs, axis=1)
print(conv_model_preds[:20])



# RNN(LSTM) model
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = token_embed(x)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64)(x)
outputs = tf.keras.layers.Dense(num_of_classes, activation="softmax")(x)
lstm_model = tf.keras.Model(inputs, outputs, name="LSTM_model")

lstm_model.compile(loss="categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

lstm_model_history =  lstm_model.fit(train_sentences,
                               train_labels_one_hot,
                               epochs=5,
                               verbose=1,
                               validation_data=(test_sentences, test_labels_one_hot))

lstm_model.evaluate(test_sentences, test_labels_one_hot)
lstm_model_pred_probs = conv_model.predict(test_sentences)
print(lstm_model_pred_probs[:20])

lstm_model_preds = tf.argmax(lstm_model_pred_probs, axis=1)
print(lstm_model_preds[:20])


# Bidirectional RNN
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = token_embed(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(64))(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(num_of_classes, activation = "sigmoid")(x)
bidirectional_model = tf.keras.Model(inputs, outputs, name="bidirectional_model")

bidirectional_model.compile(loss="categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

bidirectional_model_history =  bidirectional_model.fit(train_sentences,
                               train_labels_one_hot,
                               epochs=5,
                               verbose=1,
                               validation_data=(test_sentences, test_labels_one_hot))

bidirectional_model.evaluate(test_sentences, test_labels_one_hot)
bidirectional_model_pred_probs = bidirectional_model.predict(test_sentences)
print(bidirectional_model_pred_probs[:20])

bidirectional_model_preds = tf.argmax(bidirectional_model_pred_probs, axis=1)
print(bidirectional_model_preds[:20])