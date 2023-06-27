import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import random
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

# TF-IDF Multinomial Naive Bayes
naive_model = Pipeline([
  ("tf-idf", TfidfVectorizer()),
  ("clf", MultinomialNB())
])

naive_model.fit(X=train_sentences,
            y=train_labels_encoded)

print(naive_model.score(X=test_sentences,
              y=test_labels_encoded))

naive_preds = naive_model.predict(test_sentences)
print(naive_preds)

# some data analysis for tokenize
sent_lens = [len(sentence.split()) for sentence in train_sentences]
print(sent_lens[:10])

avg_len = np.mean(sent_lens)
print(avg_len)

out_len_seq = np.percentile(sent_lens, 95)
print(out_len_seq)

# tokenization
max_tokens = 68000
output_sequence_length = int(out_len_seq)
text_vectorizer = TextVectorization( max_tokens=max_tokens,
                                    output_sequence_length=output_sequence_length)
text_vectorizer.adapt(train_sentences)
random_sen = random.choice(train_sentences)
print(text_vectorizer([random_sen]))

# embedding
rct_20k_text_vocab = text_vectorizer.get_vocabulary()
print(f"Number of words in vocabulary: {len(rct_20k_text_vocab)}"),
print(f"Most common words in the vocabulary: {rct_20k_text_vocab[:5]}")
print(f"Least common words in the vocabulary: {rct_20k_text_vocab[-5:]}")

token_embed = tf.keras.layers.Embedding(input_dim=len(rct_20k_text_vocab),
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
print(conv_model_pred_probs)

conv_model_preds = tf.argmax(conv_model_pred_probs, axis=1)
print(conv_model_preds)

# conv_model_results = calculate_results(y_true=test_labels_encoded,
#                                     y_pred=conv_model_preds)
# print(conv_model_results)

# tfds batch dataset (much much faster)
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = token_embed(x)
x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(num_of_classes, activation="softmax")(x)
conv_tfds_model = tf.keras.Model(inputs, outputs)

conv_tfds_model.compile(loss="categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

conv_model_history_with_batch = conv_tfds_model.fit(train_dataset,
                              steps_per_epoch=int(0.1 * len(train_dataset)),
                              epochs=5,
                              validation_data=test_dataset,
                              verbose=1,
                              validation_steps=int(0.1 * len(test_dataset)))


conv_tfds_model.evaluate(test_dataset)
conv_model_tfds_pred_probs = conv_tfds_model.predict(test_dataset)
print(conv_model_tfds_pred_probs)

conv_model_tfds_preds = tf.argmax(conv_model_tfds_pred_probs, axis=1)
print(conv_model_tfds_preds)