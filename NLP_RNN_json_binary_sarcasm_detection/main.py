import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.text import Tokenizer
import random
import numpy as np


data_df = pd.read_json("dataset/Sarcasm_Headlines_Dataset.json", lines = True)
print(data_df.head())
print(data_df.columns)

print(data_df.loc[0])
print(data_df["headline"][0])

print(data_df["is_sarcastic"].value_counts())
num_of_classes = len(data_df["is_sarcastic"].value_counts())
print(num_of_classes)

# shuffle  data
data_df = data_df.sample(frac=1, random_state=42)
print(data_df.head())

# train test split
X = data_df['headline']
y = data_df["is_sarcastic"]
train_sentences, test_sentences, train_labels, test_labels = train_test_split(X, y,
                                                                            test_size=0.1,
                                                                            random_state=42)
# check the type of features and labels
print(train_sentences[:5],train_labels[:5])
print(type(train_sentences), type(train_labels))

# convert to numpy array
train_sentences, test_sentences, train_labels, test_labels = np.array(train_sentences), np.array(test_sentences), train_labels.to_numpy(), test_labels.to_numpy()

# some data analysis for tokenize
sent_lens = [len(sentence.split()) for sentence in train_sentences]
print(sent_lens[:10])
avg_len = np.mean(sent_lens)
print(avg_len)
out_len_seq = np.percentile(sent_lens, 95)
print(out_len_seq)

#converting text to sequences
num_words_ = 5000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=num_words_, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
train_sentences_token = tokenizer.texts_to_sequences(train_sentences)
test_sentences_token = tokenizer.texts_to_sequences(test_sentences)
vocab_size = len(tokenizer.word_index)

# tokenization
max_tokens = 25000
output_sequence_length = int(out_len_seq)
text_vectorizer = TextVectorization( max_tokens=max_tokens, output_sequence_length=output_sequence_length)
text_vectorizer.adapt(train_sentences)
random_sen = random.choice(train_sentences)
print(text_vectorizer([random_sen]))

# embedding
token_embed = tf.keras.layers.Embedding(vocab_size,
                               output_dim=128,
                               mask_zero=True,
                               name="token_embedding")

# Turn our data into TensorFlow Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels))
# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


# Conv1D Model
inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = token_embed(x)
x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
conv_model = tf.keras.Model(inputs, outputs)

conv_model.compile(loss="binary_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

conv_model_history =  conv_model.fit(train_dataset,
                               epochs=3,
                               verbose=1,
                               validation_data=test_dataset)

conv_model.evaluate(test_dataset)
conv_model_pred_probs = conv_model.predict(test_dataset)
print(conv_model_pred_probs[:20])