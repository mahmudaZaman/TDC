import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import random
from data_preprocessing import get_lines, text_cleaner

data = open("dataset/irish_lyrics.txt").read()
corpus = data.lower().split("\n")
print(corpus)

print("before cleaning",corpus[:5])
print(len(corpus))

text = [text_cleaner(line) for line in corpus]
print("after cleaning",text[:5])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
total_words = len(word_index) + 1
print(total_words)


# create input sequences using list of tokens
input_sequences = []

for sentence in text:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,
                                         maxlen=max_sequence_len,
                                         padding='pre'))

# embedding
token_embed = tf.keras.layers.Embedding(total_words,
                               output_dim=128,
                               mask_zero=True,
                               name="token_embedding")
# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
# create one-hot encoding of the labels
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

# Conv1D Model
model = keras.Sequential([
    token_embed,
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.GRU(64)),
    keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model.fit(predictors, label, epochs=100, verbose=1)

seed_text = "Near the presence of my lover"
next_words = 50

for _ in range(next_words):
    seq = tokenizer.texts_to_sequences([seed_text])[0] # returns a list of lists so we use 0 index
    test_pad = keras.preprocessing.sequence.pad_sequences([seq],
                                                          maxlen=max_sequence_len-1,
                                                          padding='pre')
    predicted = np.argmax(model.predict(test_pad, verbose=0), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == int(predicted):
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)