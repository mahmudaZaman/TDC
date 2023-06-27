import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
from data_preprocessing import get_lines

data = get_lines('dataset/war_peace_plain.txt')
print(data[:50])

# Tokenizing things from the data
tokenizer = keras.preprocessing.text.Tokenizer()
corpus = data.lower().split(".")
print(corpus[:5])
print(type(corpus))
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
total_words = len(word_index) + 1
print(total_words)

# tokenization
max_tokens = 25000
text_vectorizer = TextVectorization(max_tokens=max_tokens)
text_vectorizer.adapt(corpus)
random_sen = random.choice(corpus)
print("text_vectorizer",text_vectorizer([random_sen]))

# Creating input sequence
input_sequnce = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for tokens in range(1, len(token_list)):
        n_gram_sequence = token_list[:tokens + 1]
        input_sequnce.append(n_gram_sequence)


# Padded Sequences
max_sequence_len = max(len(x) for x in input_sequnce)
padded_seq = np.array(pad_sequences(input_sequnce,maxlen=max_sequence_len, padding='pre'))
print("padded_seq", padded_seq[:10])

# Crete features and lables
X, label = padded_seq[:, :-1], padded_seq[:,-1]  # Features everything expect the next word
print("x", X[:10])
print("y", label[:10])
Y = keras.utils.to_categorical(label, num_classes=total_words)  # OHE
print("y", Y[10])

# embedding
token_embed = tf.keras.layers.Embedding(total_words,
                               output_dim=128,
                               mask_zero=True,
                               name="token_embedding")

# Conv1D Model
model = keras.Sequential([
    token_embed,
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.GRU(64)),
    keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,epochs=100)

# Predictions
seed_text = "Chaitnay works a lot"
next_words = 10

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