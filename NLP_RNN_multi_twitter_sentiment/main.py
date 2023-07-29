import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# Load the Sentiment140 dataset
# Replace 'path_to_sentiment140.csv' with the actual path to your dataset
data = pd.read_csv('dataset/twitter_sentiment.csv', encoding='latin-1', header=None)
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Preprocess the data
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    return text.lower()

data['text'] = data['text'].apply(clean_text)

# Tokenize the text
max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])

# Pad sequences to have the same length
max_seq_length = 100
sequences_padded = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')

# Convert sentiment labels to binary (positive: 4, negative: 0)
data['sentiment'] = (data['sentiment'] == 4).astype(int)
print(data['sentiment'][:-50])

# Split the dataset into train and test sets
train_size = int(0.8 * len(data))
x_train, x_test = sequences_padded[:train_size], sequences_padded[train_size:]
y_train, y_test = data['sentiment'][:train_size], data['sentiment'][train_size:]

# Create the RNN model
embedding_dim = 128
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_seq_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
learning_rate = 0.001
optimizer = Adam(learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Create a learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 5 == 0:
        return lr * 0.9
    return lr

lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# Train the model
batch_size = 32
epochs = 20
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(x_test, y_test), callbacks=[lr_scheduler_callback])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
