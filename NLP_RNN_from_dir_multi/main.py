import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
import pathlib
from tensorflow.keras import utils
from tensorflow.keras import losses

AUTOTUNE = tf.data.AUTOTUNE
def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label


data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

dataset_dir = utils.get_file(
    origin=data_url,
    untar=True,
    cache_dir='stack_overflow',
    cache_subdir='')

dataset_dir = pathlib.Path(dataset_dir).parent

print(dataset_dir)
print(list(dataset_dir.iterdir()))

train_dir = dataset_dir/'train'
test_dir = dataset_dir/'test'
print(list(train_dir.iterdir()))

batch_size = 32
seed = 42

raw_train_ds = utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = utils.text_dataset_from_directory(
    test_dir,
    batch_size=batch_size)

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250
int_vectorize_layer  = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

# Make a text-only dataset (without labels), then call `TextVectorization.adapt`.
train_text = raw_train_ds.map(lambda text, labels: text)
int_vectorize_layer.adapt(train_text)

int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)
# embedding
text_vocab = int_vectorize_layer.get_vocabulary()
token_embed = tf.keras.layers.Embedding(input_dim=len(text_vocab),
                               output_dim=128,
                               mask_zero=True,
                               name="token_embedding")


model = tf.keras.Sequential([
  layers.Embedding(VOCAB_SIZE + 1, 64, mask_zero=True),
  layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
  layers.GlobalMaxPooling1D(),
  layers.Dense(4)
])

# Functional API
# input_shape = (None,)  # Input shape for variable-length sequences
# inputs = tf.keras.Input(shape=input_shape)
# x = layers.Embedding(VOCAB_SIZE + 1, 64, mask_zero=True)(inputs)
# x = layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2)(x)
# x = layers.GlobalMaxPooling1D()(x)
# outputs = layers.Dense(4)(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

history = model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)