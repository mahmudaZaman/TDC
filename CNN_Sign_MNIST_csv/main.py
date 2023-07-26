import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
from callback import checkpoint_path, create_callbacks

train = pd.read_csv('Dataset/sign_mnist_train.csv')
test = pd.read_csv('Dataset/sign_mnist_test.csv')

print(train.head())
print(train.loc[0])
print(train.info())

print(test.head())
print(test.loc[0])
print(test.info())

print(train.shape,test.shape)

y_train =train['label']
y_test =test['label']
X_train=train.drop('label',axis=1)
X_test=test.drop('label',axis=1)

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

print(X_train[:5])
print(y_train[:5])
print("X shape: ",X_train.shape, X_test.shape)
print("y shape: ",y_train.shape, y_test.shape)
print("Type of X and y: ", type(X_train), type(y_train))

# optional
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
valid_datagen = ImageDataGenerator(rescale=1./255)
datagen.fit(X_train)

train_data = datagen.flow(X_train,y_train,batch_size=128)
# optional

model = Sequential()
model.add(Conv2D(35, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(50, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=26, activation='softmax'))

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history = model.fit(train_data,
                        epochs=10,
                        validation_data=(X_test,y_test),batch_size=32,
                        verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

y_pred = model.predict(X_test)
print("y pred before: ",y_pred[:5])
y_pred = y_pred.argmax(axis=1)
print("y pred after: ",y_pred[:5])

model.save("out/fatal_health_model.h5")