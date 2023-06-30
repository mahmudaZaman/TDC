import tensorflow as tf

checkpoint_path = "tweet_weights/checkpoint.ckpt"

def create_callbacks():
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,  # set to False to save the entire model
        save_best_only=True,  # save only the best model weights instead of a model every epoch
        save_freq="epoch",  # save every epoch
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        mode='max',
        factor=0.1,
        patience=3,
        verbose=0
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=10,
        verbose=1
    )
    callbacks = [checkpoint, reducelr, earlystop]
    return callbacks
