import tensorflow as tf
checkpoint_path = "boston_housing_weights/checkpoint.ckpt"
def create_callbacks():
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,  # set to False to save the entire model
        save_best_only=True,  # save only the best model weights instead of a model every epoch
        save_freq="epoch",  # save every epoch
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        verbose=1
    )
    callbacks = [checkpoint, earlystop]
    return callbacks