import numpy as np
import tensorflow as tf
import cv2
import keras.backend as K
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from datamanager import DataManager
from tensorboard_logger import TensorBoardLogger

dm = DataManager()

# logs and callback
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_logger = TensorBoardLogger(logdir + '/train_loss')
callbacks = [tensorboard_callback, tensorboard_logger]


train_x, train_y, train_cat, train_mask, test_x, test_y, test_cat, test_mask = dm.gen_toy_detection_datasets()

train_x = train_x.astype("float32")
test_x = test_x.astype("float32")
train_mask = train_mask.astype("float32")

train_x = train_x / 255.0
train_mask = train_mask / 255.0
test_x = test_x / 255.0

inputs = keras.Input(shape=(108, 192, 3))

x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(x)
outputs = layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
    loss = 1 - (numerator / denominator)
    return loss

model.compile(optimizer='adam', loss=dice_loss)

model.fit(train_x, train_mask,
          batch_size=1, epochs=30,
          callbacks=callbacks)