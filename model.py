from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class RCNNModel(object):

    def __init__(self, num_category, h=108, w=192, c=3):
        inputs = keras.Input(shape=(h, w, c))

        x = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(inputs)

        # objectness score
        objectness = layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same", name="objectness")(x)

        # bounding boxes (tlbr, ratio, 0-1)
        x_1 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same")(x)
        x_2 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same")(x_1)
        x_3 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same")(x_2)
        x_4 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same")(x_3)
        # x_5 = layers.Conv2D(filters=16, kernel_size=(5, 3), activation="relu", padding="same")(x_4)
        bboxes = layers.Conv2D(filters=4, kernel_size=(5, 5), activation="sigmoid", padding="same", name="bboxes")(x_4)

        x_cat_1 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same")(x)
        x_cat_p_1 = layers.MaxPool2D((2, 2))(x_cat_1)
        x_cat_2 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same")(x_cat_p_1)
        x_cat_p_2 = layers.MaxPool2D((2, 2))(x_cat_2)
        x_cat_3 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation="relu", padding="same")(x_cat_p_2)
        category = layers.Conv2D(filters=num_category, kernel_size=(5, 5), activation="softmax", padding="same", name="category")(x_cat_3)

        self.model = keras.Model(inputs=inputs, outputs=[objectness, bboxes, category])
        self.model.summary()

