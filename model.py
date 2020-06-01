from tensorflow import keras
from tensorflow.keras import layers

class RCNNModel(object):

    def __init__(self):
        inputs = keras.Input(shape=(108, 192, 3))

        x = layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same")(inputs)

        # objectness score
        objectness = layers.Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same", name="objectness")(x)

        # bounding boxes (tlbr, ratio, 0-1)
        x_1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x)
        x_2 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_1)
        x_3 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_2)
        x_4 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_3)
        x_5 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")(x_4)
        bboxes = layers.Conv2D(filters=4, kernel_size=(3, 3), activation="relu", padding="same", name="bboxes")(x_5)


        self.model = keras.Model(inputs=inputs, outputs=[objectness, bboxes])
        self.model.summary()

