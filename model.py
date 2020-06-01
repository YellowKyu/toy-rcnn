from tensorflow import keras
from tensorflow.keras import layers

class RCNNModel(object):

    def __init__(self):
        inputs = keras.Input(shape=(108, 192, 3))

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(x)

        # objectness score
        objectness = layers.Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same", name="objectness")(x)

        # bounding boxes (tlbr, ratio, 0-1)
        bboxes = layers.Conv2D(filters=4, kernel_size=(1, 1), activation="relu", padding="same", name="bboxes")(x)


        self.model = keras.Model(inputs=inputs, outputs=[objectness, bboxes])
        self.model.summary()

