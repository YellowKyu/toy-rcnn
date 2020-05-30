import tensorflow as tf
import numpy as np

class TensorBoardLogger(tf.keras.callbacks.Callback):

    def __init__(self, logdir, test_x):
        self.writer = tf.summary.create_file_writer(logdir)
        self.writer.set_as_default()
        self.epoch = 0
        self.test_x = test_x


    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:

            tf.summary.scalar("dice loss (epoch)", logs['loss'], step=epoch)
            self.writer.flush()

        test_input = np.expand_dims(self.test_x[0], axis=0)
        prediction = self.model.predict(test_input)

        tf.summary.image("test_input", test_input, step=epoch)
        tf.summary.image("test_output", prediction, step=epoch)