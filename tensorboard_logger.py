import tensorflow as tf

class TensorBoardLogger(tf.keras.callbacks.Callback):

    def __init__(self, logdir):
        self.writer = tf.summary.create_file_writer(logdir)
        self.writer.set_as_default()

    def on_train_batch_end(self, batch, logs=None):
        # if logs is not None:
        #     tf.summary.scalar('dice loss (batch)', data=logs['loss'], step=batch)
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:

            tf.summary.scalar("dice loss (epoch)", logs['loss'], step=epoch)
            self.writer.flush()
