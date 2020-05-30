from datetime import datetime
from tensorflow import keras
from datamanager import DataManager
from tensorboard_logger import TensorBoardLogger
from model import RCNNModel
import loss

# generate and pre-process dummy data
dm = DataManager()
train_x, train_y, train_cat, train_mask, test_x, test_y, test_cat, test_mask = dm.gen_toy_detection_datasets()

train_x = train_x.astype("float32")
test_x = test_x.astype("float32")
train_mask = train_mask.astype("float32")

train_x = train_x / 255.0
train_mask = train_mask / 255.0
test_x = test_x / 255.0

# logs and callback
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_logger = TensorBoardLogger(logdir + '/train_loss', test_x)
callbacks = [tensorboard_callback, tensorboard_logger]

model = RCNNModel()

model.model.compile(optimizer='adam', loss=loss.dice_loss)

model.model.fit(train_x, train_mask,
          batch_size=1, epochs=30,
          callbacks=callbacks)