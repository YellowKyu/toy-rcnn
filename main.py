import numpy as np
from datetime import datetime
from tensorflow import keras
from datamanager import DataManager
from tensorboard_logger import TensorBoardLogger
from model import RCNNModel
import loss

# generate and pre-process dummy data
dm = DataManager()
train_x, train_y, train_cat, train_mask, train_mask_y, test_x, test_y, test_cat, test_mask, test_mask_y = dm.gen_toy_detection_datasets()

train_x = train_x.astype("float32")
test_x = test_x.astype("float32")
train_mask = train_mask.astype("float32")
train_mask_y = train_mask_y.astype("float32")

train_x = train_x / 255.0
train_mask = train_mask / 255.0
test_x = test_x / 255.0

# logs and callback
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_logger = TensorBoardLogger(logdir + '/train_loss', test_x, test_y)
callbacks = [tensorboard_callback, tensorboard_logger]

model = RCNNModel()

losses = {"objectness": loss.dice_loss, "bboxes": loss.masked_mae_loss}
# losses = {"objectness": 'mse', "bboxes": 'mae'}

all_train_mask = np.concatenate([train_mask_y, train_mask], axis=-1)

targets = {"objectness": train_mask, "bboxes": all_train_mask}
losses_weights = {"objectness": 1.0, "bboxes": 2.0}

model.model.compile(optimizer='adam', loss=losses, loss_weights=losses_weights)

model.model.fit(train_x, targets,
          batch_size=1, epochs=30,
          callbacks=callbacks)