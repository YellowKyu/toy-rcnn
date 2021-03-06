import numpy as np
from datetime import datetime
from tensorflow import keras
from datamanager import DataManager
from tensorboard_logger import TensorBoardLogger
from model import RCNNModel
import loss

# generate and pre-process dummy data
dm = DataManager()
train_x, train_y, _, train_mask, train_mask_y, train_cat_mask, test_x, test_y, _, test_mask, test_mask_y, test_cat_mask = dm.gen_toy_detection_datasets(train_size=600)

train_x = train_x.astype("float32")
test_x = test_x.astype("float32")
train_mask = train_mask.astype("float32")
train_mask_y = train_mask_y.astype("float32")
train_cat_mask = train_cat_mask.astype("float32")

train_x = train_x / 255.0
train_mask = train_mask / 255.0
test_x = test_x / 255.0
train_cat_mask = train_cat_mask / 255.0

# logs and callback
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_logger = TensorBoardLogger(logdir + '/train_loss', test_x, test_y)
callbacks = [tensorboard_callback, tensorboard_logger]

model = RCNNModel(num_category=3)
losses = {"objectness": loss.dice_loss, "bboxes": loss.masked_mae_loss, "category": loss.category_mask_loss}

all_train_mask = np.concatenate([train_mask_y, train_mask], axis=-1)

targets = {"objectness": train_mask, "bboxes": all_train_mask, "category": train_cat_mask}
losses_weights = {"objectness": 1.0, "bboxes": 1.0, "category": 1.0}

opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

# model.model.compile(optimizer=opt, loss=losses, loss_weights=losses_weights)
model.model.compile(optimizer='adam', loss=losses, loss_weights=losses_weights)

model.model.fit(train_x, targets,
          batch_size=1, epochs=30,
          callbacks=callbacks)