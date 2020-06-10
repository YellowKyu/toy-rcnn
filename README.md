# toy-rcnn, a tutorial and introduction to deep learning based object detection

![Results](/images/results.png)

A very simple and limited object detection model just to introduce some basic concept of object detecion based on deep learning.
I hope that this work is easier to understand and more friendly than heavier solution like Detectron, Tensorflow Object Detection API, etc.

The model is mainly implemented with Keras and supports Tensorboard. Prediction results and custom losses are logged with Tensorboard and tf.summary.

the model:

![model](/images/model.png)

# TODO 

- validation/test with metrics (mAP and such)
- multi classes dummy data generation
- add category prediction branch (currently the model is only predicting an objectness score)
