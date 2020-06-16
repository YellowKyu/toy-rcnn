import tensorflow as tf
import tensorflow.keras.backend as K

def my_print(y_true, y_pred):
    print('my_print: ', y_true.shape, y_pred.shape)
    return 0.0

def category_mask_loss(y_true, y_pred):
    y_true = K.reshape(y_true, shape=(1, y_pred.shape[1] * y_pred.shape[2], y_pred.shape[3]))
    y_pred = K.reshape(y_pred, shape=(1, y_pred.shape[1] * y_pred.shape[2], y_pred.shape[3]))
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    return K.mean(cross_entropy)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    loss = 1 - (numerator / denominator)
    return loss

def masked_mae_loss(y_true, y_pred):
    # split bbox and objectness mask
    # y_true = tf.image.resize(y_true, (54, 96),
    #                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    bbox_mask = y_true[:, :, :, 0:-1]

    objectness_mask = K.expand_dims(y_true[:, :, :, -1], axis=-1)

    # count number of pixel inside  gt bounding boxes
    total_positive = K.sum(objectness_mask)
    # l1 distance
    diff = K.abs(bbox_mask - y_pred)
    # mask region not inside gt bounding box
    diff_masked = diff * objectness_mask
    # mean over last dimension
    mean_diff_masked = K.mean(diff_masked, axis=-1)

    # loss over gt region inside gt bounding boxes
    total_mean_diff_masked = K.sum(mean_diff_masked)
    new_loss = total_mean_diff_masked / total_positive
    return new_loss