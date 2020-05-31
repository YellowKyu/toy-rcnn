import tensorflow as tf
import tensorflow.keras.backend as K

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    loss = 1 - (numerator / denominator)
    return loss

def masked_mae_loss(y_true, y_pred):
    bbox_mask = y_true[:, :, :, 0:-1]
    objectness_mask = K.expand_dims(y_true[:, :, :, -1], axis=-1)
    total_positive = K.sum(objectness_mask)
    diff = K.abs(bbox_mask - y_pred)
    diff_masked = diff * objectness_mask
    mean_diff_masked = K.mean(diff_masked, axis=-1)
    total_mean_diff_masked = K.sum(mean_diff_masked)
    new_loss = total_mean_diff_masked / total_positive
    #print(mean_diff_masked.shape, diff_masked.shape, total_positive, total_mean_diff_masked, new_loss)
    #return K.mean(diff)
    return new_loss