import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from scipy.optimize import linear_sum_assignment

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    loss = 1 - (numerator / denominator)
    return loss


def masked_mae_loss(y_true, y_pred):
    # split bbox and objectness mask
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

def l1_loss(A, B):
    rshpA = K.expand_dims(A, axis=1)
    rshpB = K.expand_dims(B, axis=0)
    diff = K.abs(rshpA - rshpB)
    diff_avg = K.mean(diff, axis=-1)
    return diff_avg

def hungarian_loss(losses):
    row_ind, col_ind = linear_sum_assignment(losses)
    idx = [[i, j] for i, j in zip(row_ind, col_ind)]
    return idx
    # val = [losses[i, j].numpy() for i, j in zip(row_ind, col_ind)]
    # min_losses = np.array(val).astype(np.float32)
    # return min_losses

def matching_loss(y_true, y_pred):
    squeezed_y_true = K.squeeze(y_true, axis=0)
    dist_loss = l1_loss(squeezed_y_true, y_pred)
    idx = tf.py_function(func=hungarian_loss, inp=[dist_loss], Tout=tf.int32)
    idx.set_shape((5, 2))
    min_val = tf.gather_nd(dist_loss, idx)
    min_val.set_shape((5))
    print(K.mean(min_val).shape, K.mean(dist_loss).shape)
    # print(min_vals.shape)
    #return K.mean(min_val)
    loss = K.mean(dist_loss) + K.mean(min_val)
    # loss_f = loss - K.mean(dist_loss)
    return loss
