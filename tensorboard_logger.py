import tensorflow as tf
import numpy as np
import cv2


class TensorBoardLogger(tf.keras.callbacks.Callback):

    def __init__(self, logdir, test_x, test_y):
        super().__init__()
        self.writer = tf.summary.create_file_writer(logdir)
        self.writer.set_as_default()
        self.epoch = 0
        self.test_x = test_x
        self.test_y = test_y
        self.img_h = test_x.shape[1]
        self.img_w = test_x.shape[2]

    def nms(self, bboxes, scores, thresh=0.5):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        scores = scores[:]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def decode_tlbr(self, bboxes, scores):
        b, y, x, c = np.where(scores > 0.1)
        confident_bboxes = []
        confident_scores = []
        for i, j in zip(y, x):
            t, l, b, r = bboxes[0 , i, j]
            s = scores[0, i, j, 0]
            # t *= 108.0
            # b *= 108.0
            # l *= 192.0
            # r *= 192.0
            x1 = int(max(j - l, 0.0))
            y1 = int(max(i - t, 0.0))
            x2 = int(min(j + r, 191))
            y2 = int(min(i + b, 107))
            confident_bboxes.append([x1, y1, x2, y2])
            confident_scores.append(s)
        return np.asarray(confident_bboxes), np.asarray(confident_scores)

    def decode_ratio_xyxy(self, bboxes, scores):
        # flatten mask of prediction to vector
        pred_bbox = np.reshape(bboxes, (bboxes.shape[1] * bboxes.shape[2], 4))
        pred_objectness = np.reshape(scores, (scores.shape[1] * scores.shape[2]))
        #print('all pred : ', pred_bbox.shape, pred_objectness.shape)

        # selecting only confident prediction
        confident_pred_score = pred_objectness[np.where(pred_objectness > 0.1)]
        confident_pred_bbox = pred_bbox[np.where(pred_objectness > 0.1)]

        confident_pred_bbox[:, 0] *= self.img_w
        confident_pred_bbox[:, 1] *= self.img_h
        confident_pred_bbox[:, 2] *= self.img_w
        confident_pred_bbox[:, 3] *= self.img_h
        return confident_pred_bbox, confident_pred_score

    def pred_post_process(self, bboxes, scores):
        # # ratio to xyxy coordinate
        # confident_pred_bbox, confident_pred_score = self.decode_ratio_xyxy(confident_pred_bbox)

        # tlbr to xyxy coordinate
        confident_pred_bbox, confident_pred_score = self.decode_tlbr(bboxes, scores)

        #print('confident pred : ', confident_pred_bbox.shape, confident_pred_score.shape)

        # filter overlapping bboxes with nms
        keep_idx = self.nms(confident_pred_bbox, confident_pred_score, 0.4)
        return confident_pred_bbox[keep_idx], confident_pred_score[keep_idx]

        # return confident_pred_bbox, confident_pred_score

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            tf.summary.scalar("dice loss (epoch)", logs['objectness_loss'], step=epoch)
            tf.summary.scalar("mae loss (epoch)", logs['bboxes_loss'], step=epoch)
            self.writer.flush()

        test_input = np.expand_dims(self.test_x[0], axis=0)
        test_input_bbox = np.expand_dims(self.test_x[0].copy(), axis=0)
        test_input_y = self.test_y[0]

        for box in test_input_y:
            test_input_bbox[0] = cv2.rectangle(test_input_bbox[0], (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

        prediction = self.model.predict(test_input)

        final_bboxes, final_scores = self.pred_post_process(prediction[1], prediction[0])
        #print('final bboxes : ', final_bboxes.shape, final_scores.shape)

        pred_image = cv2.cvtColor(prediction[0][0], cv2.COLOR_GRAY2RGB)
        for box in final_bboxes:
            pred_image = cv2.rectangle(pred_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

        results = cv2.hconcat([test_input_bbox[0], pred_image])
        results = np.expand_dims(results, axis=0)
        tf.summary.image("test_input_output", results, step=epoch)
