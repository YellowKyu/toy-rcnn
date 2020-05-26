import cv2
import sys
import random
import numpy as np

class DataManager(object):

    def __init__(self):
        pass

    def gen_triangle(self, shape):
        max_y = shape[0]
        max_x = shape[1]

        # gen three points
        pt1 = [random.randint(0, max_x), random.randint(0, max_y)]
        pt2 = [random.randint(0, max_x), random.randint(0, max_y)]
        pt3 = [random.randint(0, max_x), random.randint(0, max_y)]
        pts = [pt1, pt2, pt3]

        # gen triangle's rectangular bounding box
        x1, x2, y1, y2 = pt1[0], pt1[0], pt1[1], pt1[1]
        for pt in pts:
            x1 = min(pt[0], x1)
            x2 = max(pt[0], x2)
            y1 = min(pt[1], y1)
            y2 = max(pt[1], y2)

        return np.array(pts), np.array([x1, y1, x2, y2])

    def gen_circle(self, shape):
        max_y = shape[0]
        max_x = shape[1]
        quarter_y = int(max_y / 3)

        # gen circle's center and radius
        ctr_x = random.randint(0, max_x)
        ctr_y = random.randint(0, max_y)
        radius = random.randint(0, quarter_y)

        # gen cercle's rectangular bounding box
        # min/max to make box inside (0, max_x) and (0, max_y)
        x1 = max(0, ctr_x - radius)
        y1 = max(0, ctr_y - radius)
        x2 = min(ctr_x + radius, max_x - 1)
        y2 = min(ctr_y + radius, max_y - 1)

        return ctr_x, ctr_y, radius, np.array([x1, y1, x2, y2])

    def gen_random_shape(self, image, shape):
        shape_choice = random.randint(1,2)
        color = (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254))

        if shape_choice == 1:
            pts, box = self.gen_triangle(shape)
            cv2.drawContours(image, [pts], 0, color, -1)
        elif shape_choice == 2:
            ctr_x, ctr_y, radius, box = self.gen_circle(shape)
            cv2.circle(image, (ctr_x, ctr_y), radius, color, -1)

        return image, box, shape_choice

    def gen_toy_detection_sample(self, num, shape=(108, 192, 3), need_mask_label=False):
        x = []
        y = []
        categories = []
        ctr_mask_labels = []

        for _ in range(num):
            toy_sample = np.zeros(shape, np.uint8)
            ctr_mask_label = np.zeros((shape[0], shape[1], 1), np.uint8)

            box_list = []
            cat_list = []
            num_obj = random.randint(1, 3)
            for _ in range(num_obj):
                image, box, category = self.gen_random_shape(toy_sample, shape)
                box_list.append(box)
                cat_list.append(category)
                if need_mask_label:
                    ctr_mask_label = cv2.rectangle(ctr_mask_label, (box[0], box[1]), (box[2], box[3]), (255), -1)
                # debug drawing
                # image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            x.append(image)
            y.append(box_list)
            categories.append(cat_list)
            ctr_mask_labels.append(ctr_mask_label)

        return np.array(x), np.array(y), np.array(categories), np.array(ctr_mask_labels)

    def gen_toy_detection_datasets(self, train_size=300, test_size=100):
        train_x, train_y, train_cat, train_mask = self.gen_toy_detection_sample(train_size, need_mask_label=True)
        test_x, test_y, test_cat, test_mask = self.gen_toy_detection_sample(test_size, need_mask_label=True)
        return train_x, train_y, train_cat, train_mask, test_x, test_y, test_cat, test_mask

    def gen_mnsit_detection_datasets(self, mnist_path, train_size=300, test_size=100):
        pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    my_manager = DataManager()
    train_x, train_y, train_cat, train_mask, test_x, test_y, test_cat, test_mask = my_manager.gen_toy_detection_datasets()
    print(train_x.shape, train_y.shape, train_cat.shape, train_mask.shape)
    print(test_x.shape, test_y.shape, test_cat.shape, test_mask.shape)
    cv2.imshow('image', train_x[0])
    cv2.imshow('mask', train_mask[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()