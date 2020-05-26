from datamanager import DataManager

dm = DataManager()

train_x, train_y, train_cat, train_mask, test_x, test_y, test_cat, test_mask = dm.gen_toy_detection_datasets()

