from preprocessing.preprocess_kaggle_dataset import *

dp = DatasetProcessor()
dp.load_dataset_data_from_kaggle()
dp.save_dataset_data()
dp.preprocess_all_metadata()
dp.preprocess_all_data()