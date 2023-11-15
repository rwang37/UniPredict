from preprocessing.preprocess_kaggle_dataset import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--from-step', 
                    type=str,
                    required=True,
                    help='From which step to start the data preprocessing.')

args = parser.parse_args()
step = args.from_step

if step == 'scratch':
    dp = DatasetProcessor()
    dp.load_dataset_data_from_kaggle()
    dp.save_dataset_data()
    dp.preprocess_all_metadata()
    dp.preprocess_all_data()
elif step == 'round_1':
    # if data are downloaded
    dp = DatasetProcessor(dataset_info_list_path='files/unified/dataset_list/datasets_after_round_1.json')
    # dp.load_dataset_data_from_kaggle()
    # dp.save_dataset_data()
    dp.preprocess_all_metadata()
    dp.preprocess_all_data()
elif step == 'metadata':
    # if metadata is collected
    dp = DatasetProcessor(dataset_info_list_path='files/unified/dataset_list/datasets_after_round_1.json')
    # dp.load_dataset_data_from_kaggle()
    # dp.save_dataset_data()
    # dp.preprocess_all_metadata()
    dp.preprocess_all_data()