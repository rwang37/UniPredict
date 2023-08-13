import kaggle
import os
import json
import pandas as pd
import numpy as np
import openai
import time
import xgboost
import torch

from preprocessing.openai_api import prompt_openai
from preprocessing.xgb import *
from sklearn.calibration import CalibratedClassifierCV
from preprocessing.preprocessing import transform_data
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from math import floor, log10

curr_path = os.path.join(
    os.path.dirname(os.path.realpath(__name__)),
    'files',
    'data'
)

def load_kaggle_matadata(max_dataset_num):
    dataset_list = []
    for i in range(1, 2000):
        if i % 10 == 0:
            print(len(dataset_list))
        temp_list = kaggle.api.dataset_list(
            file_type='csv', 
            tag_ids=13302, 
            max_size=1048576, 
            page=i
        )
        if len(temp_list) == 0 or len(dataset_list) > max_dataset_num:
            break
        dataset_list.extend(temp_list)
    print(dataset_list)

    metadata_list = [d.__dict__['ref'] for d in dataset_list]

    with open('files/data/kaggle_dataset_record.json', 'w+') as f:
        json.dump(metadata_list, f, indent=4)

    return metadata_list

def save_metadata(metadata_list):
    for item in metadata_list:
        author, path = item.split('/')
        save_path = f'{author}-{path}'
        download_path = os.path.join(curr_path, 'kaggle', save_path)

        print(f'downloading and saving file to {download_path}')
        try:
            kaggle.api.dataset_metadata(
                item, 
                path=download_path
            )

            kaggle.api.dataset_download_files(
                item, 
                path=download_path, 
                unzip=True
            )
        except:
            print('failed')
            pass

def preprocess_all_metadata(metadata_list, pivot):
    count = 0
    metadata_list = metadata_list[pivot:]
    for item in metadata_list:
        print(f'Making metadata for {item}. Current progress: {count} metadata saved')
        try:
            preprocess_metadata(item)
            count += 1
        except openai.error.RateLimitError:
            # retry until no ratelimit error
            print('retrying')
            result = None
            while result is None:
                time.sleep(10)
                try:
                    preprocess_metadata(item)
                    result = 'worked'
                except openai.error.RateLimitError:
                    pass
                except:
                    break
        except:
            print('failed')
            pass
        print('\n\n')
    print(count)

def preprocess_metadata(path):
    author, path = path.split('/')
    path = f'{author}-{path}'
    file_path = os.path.join(curr_path, 'kaggle', path)

    files_lst = []
    for root, dirs, files in os.walk(file_path):
        files_lst.extend(files)
    assert len(files_lst) == 2, f'wrong number of files in the folder:{files}'

    metadata_path = os.path.join(file_path, 'dataset-metadata.json')
    files_lst.remove('dataset-metadata.json')
    dataset_path = os.path.join(file_path, files_lst[0])

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    metadata = metadata['description']

    meta, bin_lst, label_lst, target = extract_metadata_from_data(dataset_path, metadata)

    save_path = os.path.join(file_path, 'metadata.json')
    with open(save_path, 'w+') as f:
        json.dump({
            "target": target,
            "metadata": meta,
            "bins": bin_lst,
            "labels": label_lst
        }, f, indent=4)

def extract_metadata_from_data(path, metadata):
    # 1. read file
    data = pd.read_csv(path)
    col = str(data.columns.to_list())

    prompt = (
        "The following is the metadata of a tabular dataset. Return the information for:\n" 
        "    1. the target of the dataset. If no target exists, choose one from the column as target for the dataset to classify.\n"
        "    2. the features and their explanations, or N/A if there are no explanations. Replace all hyphens and/or underscores with spaces.\n\n"
        "Give your output in json. The following is an example output:\n"
        '{\n'
        '    "target": "Age",\\n'
        '    "metadata": "The target of the dataset is Age. \\n Features and their explanations:\\n    gender: an animal\'s gender.\\n    weight: an animal\'s actual weight, in kg." \\n '
        '}\n\n'
        "Do NOT respond anything else than the needed information. Make it brief but informative." 
        "Your responses should only be code, without explanation or formatting.\n\n"
        f"columns:{col}\n\n" 
        f"metadata:{metadata}\n"
        "Provide your response in stringfied JSON format." 
    )

    response = prompt_openai(prompt)
    print(repr(response))
    response = json.loads(response)
    target = response['target']
    metadata = response['metadata']
    if is_numeric_dtype(data[target]) and (data[target] != 0).all():
        dataset_overview = data[target].describe().apply(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
        seg0 = dataset_overview['25%']
        seg1 = dataset_overview['50%']
        seg2 = dataset_overview['75%']

        seg_str0 = f'<{seg0}'
        seg_str1 = f'{seg0} - {seg1}'
        seg_str2 = f'{seg1} - {seg2}'
        seg_str3 = f'>{seg2}'
        return metadata, [seg0, seg1, seg2], [seg_str0, seg_str1, seg_str2, seg_str3], target
    else:
        return metadata, "N/A", "N/A", target

def prepare_data(path):
    # replace the slash '/' in path by '-'
    if '/' in path:
        author, path = path.split('/')
        path = f'{author}-{path}'
    file_path = os.path.join(curr_path, 'kaggle', path)

    # filter out the unqualified datasets
    files_lst = []
    for root, dirs, files in os.walk(file_path):
        files_lst.extend(files)
    assert 'metadata.json' in files_lst, f'Preprocessed metadata not included in this folder: {files_lst}'

    # define paths to datasets and metadata files
    files_lst.remove('dataset-metadata.json')
    files_lst.remove('metadata.json')
    csv_path = [item for item in files_lst if '.csv' in item][0]
    dataset_path = os.path.join(file_path, csv_path)
    metadata_path = os.path.join(file_path, 'metadata.json')

    # open files and extract values
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    dataset = pd.read_csv(dataset_path)
    target_name = metadata['target']
    bin_lst = metadata['bins']
    label_lst = metadata['labels']
    annotations = metadata['metadata']
    print(target_name)
    
    # prepare samples (x), column values, labels (y) and annotations (dataset description/metadata)
    samples = dataset.drop(target_name, axis=1).round(4)
    col = samples.columns.to_list()
    if metadata['bins'] != 'N/A' and dataset[target_name].nunique() > 10:
        segs = [0] + bin_lst + [np.inf]
        labels = pd.cut(dataset[target_name], bins=segs, labels=label_lst)
    else:
        labels = dataset[target_name]
    annotations = [annotations] * len(samples)

    # preprocessing
    xgb_baseline, prompts, output = transform_data(samples, col, labels, annotations)
    
    # rd: raw_data
    # ro: raw_output
    # p: prompts
    # a: annotations
    # l: label_cats
    # o: outputs
    rd_train, rd_test, ro_train, ro_test, p_train, p_test, a_train, a_test, l_train, l_test, o_train, o_test = train_test_split(
        xgb_baseline[0],
        xgb_baseline[1],
        prompts[0],
        prompts[1],
        prompts[2],
        output,
        test_size=0.1,
        random_state=42
    )
    # overwrite output_train so that it only uses the data from the train set
    # prevent leak
    o_train, _ = overwrite_xgb_output(rd_train, ro_train)
    train = ((rd_train, ro_train), (p_train, a_train, l_train), o_train)
    test = ((rd_test, ro_test), (p_test, a_test, l_test), o_test)

    torch.save(train, os.path.join(file_path, 'train_set.pt'))
    torch.save(test, os.path.join(file_path, 'test_set.pt'))
    return (train, test)


def overwrite_xgb_output(samples, labels):
    clf = xgboost.XGBClassifier(n_estimators = 100)
    clf.fit(samples, labels)

    calibrated_clf = sklearn.calibration.CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
    calibrated_clf.fit(samples, labels)
    preds = calibrated_clf.predict_proba(samples)

    auc = calculate_auc(labels, preds)
    outputs = serialize_output(preds)
    return outputs, auc


if __name__ == '__main__':
    # 1. fetch data and save to folders
    # metadata_list = load_kaggle_matadata(2000)
    with open('files/data/kaggle_dataset_record.json', 'r') as f:
        metadata_list = json.load(f)
    # save_metadata(metadata_list)

    # 2. preprocess metadata into the target value, dataset descriptions and regression-classification transformation
    #  - saved to folder/metadata.json
    # preprocess_all_metadata(metadata_list, 1000)

    # 3. 
    metadata_list = metadata_list[0:]
    count = 0
    for item in metadata_list:
        print(f'Preprocessing {item}. Finished {count} items')
        try:
            prepare_data(item)
            count += 1
        except Exception as e:
            print(e)
        print('\n\n')
