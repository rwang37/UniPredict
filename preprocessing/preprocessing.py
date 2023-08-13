import openml
import torch

from sklearn.model_selection import train_test_split
from .openml_id import fetch_id_dict
from .dataset_annotations import fetch_table_annotation, load_dataset_info
from .xgb import get_XGB_classification
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from sklearn.preprocessing import LabelEncoder


IGNORE_LIST = [
    'HotpotQA_distractor', 
    'DBLP-QuAD',
    'German-Credit-Risk',
    'web_questions',
    'Give-Me-Some-Credit',
    'Tour-and-Travels-Customer-Churn-Prediction',
    'freMTPL2freq',
    'Mammographic-Mass-Data-Set',
    'Pulsar-Dataset-HTRU2',
    'online-shoppers-intention'
    ]

def preprocess_data(path, ID_DICT, INFO_DICT, numericalize=False):
    """
    Get the data from path and return a dict object for key/value pairs

    Input:
     - path: the path to the openml dataset, e.g. 'credit-g'
    
    For a dataset with n rows of k features:
    Output:
     - samples: the features of the dataset in the format of pandas dataframe (n*k). 
     - column_names: the names of the columns (k).
     - labels: the value of the labels (n*1).
     - annotations: the dataset description (n*1). All instances from the same dataset has the same annotation value.
    """
    dataset = openml.datasets.get_dataset(path)
    samples, labels, masks, column_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)

    annotations = [fetch_table_annotation(ID_DICT[path], INFO_DICT)]*len(samples)
    return samples, column_names, labels, annotations


def data_to_prompt(data, column_names):
    listed_data = data.values.tolist()
    prompts = []
    for row in listed_data:
        prompt = []
        for i in range(len(column_names)):
            column = column_names[i].replace('_', ' ')
            value = row[i]
            prompt_segment = f'{column} is {value}; '
            prompt.append(prompt_segment)
        prompt = ''.join(prompt)[:-2] + '.\n'
        prompts.append(prompt)
    return prompts


def label_to_prompt(label, length):
    classification = []
    prompt = []
    for key in label.keys():
        classification.append(f'class {label[key]}: xxx; ')
        prompt_segment = f'class {label[key]} stands for "{key}"; '
        prompt.append(prompt_segment)
    # Class 0 is xxx; class 1 is xxx; where class 0 stands for yyy; class 1 stands for yyy.
    full_prompt = ''.join(classification) + 'where ' + ''.join(prompt)[:-2] + '.\n'
    return [full_prompt] * length


def prepare_data_by_dataset(path):
    ## use it in the outside
    ID_DICT = fetch_id_dict()
    ##

    INFO_DICT = load_dataset_info()
    # print(f'IGNORE LIST: {IGNORE_LIST}')

    if path in IGNORE_LIST:
        raise Exception(f'{path} is in the IGNORE LIST.')
    print(f'\n\n{path}\n\n')

    data, col, labels, annotation = preprocess_data(path, ID_DICT, INFO_DICT)

    return transform_data(data, col, labels, annotation)


def transform_data(data, col, labels, annotations):
    prompt = data_to_prompt(data, col)

    categories = [cat for cat in set(labels.to_list())]
    cat_dict = {categories[i]: i for i in range(len(categories))}
    # print(cat_dict)
    print(len(prompt))
    assert len(prompt) < 20000, 'Too many samples in the file. Skipping...'

    output, auc = get_XGB_classification(data, labels, col)
    label_cat = label_to_prompt(cat_dict, len(prompt))
    
    # prompts, outputs, annotations, label_cats
    raw_data, raw_label = numericalize(data, labels, col)

    return (raw_data, raw_label), (prompt, annotations, label_cat), (output)


def numericalize(samples, labels, column_names):
    samples = samples.reset_index(drop=True)
    for i in range(len(column_names)):
        col = column_names[i]
        if is_numeric_dtype(samples[col]) or is_integer_dtype(samples[col]):
            continue
        categories = [cat for cat in set(samples[col].to_list())]
        cat_dict = {categories[i]: i for i in range(len(categories))}
        samples[col] = samples[col].map(cat_dict).astype(int)
    categories = [cat for cat in set(labels.to_list())]
    cat_dict = {categories[i]: i for i in range(len(categories))}
    labels = labels.map(cat_dict).astype(int)
    return samples.to_numpy(), labels.to_numpy().reshape(-1, 1)



