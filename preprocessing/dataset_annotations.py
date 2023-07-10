import json
import requests

from ast import literal_eval
from openml_id import fetch_id_dict
from openai_api import prompt_openai

def fetch_table_annotation(id, INFO_DICT):
    """
    example usage:
        fetch_table_annotation(ID_DICT['credit-g'])

    """
    url = f"https://www.openml.org/es//data/data/{id}"
    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)

    json_data = response.json()['_source']['description']
    features = response.json()['_source']['features']
    targets = 'Target name: ' + [feature['name'] for feature in features if 'target' in feature.keys()][0]
    features = 'Features: ' + ', '.join([feature['name'] for feature in features if 'target' not in feature.keys()])
    prompt = "The following is the description of a tabular dataset. Return the information for: \n"\
    "1. the target of the dataset. \n" \
    "2. the features and explanations if exist. Replace all hyphens and/or underscores with spaces. \n"\
    "Do NOT respond anything else than the needed information.\n\n" + json_data + '\n\n' + features
    if id in INFO_DICT.keys():
        return INFO_DICT[id]
    print('using openai api...')
    api_output = prompt_openai(prompt)
    append_dataset_info(id, api_output)
    print(f'item {id} saved to local file to prevent api reusing')
    return api_output

def load_dataset_info():
    INFO_DICT = {}
    with open('dataset_info.txt', 'r') as f:
        for line in f.readlines():
            recorded_key, recorded_value = line.split(',,')
            INFO_DICT[recorded_key] = literal_eval(recorded_value)
    print(INFO_DICT)
    return INFO_DICT

def append_dataset_info(key, value):
    with open('dataset_info.txt', 'a') as f:
        f.write(f'{key},,{repr(value)}\n')

def overwrite_dataset_info(INFO_DICT):
    with open('dataset_info.txt', 'w') as f:
    for key in info.keys():
        f.write(f'{key},,{repr(INFO_DICT[key])}\n')


if __name__ == '__main__':
    # example usage:
    INFO_DICT = load_dataset_info()
    ID_DICT = fetch_id_dict()
    for dataset in ID_DICT.keys()
        dataset_id = ID_DICT['dataset']
        dataset_annotation = f'Info of dataset {dataset_id}: {dataset} is yet retrieved.'
        # Uncomment the following line if you want to use the openai api to 
        # fetch annotation. Using openai API costs money. 

        # dataset_annotation = fetch_table_annotation(dataset_id, INFO_DICT)
        print(dataset_annotation)





