import openml
import torch

from sklearn.model_selection import train_test_split
from openml_id import fetch_id_dict
from dataset_annotations import fetch_table_annotation, load_dataset_info
from xgb import get_XGB_classification


IGNORE_LIST = ['HotpotQA_distractor', 'DBLP-QuAD']

def preprocess_data(path, ID_DICT, INFO_DICT, numericalize=False):
    """
    Get the data from path and return a dict object for key/value pairs

    Input:
     - path: the path to the openml dataset, e.g. 'credit-g'
     - numericalize: *deprecated*
    
    For a dataset with n rows of k features:
    Output:
     - samples: the features of the dataset in the format of pandas dataframe (n*k). 
     - column_names: the names of the columns (k).
     - labels: the value of the labels (n*1).
     - annotations: the dataset description (n*1). All instances from the same dataset has the same annotation value.
    """
    dataset = openml.datasets.get_dataset(path)
    samples, labels, masks, column_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
        )

    annotations = [fetch_table_annotation(ID_DICT[path], INFO_DICT)]*len(samples)
    return samples, column_names, labels, annotations


def data_to_prompt(data, column_names):
    listed_data = data.values.tolist()
    prompts = []
    for row in listed_data:
        prompt = ''
        for i in range(len(column_names)):
            column = column_names[i].replace('_', ' ')
            value = row[i]
            prompt_segment = f'{column} is {value}; '
            prompt += prompt_segment
        prompt = prompt[:-1] + '\n'
        prompts.append(prompt)
    return prompts


def label_to_prompt(label, length):
    prompt = ''
    for key in label.keys():
        prompt_segment = f'{label[key]} for "{key}"; '
        prompt += prompt_segment
    return [prompt + '\n'] * length


def prepare_all_data(paths, numericalize=False):
    ID_DICT = fetch_id_dict()
    INFO_DICT = load_dataset_info()
    print(f'IGNORE LIST: {IGNORE_LIST}')

    prompts = []
    outputs = []
    annotations = []
    label_cats = []
    for i in range(len(paths)):
        if paths[i] in IGNORE_LIST:
            print(f'\n\n{paths[i]} skipped\n\n')
            continue
        print(f'\n\n{paths[i]}\n\n')
        try:
            data, col, labels, annotation = preprocess_data(paths[i], ID_DICT, INFO_DICT, numericalize)
            prompt = data_to_prompt(data, col)

            categories = [cat for cat in set(labels.to_list())]
            cat_dict = {categories[i]: i for i in range(len(categories))}
            print(cat_dict)
            print(len(prompt))

            # labels = labels.map(cat_dict).astype(int)
            # labels = labels.to_list()
            # all_sample_labels = labels
            output, auc = get_XGB_classification(data, labels, col)
        except Exception as e:
            print(e)
            # raise e
            print(f'Something went wrong in dataset {paths[i]}. Skipping...')
            continue

        prompts.extend(prompt)
        # outputs.extend(all_sample_labels)
        outputs.extend(output)
        annotations.extend(annotation)
        label_cats.extend(label_to_prompt(cat_dict, len(prompt)))

    train_prompts, test_prompts, train_outputs, test_outputs, train_annotations, test_annotations, train_labels, test_labels = train_test_split(
        prompts,
        outputs,
        annotations,
        label_cats,
        test_size=0.1,
        random_state=42
        )

    train = [{'prompt': train_prompts[i], 'output': train_outputs[i], 'annotations': train_annotations[i], 'labels': train_labels[i]} for i in range(len(train_prompts))]
    test = [{'prompt': test_prompts[i], 'output': test_outputs[i], 'annotations': test_annotations[i], 'labels': test_labels[i]} for i in range(len(test_prompts))]

    return train, test


def preprocess_by_size(size=0):
    ID_DICT = fetch_id_dict()
    if size == 0:
        selected_datasets = list(ID_DICT.keys())[:]
    else:
        selected_datasets = list(ID_DICT.keys())[:size]
    train, test = prepare_all_data(selected_datasets, False)
    return train, test


if __name__ == '__main__':
    ID_DICT = fetch_id_dict()
    selected_datasets = list(ID_DICT.keys())[:2]
    train, test = prepare_all_data(selected_datasets, False)


