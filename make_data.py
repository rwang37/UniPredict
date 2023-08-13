import torch
import argparse
import os
import json
from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments

from preprocessing.preprocessing import prepare_data_by_dataset, fetch_id_dict, transform_data
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from model.dataset import *

# parser = argparse.ArgumentParser()
# parser.add_argument('--start', type=int,  nargs='?', const='0', required=False)
# parser.add_argument('--size', type=int,  nargs='?', const='10', required=True)
# args = parser.parse_args()

TEMPLATE = (
        "Below is the description of a dataset, an object profile from the dataset and a target description. "
        "Predict the target by the given information of the object.\n"
        "# Dataset description: {annotations}\n"
        "# Object description: {prompt}\n"
        "# You should return the probability of each class by: \n{labels}\n"
        "# Answer: \n"
    )

def make_full_dataset():
    with open('files/data/kaggle_dataset_record.json', 'r') as f:
        metadata_list = json.load(f)
    dataset_list = []
    train_set = []
    zero_shot_test = []
    # test_set = []

    for item in metadata_list:
        author, path = item.split('/')
        path = f'{author}-{path}'
        print(f'Loading {path}...')
        try:
            _, prompt_components, outputs = torch.load(f'files/data/kaggle/{path}/train_set.pt')
            prompts, annotations, labels = prompt_components
        except Exception as e:
            print(f'Failed due to {e}. Skipping...')
            continue
        #[{'prompt': train_prompts[i], 'output': train_outputs[i], 'annotations': train_annotations[i], 'labels': train_labels[i]} for i in range(len(train_prompts))]
        
        if len(prompts[0]) + len(annotations[0]) + len(labels[0]) > 4000:
            print(f'Dataset {item} too long: {len(prompts[0]) + len(annotations[0]) + len(labels[0])} chars total.')
            continue

        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        if len(samples) > 7500:
            samples = resample(samples, n_samples=7500, replace=False, random_state=42)
        elif len(samples) < 100:
            zero_shot_test.extend(samples)
            continue
        elif len(samples) > 500:
            train_set.extend(samples)
            dataset_list.append((path, len(samples)))
            print(len(train_set))

    # os.mkdir('files/data/processed/trial_1')
    with open('files/data/processed/trial_1/dataset_info.json', 'w+') as f:
        json.dump(dataset_list, f, indent=4)
    with open('files/data/processed/trial_1/untok_train_kaggle.json', 'w+') as f:
        json.dump(train_set, f, indent=4)
    with open('files/data/processed/trial_1/zero_shot_test.json', 'w+') as f:
        json.dump(zero_shot_test, f, indent=4)
    model, tokenizer = setup_model_and_tokenizer('gpt2')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data=train_set)
    torch.save(data_module, 'files/data/processed/trial_1/train_kaggle.pt')
            
if __name__ == '__main__':
    make_full_dataset()