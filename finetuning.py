import torch
import argparse
import os
import json
from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments

from preprocessing.preprocessing import prepare_data_by_dataset, fetch_id_dict, transform_data
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from model.dataset import *
from utils import *

def make_zeroshot_dataset(zero_shot=100):
    metadata_list = read_json('files/data/kaggle_dataset_record.json')

    dataset_list = []
    zero_shot_test = []
    zero_shot_set_list = []
    test_set = []

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

        if len(samples) < zero_shot:
            resampled_samples = resample(samples, n_samples = int(len(samples) * 5 / 9), replace = False, random_state = 42)
            zero_shot_test.extend(resampled_samples)
            zero_shot_set_list.append(path)

            _, prompt_components, outputs = torch.load(f'files/data/kaggle/{path}/test_set.pt')
            prompts, annotations, labels = prompt_components
            samples_test = [{
                'prompt': prompts[i],
                'annotations': annotations[i],
                'labels': labels[i],
                'output': outputs[i],
            } for i in range(len(prompts))]
            test_set.extend(samples_test)

    save_json('files/data/processed/trial_1/zero_shot_dataset_info.json', zero_shot_set_list)
    save_json('files/data/processed/trial_1/zero_shot_test_set.json', test_set)

    model, tokenizer = setup_model_and_tokenizer('gpt2')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data=zero_shot_test)
    torch.save(data_module, 'files/data/processed/trial_1/fine_tune.pt')
    return data_module