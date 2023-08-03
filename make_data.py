import torch
import argparse
from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments

from preprocessing.preprocessing import prepare_data_by_dataset, fetch_id_dict, transform_data
from sklearn.model_selection import train_test_split
from files.data import preprocess_custom
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


def preprocess_from_custon():
    for item in preprocess_custom.TARGETS.keys():
        try:
            data, col, labels, annotations = preprocess_custom.prepare_data(item)
            xgb_baseline, prompts, output = transform_data(data, col, labels, annotations)

            torch.save(xgb_baseline, f'files/data/processed/xgb_baseline_{item}.pt')
            torch.save(prompts, f'files/data/processed/prompts_{item}.pt')
            torch.save(output, f'files/data/processed/output_{item}.pt')

            # print(xgb_baseline[0][0], xgb_baseline[1][0])
            # print(prompts[0][0], prompts[1][0], prompts[2][0])
            # print(output[0])
            print(f'{item}: files saved!')
        except Exception as e:
            print(e)
            raise e
            print('Error raised while preprocessing dataset. Skipping...')

    print('Finished!')

def make_full_dataset():
    train_set = []
    test_set = []
    for item in preprocess_custom.TARGETS.keys():
        print(f'loading {item} dataset...')
        prompts, annotations, labels = torch.load(f'files/data/processed/prompts_{item}.pt')
        outputs = torch.load(f'files/data/processed/output_{item}.pt')
        #[{'prompt': train_prompts[i], 'output': train_outputs[i], 'annotations': train_annotations[i], 'labels': train_labels[i]} for i in range(len(train_prompts))]
        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]
        train_samples, test_samples = train_test_split(
            samples,
            test_size=0.1,
            random_state=42
        )
        train_set.extend(train_samples)
        test_set.extend(test_samples)
    
    print(len(train_set))
    print(len(test_set))
    torch.save(train_set, 'files/data/processed/untok_train_full.pt')
    torch.save(test_set, 'files/data/processed/untok_test_full.pt')
    model, tokenizer = setup_model_and_tokenizer('gpt2')
    data_module = make_supervised_data_module(tokenizer=tokenizer, data=train_set)
    torch.save(data_module, 'files/data/processed/train_full.pt')
            
if __name__ == '__main__':
    make_full_dataset()