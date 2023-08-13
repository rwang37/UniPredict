import argparse
import json
import os
import xgboost as xgb

from model.test import *
from model.dataset import *

def train_xgb_on_dataset(dataset):
    data_train = torch.load(f'files/data/kaggle/{dataset}/train_set.pt')
    x_train, y_train = data_train[0]
    clf = xgb.XGBClassifier(n_estimators=50)
    clf.fit(x_train, y_train)
    torch.save(clf, f'files/data/kaggle/{dataset}/xgb_model.pt')


def test_xgb_on_dataset(dataset):
    data_test = torch.load(f'files/data/kaggle/{dataset}/test_set.pt')
    x_test, y_test = data_test[0]
    clf = torch.load(f'files/data/kaggle/{dataset}/xgb_model.pt')
    pred = clf.predict(x_test)
    correctness = y_test.squeeze(-1) == pred
    acc = sum(correctness) / len(correctness)
    with open(f'files/data/kaggle/{dataset}/baseline_acc.json', 'w+') as f:
        json.dump({'xgboost_accuracy': acc}, f, indent = 4)


def test_model_on_dataset(model, tokenizer, dataset):
    _, prompt_components, outputs = torch.load(f'files/data/kaggle/{dataset}/test_set.pt')
    prompts, annotations, labels = prompt_components
    test_samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]
    test_samples = [
        {
            'sample': PROMPT_DICT["prompt_input"].format_map(item),
            'output': item['output']
        } 
        for item in test_samples]
    model = model.cuda()

    test_size = len(test_samples)
    accs = []
    count = 0
    results = []
    for i in tqdm(range(test_size)):
        item_outcome = {}
        test = test_samples[i]['sample']
        output = test_samples[i]['output']

        pred = test_input(test, model, tokenizer).split('\n')[-1]
        item_outcome['test input'] = test
        item_outcome['model prediction'] = pred
        item_outcome['reference'] = output
        corr = check_correctness(pred, output)
        item_outcome['model correctness'] = corr
        accs.append(corr)
        count += 1
        results.append(item_outcome)
    acc = sum(accs) / count
    dataset_outcome = {
        'test size': count,
        'test accuracy': acc,
        'items': results
    }
    os.mkdir(f'files/logs/trial1/{dataset}')
    with open(f'files/logs/trial1/{dataset}/model_prediction.json', 'w+') as f:
        json.dump(dataset_outcome, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-from', type=str,  nargs='?', const='pretrained', required=True)
    parser.add_argument('--checkpoint', type=int, nargs='?', const=130000, required=False)
    args = parser.parse_args()

    with open('files/data/processed/trial_1/dataset_info.json', 'r') as f:
        dataset_info = json.load(f)

    if args.model_from == 'pretrained':
        model = GPT2LMHeadModel.from_pretrained(f'files/model_checkpoints/checkpoint-{args.checkpoint}')
    else:
        model = torch.load('files/data/processed/trial_1/model.pt')
    _, tokenizer = setup_model_and_tokenizer('gpt2')

    for item in dataset_info:
        name = item[0]
        print(name)
        try:
            # test_model_on_dataset(model, tokenizer, name)\
            test_xgb_on_dataset(name)
        except Exception as e:
            print(e)
            continue