import argparse
import json
import os
import xgboost as xgb

from model.test import *
from sklearn.neural_network import MLPClassifier
from model.dataset import *
from sklearn.utils import resample
from utils import *
from pytorch_tabnet.tab_model import TabNetClassifier

def train_xgb_on_dataset(dataset):
    data_train = torch.load(f'files/data/kaggle/{dataset}/train_set.pt')
    x_train, y_train = data_train[0]
    x_train, y_train = resample(x_train, y_train, n_samples = int(len(y_train) * 5 / 9), replace = False, random_state = 42)
    clf = xgb.XGBClassifier(n_estimators=100)
    clf.fit(x_train, y_train.squeeze(1))
    torch.save(clf, f'files/data/kaggle/{dataset}/xgb_model.pt')


def test_xgb_on_dataset(dataset):
    data_test = torch.load(f'files/data/kaggle/{dataset}/test_set.pt')
    x_test, y_test = data_test[0]
    clf = torch.load(f'files/data/kaggle/{dataset}/xgb_model.pt')
    pred = clf.predict(x_test)
    correctness = y_test.squeeze(-1) == pred
    acc = sum(correctness) / len(correctness)
    try:
        acc_dict = read_json(f'files/data/kaggle/{dataset}/baseline_acc.json')
    except:
        acc_dict = {}
    acc_dict['xgboost_accuracy'] = acc
    save_json(f'files/data/kaggle/{dataset}/baseline_acc.json', acc_dict)


def train_mlp_on_dataset(dataset):
    data_train = torch.load(f'files/data/kaggle/{dataset}/train_set.pt')
    x_train, y_train = data_train[0]
    x_train, y_train = resample(x_train, y_train, n_samples = int(len(y_train) * 5 / 9), replace = False, random_state = 42)
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(x_train, y_train.squeeze(1))
    torch.save(clf, f'files/data/kaggle/{dataset}/mlp_model.pt')


def test_mlp_on_dataset(dataset):
    data_test = torch.load(f'files/data/kaggle/{dataset}/test_set.pt')
    x_test, y_test = data_test[0]
    clf = torch.load(f'files/data/kaggle/{dataset}/mlp_model.pt')
    pred = clf.predict(x_test)
    correctness = y_test.squeeze(-1) == pred
    acc = sum(correctness) / len(correctness)
    try:
        acc_dict = read_json(f'files/data/kaggle/{dataset}/baseline_acc.json')
    except:
        acc_dict = {}
    acc_dict['mlp_accuracy'] = acc
    save_json(f'files/data/kaggle/{dataset}/baseline_acc.json', acc_dict)


def train_tbn_on_dataset(dataset):
    data_train = torch.load(f'files/data/kaggle/{dataset}/train_set.pt')
    x_train, y_train = data_train[0]
    x_train, y_train = resample(x_train, y_train, n_samples = int(len(y_train) * 5 / 9), replace = False, random_state = 42)
    clf = TabNetClassifier()
    clf.fit(x_train, y_train.squeeze(1))
    torch.save(clf, f'files/data/kaggle/{dataset}/tbn_model.pt')


def test_tbn_on_dataset(dataset):
    data_test = torch.load(f'files/data/kaggle/{dataset}/test_set.pt')
    x_test, y_test = data_test[0]
    clf = torch.load(f'files/data/kaggle/{dataset}/tbn_model.pt')
    pred = clf.predict(x_test)
    correctness = y_test.squeeze(-1) == pred
    acc = sum(correctness) / len(correctness)
    try:
        acc_dict = read_json(f'files/data/kaggle/{dataset}/baseline_acc.json')
    except:
        acc_dict = {}
    acc_dict['tbn_accuracy'] = acc
    save_json(f'files/data/kaggle/{dataset}/baseline_acc.json', acc_dict)


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
    # os.mkdir(f'files/logs/trial2/{dataset}')
    save_json(f'files/logs/trial2/{dataset}/model_prediction_before_finetuning.json', dataset_outcome)



def test_model_on_zeroshot(model, tokenizer, dataset):
    test_samples = read_json(dataset)
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
    save_json(f'files/logs/trial2/fine_tuned_prediction.json', dataset_outcome)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-from', type=str,  nargs='?', const='pretrained', required=True)
    parser.add_argument('--checkpoint', type=str, nargs='?', const=130000, required=False)
    args = parser.parse_args()

    dataset_info = read_json('files/data/processed/trial_1/zero_shot_dataset_info.json')

    if args.model_from == 'pretrained':
        model = GPT2LMHeadModel.from_pretrained(f'files/model_checkpoints/checkpoint-{args.checkpoint}')
    else:
        model = torch.load(f'files/data/processed/{args.checkpoint}/model.pt')
    _, tokenizer = setup_model_and_tokenizer('gpt2')
    model.eval()
    # test_model_on_zeroshot(model, tokenizer, 'files/data/processed/trial_1/zero_shot_test_set.json')

    for item in dataset_info:
        # name = item[0]
        name = item
        print(name)
        acc_dict = read_json(f'files/data/kaggle/{name}/baseline_acc.json')
        if 'ftt_accuracy' in acc_dict.keys():
            save_json(f'files/data/kaggle/{name}/baseline_acc.json', {'ftt_accuracy': acc_dict['ftt_accuracy']})
        else:
            save_json(f'files/data/kaggle/{name}/baseline_acc.json', {})
        try:
            train_mlp_on_dataset(name)
            test_mlp_on_dataset(name)
        except Exception as e:
            print(e)

        try:
            train_xgb_on_dataset(name)
            test_xgb_on_dataset(name)
        except Exception as e:
            print(e)

        try:
            train_tbn_on_dataset(name)
            test_tbn_on_dataset(name)
        except Exception as e:
            print(e)
        # test_model_on_dataset(model, tokenizer, name)
        
