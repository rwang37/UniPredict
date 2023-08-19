import numpy as np

from utils import *

def exec_func_on_each_dataset(function, path):
    data = read_json(path)
    outcome = []
    for item in data:
        name = item[0]
        print(name)
        try:
            temp = function(name)
            outcome.append((name, temp))
        except Exception as e:
            print(e)
            continue
    return outcome

def get_comparison(dataset):
    baseline_dict = read_json(f'files/data/kaggle/{dataset}/baseline_acc.json')
    xgb_acc = baseline_dict['xgboost_accuracy']
    if 'mlp_accuracy' in baseline_dict:
        mlp_acc = baseline_dict['mlp_accuracy']
    else:
        mlp_acc = 0
    model_acc = read_json(f'files/logs/trial2/{dataset}/model_prediction.json')['test accuracy']
    model_acc_2 = read_json(f'files/logs/trial2/{dataset}/model_40000_prediction.json')['test accuracy']
    return [model_acc, xgb_acc, mlp_acc, model_acc_2]

def to_latex(output):
    cells = [f'{output[0][i]} & {output[1][i][0]} & {output[1][i][1]} & {output[1][i][2]}\\\\\n' for i in range(len(output[0]))]
    template = "\\begin\{center\}\n\\begin\{tabular\}{ |c|c|c|c| }\nDataset Name & Model & XGBoost & MLP\n" + ''.join(cells) + "\\hline\n\\end\{tabular\}\n\\end\{center\}"
    return template
    
if __name__ == '__main__':
    path = 'files/data/processed/trial_1/dataset_info.json'
    output = exec_func_on_each_dataset(get_comparison, path)
    accs = np.array([item[1] for item in output])
    accs = np.round(accs, 4)
    output = ([item[0] for item in output], accs)
    rank = np.flip(accs.argsort(), axis=1).argsort() + 1
    avg_acc = np.average(accs, axis=0)
    std_acc = np.std(accs, axis=0)
    avg_rnk = np.average(rank, axis=0)
    print(f'{avg_acc}+-{std_acc}')
    print(avg_rnk)
    print(to_latex(output))
