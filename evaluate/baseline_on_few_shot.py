import json
import os
import xgboost as xgb
import torch

from copy import deepcopy
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from .utils import *

MODELDICT = {
    'xgb': xgb.XGBClassifier(n_estimators=100),
    'mlp': MLPClassifier(random_state=1, max_iter=300),
    'tbn': TabNetClassifier()
}
class FewShotBaselineTester():
    def __init__(
        self, 
        name, 
        model_type,
        path=DEFAULT_DATASET_SAVING_PATH, 
        debug=False
    ):
        self.model = deepcopy(MODELDICT[model_type])
        self.debug = debug
        self.train_loc = path + name.replace('/', '-') + '/train_set.pt'
        self.test_loc = path + name.replace('/', '-') + '/test_set.pt'
        self.train = torch.load(self.train_loc)
        self.test = torch.load(self.test_loc)
    
    def reshuffle_data(self, ratio):
        x_train, y_train = self.train[0]
        x_test, y_test = self.test[0]
        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=ratio,
            random_state=42
        )
        self.train = x_train, y_train
        self.test = x_test, y_test
    
    def train_model(self):
        x_train, y_train = self.train
        y_train = y_train
        self.model.fit(x_train, y_train.squeeze(-1))
    
    def test_model(self):
        x_test, y_test = self.test
        pred = self.model.predict(x_test)
        correctness = y_test == pred
        self.acc = sum(correctness) / len(correctness)

    def get_acc(self):
        return self.acc
    
class FewShotBaselineTesterGroup():
    def __init__(
        self, 
        model_type,
        dataset_list='few_shot_datasets.json',
        dataset_list_path = DEFAULT_DATASET_INDEXING_PATH,
        dataset_path=DEFAULT_DATASET_SAVING_PATH, 
        debug=False
    ):
        self.dataset_list = read_json(dataset_list_path + dataset_list)
        self.model_type = model_type
        self.debug = debug
        self.dataset_path = dataset_path
        self.acc_dict = {}
    
    def get_accuracy(self):
        for i in range(1, 10):
            ratio = i/10
            for item in self.dataset_list:
                print(item)
                try:
                    tester = FewShotBaselineTester(
                        item, 
                        self.model_type, 
                        path=self.dataset_path, 
                        debug=self.debug
                    )
                    tester.reshuffle_data(ratio=ratio)
                    tester.train_model()
                    tester.test_model()
                    acc = float(tester.get_acc())
                    if item in self.acc_dict.keys():
                        self.acc_dict[item][self.model_type] = acc
                    else:
                        self.acc_dict[item] = {self.model_type: acc}
                except Exception as e:
                    print(e)
                    continue
            
    def load_acc_dict(self, path='files/unified/results/few_shot2.json'):
        self.acc_dict = read_json(path)
    
    def save_acc_dict(self, path='files/unified/results/few_shot2.json'):
        save_json(path, self.acc_dict)