import json
import os
import xgboost as xgb

from copy import deepcopy
from sklearn.neural_network import MLPClassifier
from .utils import *
from pytorch_tabnet.tab_model import TabNetClassifier

MODELDICT = {
    'xgb': xgb.XGBClassifier(n_estimators=100),
    'mlp': MLPClassifier(random_state=1, max_iter=300),
    'tbn': TabNetClassifier()
}
class BaselineTester():
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
    
    def train_model(self):
        x_train, y_train = self.train[0]
        y_train = y_train.squeeze(1)
        self.model.fit(x_train, y_train.squeeze(1))
    
    def test_model(self):
        x_test, y_test = self.test[0]
        pred = self.model.predict(x_test)
        correctness = y_test.squeeze(-1) == pred
        self.acc = sum(correctness) / len(correctness)

    def get_acc(self):
        return self.acc
    
class BaselineTesterGroup():
    def __init__(
        self, 
        model_type,
        dataset_list='supervised_datasets.json',
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
        for item in self.dataset_list:
            print(item)
            try:
                tester = BaselineTester(
                    item, 
                    self.model_type, 
                    path=self.dataset_path, 
                    debug=self.debug
                )
                tester.train_model()
                tester.test_model()
                acc = tester.get_acc()
                if item in self.acc_dict.keys():
                    self.acc_dict[item][self.model_type] = acc
                else:
                    self.acc_dict[item] = {self.model_type: acc}
            except Exception as e:
                print(e)
                continue

    def load_acc_dict(self, path='files/unified/results/supervised2.json'):
        self.acc_dict = read_json(path)
    
    def save_acc_dict(self, path='files/unified/results/supervised2.json'):
        save_json(path, self.acc_dict)