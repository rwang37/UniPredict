import xgboost as xgb
import pandas as pd
import sklearn
import torch
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype, is_integer_dtype 
from preprocessing.xgb import *
from files.data import preprocess_custom

def get_XGB_classification(samples, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        samples, 
        labels, 
        test_size=0.1,
        random_state=42
    )
    clf = xgb.XGBClassifier(n_estimators = 100)
    clf.fit(x_train, y_train)
    calibrated_clf = sklearn.calibration.CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
    calibrated_clf.fit(x_train, y_train)
    preds = calibrated_clf.predict_proba(x_test)

    auc = calculate_auc(y_test, preds)
    print(auc)
    return auc

if __name__ == '__main__':
    for item in preprocess_custom.TARGETS.keys():
        print(item)
        samples, labels = torch.load(f'files/data/processed/xgb_baseline_{item}.pt')
        try:
            get_XGB_classification(samples, labels)
        except Exception as e:
            print(e)
        print()
    