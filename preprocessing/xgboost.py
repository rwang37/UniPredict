import xgboost as xgb
import pandas as pd
import sklearn
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


def numericalize(samples, labels, column_names):
    for i in range(len(column_names)):
        col = column_names[i]
        categories = [cat for cat in set(samples[col].to_list())]
        cat_dict = {categories[i]: i for i in range(len(categories))}
        samples[col] = samples[col].map(cat_dict).astype(int)
    categories = [cat for cat in set(labels.to_list())]
    cat_dict = {categories[i]: i for i in range(len(categories))}
    labels = labels.map(cat_dict).astype(int)
    return samples, labels


def get_XGB_classification(samples, labels, column_names):
    samples, labels = numericalize(samples, labels, column_names)
    samples = samples.to_numpy()
    labels = labels.to_numpy()

    # print(samples.shape, labels.shape)
    # le = LabelEncoder()
    # labels = le.fit_transform(labels)

    calibrated_clf = sklearn.calibration.CalibratedClassifierCV(xgb.XGBClassifier(n_estimators = 400), method='isotonic')
    calibrated_clf.fit(samples, labels)
    preds = calibrated_clf.predict_proba(samples)

    auc = calculate_auc(labels, preds)
    outputs = serialize_output(preds)
    return outputs, auc

def calculate_auc(labels, preds):
    # ground truth
    y_gt = labels
    onehot = np.zeros((y_gt.size, y_gt.max() + 1))
    onehot[np.arange(y_gt.size), y_gt] = 1
    y_gt = onehot
    # pred
    y_pred = preds
    return roc_auc_score(y_gt, y_pred, average=None)

def serialize_output(preds):
    outputs = []
    for i in preds:
        out_strs = ''
        for j, k in enumerate(i):
            out_str = f'Class {j}: {np.round_(k, 2)}; '
            out_strs += out_str
        out_strs = out_strs[:-2] + '.'
        outputs.append(out_strs)
    return outputs