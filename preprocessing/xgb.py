import xgboost as xgb
import pandas as pd
import sklearn
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_numeric_dtype, is_integer_dtype 


def numericalize(samples, labels, column_names):
    samples = samples.reset_index(drop=True)
    for i in range(len(column_names)):
        col = column_names[i]
        if is_numeric_dtype(samples[col]) or is_integer_dtype(samples[col]):
            continue
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
    clf = xgb.XGBClassifier(n_estimators = 100)
    clf.fit(samples, labels)

    calibrated_clf = sklearn.calibration.CalibratedClassifierCV(estimator=clf, method='isotonic', cv='prefit')
    calibrated_clf.fit(samples, labels)
    preds = calibrated_clf.predict_proba(samples)

    auc = calculate_auc(labels, preds)
    outputs = serialize_output(preds)
    # print(auc)
    return outputs, auc

def calculate_auc(labels, preds):
    # ground truth
    if len(labels.shape) > 1:
        y_gt = labels.squeeze(1)
    else: 
        y_gt = labels
    onehot = np.zeros((y_gt.size, y_gt.max().astype(int) + 1))
    onehot[np.arange(y_gt.size), y_gt] = 1
    y_gt = onehot
    # pred
    y_pred = preds
    return roc_auc_score(y_gt, y_pred, average=None)

def serialize_output(preds):
    outputs = []
    for i in preds:
        out_strs = []
        for j, k in enumerate(i):
            out_str = f'class {j}: {np.round(k, 2)}; '
            out_strs.append(out_str)
        out_strs = ''.join(out_strs)
        out_strs = out_strs[:-2] + '.'
        outputs.append(out_strs)
    return outputs