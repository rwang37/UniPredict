import pandas as pd
import numpy as np
import json
import os

# DATASET_PATH = os.path.join(
#     os.path.abspath('files'),
#     'data',
#     )

DATASET_PATH = os.path.join(
    os.path.abspath('')
    )

with open(os.path.join(DATASET_PATH, 'targets.json')) as f:
    TARGETS = json.load(f)

with open(os.path.join(DATASET_PATH, 'segmentations.json')) as f:
    LABEL_CUT = json.load(f)  

with open(os.path.join(DATASET_PATH, 'annotations.json')) as f:
    ANNOTATIONS = json.load(f)


def prepare_data(name):
    # samples, column_names, labels, annotations
    # with open(os.path.join(DATASET_PATH, 'custom', 'annotations.json')) as f:
    #     ANNOTATIONS = json.load(f)

    dataset = pd.read_csv(os.path.join(DATASET_PATH, 'custom', f'{name}.csv'))
    target_name = TARGETS[name]
    
    samples = dataset.drop(target_name, axis=1).round(4)
    
    col = samples.columns.to_list()

    if name in LABEL_CUT.keys():
        segs = [0] + LABEL_CUT[name]['bins'] + [np.inf]
        labels = pd.cut(dataset[target_name], bins=segs, labels=LABEL_CUT[name]['labels'])
    else:
        labels = dataset[target_name]
    
    annotations = [ANNOTATIONS[name]] * len(samples)
    return samples, col, labels, annotations

if __name__ == '__main__':
    for i in TARGETS.keys():
        print(prepare_data(i)[1])