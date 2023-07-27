import pandas as pd
import numpy as np
import json
import os

DATASET_PATH = os.path.join(
    os.path.abspath('files'),
    'data',
    )

TARGETS = {
    'california': 'median_house_value',
    'crab_age': 'Age',
    'dubai': 'price',
    'gem_price': 'price',
    'heloc': 'RiskPerformance',
    'medical_cost': 'charges'
}

LABEL_CUT = {
    'california': {
        'bins': [0, 100000, 200000, 300000, 400000, 500000, np.inf], 
        'labels': ['<100k', '100k-200k', '200k-300k', '300k-400k', '400k-500k', '>500k']
    },
    'dubai': {
        'bins': [0, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, np.inf],
        'labels': ['<1M', '1M-1.5M', '1.5M-2M', '2M-2.5M', '2.5M-3M', '3M-3.5M', '>3.5M'] 
    },
    'gem_price': {
        'bins': [0, 1000, 2000, 3000, 4000, 5000, 6000, np.inf],
        'labels': ['<1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '>6000']
    },
    'medical_cost': {
        'bins': [0, 5000, 10000, 15000, 20000, np.inf],
        'labels': ['<5000', '5000-10000', '10000-15000', '15000-20000', '>20000']
    }
}

def prepare_data(name):
    # samples, column_names, labels, annotations
    with open(os.path.join(DATASET_PATH, 'custom', 'annotations.json')) as f:
        annotation_dict = json.load(f)

    dataset = pd.read_csv(os.path.join(DATASET_PATH, 'custom', f'{name}.csv'))
    target_name = TARGETS[name]
    
    samples = dataset.drop(target_name, axis=1).round(4)
    
    col = samples.columns.to_list()

    if name in LABEL_CUT.keys():
        labels = pd.cut(dataset[target_name], bins=LABEL_CUT[name]['bins'], labels=LABEL_CUT[name]['labels'])
    else:
        labels = dataset[target_name]
    
    annotations = [annotation_dict[name]] * len(samples)
    return samples, col, labels, annotations

if __name__ == '__main__':
    for i in TARGETS.keys():
        print(prepare_data(i))