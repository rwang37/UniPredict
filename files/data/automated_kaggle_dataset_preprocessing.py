import kaggle
import os
import json

curr_path = os.path.dirname(os.path.realpath(__name__))

dataset_list = []
for i in range(1, 1000):
    temp_list = kaggle.api.dataset_list(
        file_type='csv', 
        tag_ids=13302, 
        max_size=1048576, 
        page=i
    )
    if len(temp_list) == 0:
        break
    dataset_list.extend(temp_list)


for item in dataset_list:
    download_path = f'kaggle/{item}'
    kaggle.api.dataset_metadata(
        'yapwh1208/students-score', 
        path=kaggle.api.get_default_download_dir()
    )

    kaggle.api.dataset_download_files(
        'yapwh1208/students-score', 
        path=kaggle.api.get_default_download_dir(), 
        unzip=True
    )