#!/bin/bash

python download_data.py --from-step metadata
python preprocess_data.py --model-type unipred
python train.py --model-type unipred
python test_model.py