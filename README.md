# UniPredict

This repo is a demo for supporting the reproducibility of the [UniPredict](https://arxiv.org/abs/2310.03266) framework. The official repository will be released after the review process finishes.

# Example Usage
Python 3.11
## Install Dependencies
```
pip install -r requirements.txt
```
After dependencies installed, you can choose to run either from any checkpoints we provided, or a provided pipeline. For the latter, run

```
./run.sh
```
For the former, follow the steps below:

## Download & Preprocess Datasets
We provide several checkpoints for downloading and preprocessing datasets.

### Start everything from scratch
```
python download_data.py --from-step scratch
```
This is not recommended as it will download > 2000 datasets to your computer. As an alternative, we provided a small subset for you to test on.

### Start from metadata
```
python download_data.py --from-step round_1
```
Before running this line, you have to put a valid openai api key to `.env` file. You are not required to do your own metadata preprocessing because we have included the preprocessed metadata in each provded datasets.

### Start from preprocessing
```
python download_data.py --from-step metadata
```

## Run 
```
python preprocess_data.py --model-type unipred
python train.py --model-type unipred
```

## Test
```
python test_model.py
```

## Plot Graphs
Use `display_data.ipynb`.
