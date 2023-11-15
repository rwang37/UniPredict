\* This repo is a demo for addressing reproducibility concerns to the `UniPredict` framework. We are currently optimizing the codes for better readability and usability. The final release will be posted on Github after the reviewing stage finishes. 

## UniPredict

This is the official implementation for the paper `UniPredict: Large Language Models are Universal Tabular Predictors`.
UniPredict is a LLM-powered system for universal tabular prediction.

## Example Usage
Python 3.11
### Install Dependencies
```
pip install -r requirements.txt
```
### Download Datasets
You can download the pre-processed datasets we have used [here](https://drive.google.com/file/d/1jnqWAPGyaAWoxV0bSlNnt-I_SH6-X7Vc/view?usp=sharing). 

### Run 
```
python preprocess_data.py --model-type unipred
python preprocess_data.py --model-type light
python preprocess_data.py --model-type ablation
python train.py
```

### Test
```
python test_model.py
```

### Plot Graphs
Use `display_data.ipynb`.