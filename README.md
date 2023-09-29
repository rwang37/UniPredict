## UniPredict

This is an implementation for the paper `UniPredict: Large Language Models are Universal Tabular Predictors`.
UniPredict is a LLM-powered system for universal tabular prediction.

## Example Usage
```
python -m pip install -r requirements.txt
cd model
python dataset.py --size small
python model.py --size small
python test.py --size small --test-size 20 > test.out
```
