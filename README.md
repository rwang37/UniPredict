# AnyPredict2.0
AnyPredict2.0 is a LLM-powered system for tabular prediction.

## Example Usage
```
python -m pip install -r requirements.txt
cd model
python dataset.py --size small
python model.py --size small
python test.py --size small --test-size 20 > test.out
```
