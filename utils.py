import json

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(path, data):
    with open(path, 'w+') as f:
        json.dump(data, f, indent=4)