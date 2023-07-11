from preprocessing import prepare_all_data, preprocess_by_size
from openml_id import fetch_id_dict


if __name__ == '__main__':
    train, test = preprocess_by_size(10)
    print(train[20])
    print(len(train))