import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import delu

from utils import *
from sklearn.utils import resample

def train_and_test_ftt(dataset):
    # reset
    acc_dict = read_json(f'files/data/kaggle/{dataset}/baseline_acc.json')
    if 'ftt_accuracy' in acc_dict.keys():
        acc_dict.pop('ftt_accuracy')
        save_json(f'files/data/kaggle/{dataset}/baseline_acc.json', acc_dict)

    device = 'cuda'
    data_train = torch.load(f'files/data/kaggle/{dataset}/train_set.pt')
    data_test = torch.load(f'files/data/kaggle/{dataset}/test_set.pt')

    X = {}
    y = {}
    X['train'], y['train'] = data_train[0]

    X['train'], y['train'] = resample(X['train'], y['train'], n_samples = int(len(y['train']) * 5 / 9), replace = False, random_state = 42)

    X['test'], y['test'] = data_test[0]
    X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
        X['train'], y['train'], train_size=0.8
    )

    # not the best way to preprocess features, but enough for the demonstration
    preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
    X = {
        k: torch.tensor(preprocess.transform(v), device=device).float()
        for k, v in X.items()
    }
    y = {k: torch.tensor(v).type(torch.LongTensor).to(device) for k, v in y.items()}

    model = rtdl.FTTransformer.make_default(
        n_num_features=X['train'].shape[1],
        cat_cardinalities=None,
        last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
        d_out= int(max(y['train'])) + 1,
    )

    model.to(device)
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    loss_fn = F.cross_entropy

    batch_size = 256
    train_loader = delu.data.IndexLoader(len(X['train']), batch_size, device=device)
    progress = delu.ProgressTracker(patience=100)

    def apply_model(x_num, x_cat=None):
        if isinstance(model, rtdl.FTTransformer):
            return model(x_num, x_cat)
        elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
            assert x_cat is None
            return model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(model)}.'
                ' Then you have to implement this branch first.'
            )

    @torch.no_grad()
    def evaluate(part):
        model.eval()
        prediction = []
        for batch in delu.iter_batches(X[part], 1024):
            prediction.append(apply_model(batch))
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        target = y[part].cpu().numpy()

        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)

        return score
        
    print(f'Test score before training: {evaluate("test"):.4f}')

    n_epochs = 100
    report_frequency = len(X['train']) // batch_size // 5
    for epoch in range(1, n_epochs + 1):
        for iteration, batch_idx in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x_batch = X['train'][batch_idx]
            y_batch = y['train'][batch_idx]
            loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch.squeeze(1))
            loss.backward()
            optimizer.step()
            # if iteration % report_frequency == 0:
            #     print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

        val_score = evaluate('val')
        test_score = evaluate('test')
        print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
        progress.update(val_score)
        if progress.success:
            print(' <<< BEST VALIDATION EPOCH', end='')
        print()
        if progress.fail:
            break
    test_score = evaluate('test')
    acc_dict = read_json(f'files/data/kaggle/{dataset}/baseline_acc.json')
    acc_dict['ftt_accuracy'] = test_score
    save_json(f'files/data/kaggle/{dataset}/baseline_acc.json', acc_dict)
    torch.save(model, f'files/data/kaggle/{dataset}/ftt_model.pt')


if __name__ == '__main__':
    from display_test import exec_func_on_each_dataset
    # path = 'files/data/processed/trial_1/dataset_info.json'
    path = 'files/data/processed/trial_1/zero_shot_dataset_info.json'
    exec_func_on_each_dataset(train_and_test_ftt, path)
