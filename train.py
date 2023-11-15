import transformers
import torch
import numpy as np
import os

from preprocessing.dataset import *

def train_model(dataset='files/unified/prompts/toked_train_set.pt', save_model='files/unified/models/unipred.pt', save_state='files/unified/models/unipred_state.pt'):
    model, tokenizer = setup_model_and_tokenizer('gpt2')
    data_module = torch.load(dataset)
    model = model.cuda()
    print('data loaded!')

    training_args = TrainingArguments(
        "files/checkpoints",
        per_device_train_batch_size=4,
        )
    training_args = training_args.set_save(strategy="steps", steps=10000, total_limit=3)

    # max_grad_norm = 1
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print(trainer.args._n_gpu)
    print(trainer.args.parallel_mode)
    trainer.train()
    print('training finished!')

    torch.save(model, save_model)
    torch.save(model.state_dict(), save_state)
    print('model saved!')
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', 
                        type=str,
                        required=True,
                        help='The input model type. Must be a value among "unipred", "light", and "ablation".')

    args = parser.parse_args()
    model_type = args.model_type
    if model_type == 'unipred':
        train_model(dataset='files/unified/prompts/toked_train_set.pt', save_model='files/unified/models/unipred.pt', save_state='files/unified/models/unipred_state.pt')
    elif model_type == 'lignt':
        train_model(dataset='files/unified/prompts/toked_light_train_set.pt', save_model='files/unified/models/light.pt', save_state='files/unified/models/light_state.pt')
    elif model_type == 'ablation':
        train_model(dataset='files/unified/prompts/toked_abl_aug_train_set.pt', save_model='files/unified/models/abl_aug.pt', save_state='files/unified/models/abl_aug_state.pt')