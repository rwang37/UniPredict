import transformers
import torch
import numpy as np

from model.dataset import *

def train_model(dataset='files/data/processed/trial_1/train_kaggle.pt'):
    model, tokenizer = setup_model_and_tokenizer('gpt2')
    data_module = torch.load(dataset)
    model = model.cuda()
    print('data loaded!')

    training_args = TrainingArguments("files/model_checkpoints")
    training_args = training_args.set_save(strategy="steps", steps=10000, total_limit=10)

    # max_grad_norm = 1
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    print('training finished!')

    torch.save(model, 'files/data/processed/trial_1/model.pt')
    print('model saved!')

if __name__ == '__main__':
    train_model()