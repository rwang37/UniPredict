import transformers
import torch
import numpy as np

from files.data import preprocess_custom
from model.dataset import *

def train_model(size='full'):
    model, tokenizer = setup_model_and_tokenizer('gpt2')
    data_module = torch.load(f'files/data/processed/train_{size}.pt')
    model = model.cuda()
    print('data loaded!')

    training_args = TrainingArguments("files/model_checkpoints")
    training_args = training_args.set_save(strategy="epoch", total_limit=10)

    # max_grad_norm = 1
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    print('training finished!')

    torch.save(model, f'files/testing/trained_gpt2_{size}.pt')
    print('model saved!')

if __name__ == '__main__':
    train_model()