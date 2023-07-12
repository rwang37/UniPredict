import torch
import pandas as pd
import json
import copy
import transformers

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.utils.data import Dataset
from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments
from sklearn.model_selection import train_test_split
from dataset import setup_model_and_tokenizer, SupervisedDataset, DataCollatorForSupervisedDataset

model, tokenizer = setup_model_and_tokenizer('gpt2')
data_module = torch.load('train_prompt_large.pt')
model = model.cuda()
print('data loaded!')

args = TrainingArguments("checkpoints")
args = args.set_save(strategy="epoch", total_limit=10)

# max_grad_norm = 1
trainer = Trainer(model=model, tokenizer=tokenizer, args=args, **data_module)
trainer.train()
print('training finished!')

torch.save(model, 'gpt2_large.pt')
print('model saved!')