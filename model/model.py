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


model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
prompt_test = test[0]


tokenizer = AutoTokenizer.from_pretrained(
        'gpt2',
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN, mask_token=DEFAULT_MASK_TOKEN),
        tokenizer=tokenizer,
        model=model,
)
data_module = make_supervised_data_module(tokenizer=tokenizer, data=train)

# max_grad_norm = 1
trainer = Trainer(model=model, tokenizer=tokenizer, **data_module)
trainer.train()