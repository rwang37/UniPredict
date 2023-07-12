import torch
import copy
import transformers

from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments
from dataset import setup_model_and_tokenizer, SupervisedDataset, DataCollatorForSupervisedDataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is the description of a dataset, an object profile from the dataset and a target description. "
        "Predict the target by the given information of the object.\n"
        "# Dataset description: {annotations}\n"
        "# Object description: {prompt}\n"
        "# Predict the target class by responding: {labels}\n"
        "# Solution: "
    )
}

def test_input(prompt, model):
    _, tokenizer = setup_model_and_tokenizer('gpt2')
    # print('\n======== Testing ========')
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    # print(inputs, inputs['input_ids'].squeeze(0).shape, tokenizer.decode(inputs['input_ids'].squeeze(0)))
    outputs = model.generate(
        **inputs,
        do_sample=True,
        max_length=1024,
        top_k=50,
        top_p=0.95,
        num_return_sequences=3
    )
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == '__main__':
    prompt = torch.load('test_prompt_large_untok.pt')
    model = torch.load('gpt2_large.pt')

    for i in range(100):
        test = prompt[i]
        output = test['output']
        test = PROMPT_DICT["prompt_input"].format_map(test)

        pred = test_input(test, model).split('\n')[-1]
        print(f'---Pred: {pred}\n---Reference: {output}\n\n')
