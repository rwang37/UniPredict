import torch
import copy
import transformers
import re

from transformers import Trainer, GPT2LMHeadModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments
from dataset import setup_model_and_tokenizer, SupervisedDataset, DataCollatorForSupervisedDataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is the description of a dataset, an object profile from the dataset and a target description. "
        "Predict the target by the given information of the object.\n"
        "# Dataset description: {annotations}\n"
        "# Object description: {prompt}\n"
        "# You should return the probability of each class by: {labels}\n"
        "# Answer: \n"
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


def response_to_class(response):
    return re.findall(r'[\d]*[.][\d]+', response)

def check_correctness(pred, ref):
    try:
        pred_cls = response_to_class(pred)
        ref_cls = response_to_class(ref)
        pred_idx = pred_cls.index(max(pred_cls))
        ref_idx = ref_cls.index(max(ref_cls))
        if pred_idx == ref_idx:
            return True
    except:
        return False
    return False



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str,  nargs='?', const='small', required=True)
    parser.add_argument('--test-size', type=int, nargs='?', const=20, required=True)
    args = parser.parse_args()

    prompt = torch.load(f'test_prompt_{args.size}_untok.pt')
    model = torch.load(f'gpt2_{args.size}.pt')

    test_size = args.test_size if args.test_size else len(prompt)
    accs = []

    for i in range(test_size):
        test = prompt[i]
        output = test['output']
        test = PROMPT_DICT["prompt_input"].format_map(test)

        pred = test_input(test, model).split('\n')[-1]
        # print(test)
        print(f'---Pred: {pred}\n---Reference: {output}\n\n')
        corr = check_correctness(pred, output)
        print(corr)
        accs.append(corr)
    print(sum(accs) / test_size)