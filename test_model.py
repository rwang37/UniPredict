from model.test import *
from model.dataset import *
import argparse


def test_model_on_dataset(model, tokenizer, dataset):
    prompt = torch.load(f'files/data/kaggle/{dataset}/test_set.pt')
    reference_prompt = torch.load(f'files/data/kaggle/{dataset}/train_set.pt')
    model = model.cuda()

    test_size = len(prompt)
    accs = []
    count = 0

    for i in tqdm(range(test_size)):
        test = prompt[i]
        if test in reference_prompt:
            continue
        output = test['output']
        test = PROMPT_DICT["prompt_input"].format_map(test)

        pred = test_input(test, model, tokenizer).split('\n')[-1]
        print(test)
        print(f'---Pred: {pred}\n---Reference: {output}')
        corr = check_correctness(pred, output)
        print(corr)
        print('\n\n')
        accs.append(corr)
        count += 1
    print(sum(accs) / count, count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-from', type=str,  nargs='?', const='pretrained', required=True)
    parser.add_argument('--checkpoint', type=int, nargs='?', const=130000, required=False)
    args = parser.parse_args()

    if args.model_from = 'pretrained':
        model = GPT2LMHeadModel.from_pretrained(f'files/model_checkpoints/checkpoint-{args.checkpoint}')
    else:
        model = torch.load('files/processed/trial1/model.pt')
    _, tokenizer = setup_model_and_tokenizer('gpt2')

    test_model_on_dataset(model, tokenizer, 'dataset_name')