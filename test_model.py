from model.test import *
from model.dataset import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=str,  nargs='?', const='full', required=True)
parser.add_argument('--test-size', type=int, nargs='?', const=20, required=True)
args = parser.parse_args()

prompt = torch.load(f'files/data/processed/untok_test_{args.size}.pt')
reference_prompt = torch.load(f'files/data/processed/untok_train_{args.size}.pt')
model = torch.load(f'files/testing/trained_gpt2_{args.size}.pt').cuda()
_, tokenizer = setup_model_and_tokenizer('gpt2')

test_size = args.test_size if args.test_size else len(prompt)
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