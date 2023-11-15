import torch
import numpy as np

from .utils import *

class SupervisedModelTester():
    def __init__(
        self, 
        name, 
        path=DEFAULT_DATASET_SAVING_PATH,
        output_type='Default', 
        debug=False
    ):
        self.location = path + name.replace('/', '-') + '/test_set.pt'
        self.test = torch.load(self.location)
        self.output_type = output_type
        self.debug = debug
        self.reshape_output()
        self.make_prompt()

    def reshape_output(self):
        if self.output_type == 'TabLLM':
            output = [f'class {int(item)}' for item in self.test[0][1]]
            self.test = (self.test[0], self.test[1], output)
        elif self.output_type == 'Ablation_aug':
            temp = self.test[0][1]
            temp = temp.squeeze(-1)
            onehot = np.zeros((temp.size, temp.max() + 1))
            onehot[np.arange(temp.size), temp] = 1
            output = serialize_output(onehot)
            self.test = (self.test[0], self.test[1], output)
        elif self.output_type == 'Default' or self.output_type == 'light':
            pass
        else:
            raise NotImplementedError   

    def make_prompt(self):     
        _, prompt_components, outputs = self.test
        prompts, annotations, labels = prompt_components

        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        if self.output_type == 'TabLLM':
            for item in samples:
                item['labels'] = item['labels'][item['labels'].index('where'): ]

        prompt_input = PROMPT_DICT[self.output_type]
        self.prompts = [prompt_input.format_map(example) for example in samples]
        self.samples = samples

    def get_model_accuracy(self, model, tokenizer):
        tblm = True if self.output_type=='TabLLM' else False
        correct_preds = 0
        for i in range(len(self.prompts)):
            prompt = self.prompts[i]
            reference = self.samples[i]['output']
            pred = self.test_model_on_one_prompt(prompt, model, tokenizer).split('\n')[-1]
            corr = check_correctness(pred, reference, tblm=tblm)
            if self.debug:
                print(corr)
            correct_preds += corr
        if self.debug:
            print(correct_preds / len(self.prompts))
        self.accuracy = correct_preds / len(self.prompts)
    
    def test_model_on_one_prompt(self, prompt, model, tokenizer):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_length=1024,
            top_k=50,
            top_p=0.95,
            num_return_sequences=3,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

class SupervisedTester():
    def __init__(
        self, 
        dataset_list='supervised_datasets.json',
        model_name='unipred_state.pt', 
        dataset_list_path = DEFAULT_DATASET_INDEXING_PATH,
        dataset_path=DEFAULT_DATASET_SAVING_PATH, 
        model_path=DEFAULT_MODEL_PATH, 
        output_type='Default', 
        debug=False
    ):
        self.dataset_list_loc = dataset_list_path + dataset_list
        self.dataset_path = dataset_path
        self.model_loc = model_path + model_name
        self.dataset_list = read_json(self.dataset_list_loc)
        self.output_type = output_type
        self.debug = debug
        self.set_up_model()
        self.acc_dict = {}

    def clear_model(self):
        self.tokenizer = None
        self.model = None
    
    def set_up_model(self):
        self.model, self.tokenizer = setup_model_and_tokenizer('gpt2')
        self.model.load_state_dict(torch.load(self.model_loc, map_location='cuda:0'))
    
    def get_accuracy_on_all_datasets(self):
        for item in self.dataset_list:
            print(item)
            try:
                tester = SupervisedModelTester(item, path=self.dataset_path, output_type=self.output_type, debug=self.debug)
                tester.get_model_accuracy(self.model, self.tokenizer)
                acc = tester.accuracy
                if item in self.acc_dict.keys():
                    self.acc_dict[item][self.output_type] = acc
                else:
                    self.acc_dict[item] = {self.output_type: acc}
            except Exception as e:
                print(e)
                continue
            
    def load_acc_dict(self, path='files/unified/results/supervised.json'):
        try:
            self.acc_dict = read_json(path)
        except:
            self.acc_dict = {}
    
    def save_acc_dict(self, path='files/unified/results/supervised.json'):
        save_json(path, self.acc_dict)
