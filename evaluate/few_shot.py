import torch
import numpy as np

from .utils import *
from sklearn.model_selection import train_test_split

class FewShotModelTester():
    def __init__(
        self, 
        name, 
        train_ratio,
        model='unipred_state.pt',
        model_path=DEFAULT_MODEL_PATH,
        path=DEFAULT_DATASET_SAVING_PATH,
        output_type='Default', 
        debug=False
    ):
        self.train_ratio = train_ratio
        self.model_loc = model_path + model
        self.location = path + name.replace('/', '-')
        self.train = torch.load(self.location + '/train_set.pt')
        self.test = torch.load(self.location + '/test_set.pt')
        self.output_type = output_type
        self.debug = debug
        self.reshape_output()
        self.merge_data()
        self.split(self.train_ratio)

        self.make_prompt()
        self.set_up_model()

    def reshape_output(self):
        if self.output_type == 'TabLLM':
            output = [f'class {int(item)}' for item in self.train[0][1]]
            self.train = (self.train[0], self.train[1], output)
            output = [f'class {int(item)}' for item in self.test[0][1]]
            self.test = (self.test[0], self.test[1], output)
        elif self.output_type == 'Ablation_aug':
            temp = self.train[0][1]
            temp = temp.squeeze(-1)
            onehot = np.zeros((temp.size, temp.max() + 1))
            onehot[np.arange(temp.size), temp] = 1
            output = serialize_output(onehot)
            self.train = (self.train[0], self.train[1], output)

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
        
    def merge_data(self):

        _, prompt_components, outputs = self.train
        prompts, annotations, labels = prompt_components
        self.gt = _[1]
        samples = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]
        # print([samples[i]['output'] for i in range(10)])
        # print(self.gt[:10])
        # print(samples[0]['labels'])

        _, prompt_components, outputs = self.test
        prompts, annotations, labels = prompt_components

        samples2 = [{
            'prompt': prompts[i],
            'annotations': annotations[i],
            'labels': labels[i],
            'output': outputs[i],
        } for i in range(len(prompts))]

        samples.extend(samples2)
        self.samples = samples
        # print(self.samples[0])

    def split(self, train_ratio):
        self.train, self.test = train_test_split(
            self.samples,
            train_size=train_ratio,
            random_state=42
        )

    def fine_tune(self, epochs=30):
        self.model.train()
        data_module = make_supervised_data_module(tokenizer=self.tokenizer, data=self.train, prompt_type=self.output_type)
        training_args = TrainingArguments("files/model_checkpoints", num_train_epochs=epochs)
        training_args = training_args.set_save(strategy="steps", steps=10000, total_limit=10)

        trainer = Trainer(model=self.model, tokenizer=self.tokenizer, args=training_args, **data_module)
        trainer.train()
    
    def make_prompt(self):     
        if self.output_type == 'TabLLM':
            for item in self.test:
                item['labels'] = item['labels'][item['labels'].index('where'): ]

        prompt_input = PROMPT_DICT[self.output_type]
        self.prompts = [prompt_input.format_map(example) for example in self.test]
        # print(self.prompts[0])

    def get_model_accuracy(self):
        tblm = True if self.output_type=='TabLLM' else False
        self.model.eval()
        correct_preds = 0
        for i in range(len(self.prompts)):
            prompt = self.prompts[i]
            # print(prompt)
            reference = self.test[i]['output']
            pred = self.test_model_on_one_prompt(prompt).split('\n')[-1]
            if self.debug:
                print(pred)
                print(reference)
            corr = check_correctness(pred, reference, tblm=tblm)
            if self.debug:
                print(corr)
            correct_preds += corr
        if self.debug:
            print(correct_preds / len(self.prompts))
        self.accuracy = correct_preds / len(self.prompts)
    
    def test_model_on_one_prompt(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        # print(inputs, inputs['input_ids'].squeeze(0).shape, tokenizer.decode(inputs['input_ids'].squeeze(0)))
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_length=1024,
            top_k=50,
            top_p=0.95,
            num_return_sequences=3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def clear_model(self):
        self.tokenizer = None
        self.model = None
    
    def set_up_model(self):
        self.model, self.tokenizer = setup_model_and_tokenizer('gpt2')
        self.model.load_state_dict(torch.load(self.model_loc, map_location='cuda:0'))

class FewShotTester():
    def __init__(
        self, 
        model='unipred_state.pt',
        dataset_list='few_shot_datasets.json',
        model_path=DEFAULT_MODEL_PATH,
        dataset_list_path = DEFAULT_DATASET_INDEXING_PATH,
        dataset_path=DEFAULT_DATASET_SAVING_PATH, 
        output_type='Default', 
        debug=False,
        test_mode=False
    ):
        self.model = model
        self.model_path = model_path
        self.dataset_list_loc = dataset_list_path + dataset_list
        self.dataset_path = dataset_path
        self.dataset_list = read_json(self.dataset_list_loc)
        if test_mode:
            self.dataset_list = self.dataset_list[:3]
        self.output_type = output_type
        self.debug = debug
        self.acc_dict = {}
        self.delta_acc = 0
    
    def get_accuracy(self):
        for i in range(1, 10):
            ratio = i/10
            for item in self.dataset_list:
                try:
                    tester = FewShotModelTester(
                        item, 
                        ratio, 
                        model=self.model, 
                        model_path=self.model_path, 
                        path=self.dataset_path, 
                        output_type=self.output_type, 
                        debug=self.debug
                    )
                    tester.fine_tune()
                    tester.get_model_accuracy()
                    acc = tester.accuracy
                    if self.debug:
                        print(acc)
                    if item in self.acc_dict.keys():
                        if str(ratio) in self.acc_dict[item].keys():
                            if self.output_type in self.acc_dict[item][str(ratio)].keys():
                                basl = self.acc_dict[item][str(ratio)][self.output_type]
                                self.delta_acc += (acc - basl)
                                # print(self.delta_acc)
                            self.acc_dict[item][str(ratio)][self.output_type] = acc
                        else:
                            self.acc_dict[item][str(ratio)] = {self.output_type: acc}
                    else:
                        self.acc_dict[item] = {str(ratio): {self.output_type: acc}}
                except Exception as e:
                    if self.debug:
                        print('=======debug output=======')
                        print(e)
                    continue

    def load_acc_dict(self, path='files/unified/results/few_shot.json'):
        try:
            self.acc_dict = read_json(path)
        except:
            self.acc_dict = {}
    
    def save_acc_dict(self, path='files/unified/results/few_shot.json'):
        save_json(path, self.acc_dict)