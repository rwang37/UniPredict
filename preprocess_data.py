from preprocessing.make_data import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model-type', 
                    type=str,
                    required=True,
                    help='The input model type. Must be a value among "unipred", "light", and "ablation".')

args = parser.parse_args()
model_type = args.model_type

supervised_dataset_path = 'files/unified/dataset_list/supervised_datasets.json'

if model_type == 'unipred':
    pg = Prompt_generator(dataset_info_list_path=supervised_dataset_path)
    pg.make_all_prompts()
    pg.show_prompt()
    pg.tokenize_supervised_prompt()
    pg.save_tokenized_prompt()
elif model_type == 'light':
    pg = Prompt_generator(dataset_info_list_path=supervised_dataset_path)
    pg.make_all_prompts()
    pg.show_prompt()
    pg.tokenize_supervised_prompt(prompt_type='without_metadata')
    pg.save_tokenized_prompt(name='toked_light_train_set.pt')
elif model_type == 'ablation':
    pg = Prompt_generator(dataset_info_list_path=supervised_dataset_path)
    pg.make_all_prompts(output_type='Ablation')
    pg.show_prompt()
    pg.tokenize_supervised_prompt()
    pg.save_tokenized_prompt(name='toked_abl_aug_train_set.pt')