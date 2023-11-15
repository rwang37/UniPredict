from evaluate.tabllm import *
from evaluate.few_shot import *
from evaluate.supervised import *

tg = TabLLMTesterGroup(dataset_list='supervised_datasets.json', debug=False)
tg.load_acc_dict('files/unified/results/supervised.json')
tg.get_supervised_accuracy()
tg.save_acc_dict('files/unified/results/supervised.json')

tg = TabLLMTesterGroup(debug=False)
tg.load_acc_dict('files/unified/results/few_shot.json')
tg.get_few_shot_accuracy()
tg.save_acc_dict('files/unified/results/few_shot.json')

for model, output_type in zip(['unipred', 'abl_aug', 'light'], ['Default', 'Ablation_aug', 'light']):
    print(f'Testing model {output_type}')
    name = model + '_state.pt'

    st = SupervisedTester(model_name=name, output_type=output_type, debug=False)
    st.load_acc_dict()
    st.get_accuracy_on_all_datasets()
    print('Saving...')
    st.save_acc_dict()

    st = FewShotTester(model=name, output_type=output_type, debug=False)
    st.load_acc_dict()
    st.get_accuracy()
    print('Saving...')
    st.save_acc_dict()