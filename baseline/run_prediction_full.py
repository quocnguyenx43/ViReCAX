import random
import pandas as pd

import torch
from torch.utils.data import DataLoader

import argparse as arg
from rich.console import Console
import warnings

import my_datasets as dst

import torch
import torch.nn as nn

import models as md
import functions as func

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import gc



#####################
console = Console(record=True)
warnings.filterwarnings("ignore")

parser = arg.ArgumentParser(description="Params")

parser.add_argument("--path1", type=str, default="")
parser.add_argument("--path2", type=str, default="")
parser.add_argument("--path3", type=str, default="")
parser.add_argument("--output_file", type=str, default="")

parser.add_argument("--source_len_1", type=int, default=256)
parser.add_argument("--source_len_2", type=int, default=256)
parser.add_argument("--source_len_3", type=int, default=768)
parser.add_argument("--target_len", type=int, default=128)

parser.add_argument("--only_test", type=bool, default=False)

parser.add_argument("--batch_size", type=int, default=48)
parser.add_argument("--device", type=str, default='cuda')

# for LSTM
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=1)

# for CNN
parser.add_argument("--num_channels", type=int, default=64)
parser.add_argument("--kernel_size", type=int, default=64)
parser.add_argument("--padding", type=int, default=64)

args = parser.parse_args()
args = vars(args)

only_test = args['only_test']
output_file = args['output_file']

model_path_1 = args['path1']
model_path_2 = args['path2']
model_path_3 = args['path3']

model_name_mapping = {
    'phobert-base': 'vinai/phobert-base',
    'visobert': 'uitnlp/visobert',
    'CafeBERT': 'uitnlp/CafeBERT',
    'xlm-roberta-base': 'xlm-roberta-base',
    'bert-base-multilingual-cased': 'bert-base-multilingual-cased',
    'distilbert-base-multilingual-cased': 'distilbert-base-multilingual-cased',
    'vit5-base': 'VietAI/vit5-base',
    'bartpho-syllable-base': 'vinai/bartpho-syllable-base',
    'bartpho-word-base': 'vinai/bartpho-word-base',
    'None': 'None',
}

model_name_1 = model_name_mapping[model_path_1.split('_')[1]]
model_name_2 = model_name_mapping[model_path_2.split('_')[1]]
model_name_3 = model_name_mapping[model_path_3.split('_')[0]]

model_type_1 = model_path_1.split('_')[0]
model_type_2 = model_path_2.split('_')[0]
model_type_3 = 'simple'

padding_1 = args['source_len_1']
padding_2 = args['source_len_2']
padding_3 = args['source_len_3']
target_padding = args['target_len']

model_path_1 = './models/task_1/' + model_path_1
model_path_2 = './models/task_2/' + model_path_2
model_path_3 = './models/task_3/' + model_path_3

batch_size = args['batch_size']
device = args['device']
params = {
    'hidden_size': args['hidden_size'],
    'num_layers': args['num_layers'],
    'num_channels': args['num_channels'],
    'kernel_size': args['kernel_size'],
    'padding': args['padding'],
}

print(f'model_1: {model_type_1}, model_name_1: {model_name_1}, path_1: {model_path_1}')
print(f'model_2: {model_type_2}, model_name_2: {model_name_2}, path_2: {model_path_2}')
print(f'model_3: {model_type_3}, model_name_3: {model_name_3}, path_3: {model_path_3}')
print()

if not only_test:
    dev_df = pd.read_csv('./data/preprocessed/dev_preprocessed.csv')
    dev_df.explanation = dev_df.explanation.fillna('')
    print(f'dev shape: {dev_df.shape}')

test_df = pd.read_csv('./data/preprocessed/test_preprocessed.csv')
test_df.explanation = test_df.explanation.fillna('')
print(f'test shape: {test_df.shape}')

print()


# Create dataloader
def create_dataloader(df, model_name, source_padding, target_padding, task='task-1'):
    test_dataset = dst.RecruitmentDataset(
        df, tokenizer_name=model_name,
        padding_len=source_padding, target_len=target_padding,
        task=task,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader

# Load model
def load_model(task, model_type, model_name, model_path, params):
    # if task_1
    if task == 'task-1':
        if model_type == 'simple':
            model = md.SimpleCLSModel(pretrained_model_name=model_name)
        else:
            model = md.ComplexCLSModel(model_type=model_type, params=params, pretrained_model_name=model_name)
    
    # if task_2
    elif task == 'task-2':
        if model_type == 'simple':
            model = md.SimpleAspectModel(pretrained_model_name=model_name)
        else:
            model = md.ComplexAspectModel(model_type=model_type, params=params, pretrained_model_name=model_name)

    elif task == 'task-3':
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
        
    return model


### TASK 1
print('TASK 1')
model_1 = load_model('task-1', model_type_1, model_name_1, model_path_1, params)
criterion = nn.CrossEntropyLoss()

if not only_test:
    print('TASK 1 PREDICTION (DEV)')
    task_1_dev_dataloader = create_dataloader(dev_df, model_name_1, padding_1, None, 'task-1')
    predictions_task_1_dev, _, _ = func.evaluate(
        model_1, criterion,
        dataloader=task_1_dev_dataloader, 
        task_running='task-1',
        cm=True, cr=True, last_epoch=True,
        device=device,
    )
    print()

task_1_test_dataloader = create_dataloader(test_df, model_name_1, padding_1, None, 'task-1')
print('TASK 1 PREDICTION (TEST)')
predictions_task_1_test, _, _ = func.evaluate(
    model_1, criterion,
    dataloader=task_1_test_dataloader, 
    task_running='task-1',
    cm=True, cr=True, last_epoch=True,
    device=device,
)
print()

torch.cuda.empty_cache()
gc.collect()

### TASK 2
print('TASK 2')
model_2 = load_model('task-2', model_type_2, model_name_2, model_path_2, params)
criterion = nn.CrossEntropyLoss()

if not only_test:
    print('TASK 2 PREDICTION (DEV)')
    task_2_dev_dataloader = create_dataloader(dev_df, model_name_2, padding_2, None, 'task-2')
    predictions_task_2_dev, _, _ = func.evaluate(
        model_2, criterion,
        dataloader=task_2_dev_dataloader, 
        task_running='task-2',
        cm=True, cr=True, last_epoch=True,
        device=device,
    )
    
print('TASK 2 PREDICTION (TEST)')
task_2_test_dataloader = create_dataloader(test_df, model_name_2, padding_2, None, 'task-2')
predictions_task_2_test, _, _ = func.evaluate(
    model_2, criterion,
    dataloader=task_2_test_dataloader, 
    task_running='task-2',
    cm=True, cr=True, last_epoch=True,
    device=device,
)

torch.cuda.empty_cache()
gc.collect()    

### TASK 3
mapping_aspect = {0: 'trung tính', 1: 'tích cực', 2: 'tiêu cực', 3: 'không đề cập'}
mapping_label = {0: 'rõ ràng', 1: 'cảnh báo', 2: 'có yếu tố thu hút'}

if not only_test:
    df1 = pd.DataFrame(predictions_task_1_dev, columns=['predicted_label'])
    df3 = pd.DataFrame(predictions_task_2_dev, 
                    columns=['predicted_title', 'predicted_desc', 'predicted_comp', 'predicted_other'])
    df_predictions_dev = pd.concat([df1, df3], axis=1)

df2 = pd.DataFrame(predictions_task_1_test, columns=['predicted_label'])
df4 = pd.DataFrame(predictions_task_2_test, 
                   columns=['predicted_title', 'predicted_desc', 'predicted_comp', 'predicted_other'])
df_predictions_test = pd.concat([df2, df4], axis=1)

if model_path_3 == './models/task_3/None':
    if not only_test:
        saving_path = output_file + '_dev.csv'
        df_predictions_dev.to_csv(saving_path)
        print(f'saving_path dev: {saving_path}')

    saving_path = output_file + '_test.csv'
    df_predictions_test.to_csv(saving_path)
    print(f'saving_path test: {saving_path}')

def adding_previous_tasks(df):
    previous_task_outputs = []
    for index in range(len(df)): 
        s = "khía cạnh tiêu đề: " + mapping_aspect[df.iloc[index]['predicted_title']] + " [SEP] " \
            + "khía cạnh mô tả: " + mapping_aspect[df.iloc[index]['predicted_desc']] + " [SEP] " \
            + "khía cạnh công ty: " + mapping_aspect[df.iloc[index]['predicted_comp']] + " [SEP] " \
            + "khía cạnh khác: " + mapping_aspect[df.iloc[index]['predicted_other']] + " [SEP] " \
            + "nhãn chung: " + mapping_label[df.iloc[index]['predicted_label']]  + " [SEP] "
        
        previous_task_outputs.append(s[:-1])

    df['pre_tasks'] = previous_task_outputs
    return df

model_3 = load_model('task-3', model_type_3, model_name_3, model_path_3, params)
tokenizer_3 = AutoTokenizer.from_pretrained(model_name_3)

if not only_test:
    print('TASK 3 PREDICTION (DEV)')
    df_predictions_dev = adding_previous_tasks(df_predictions_dev)
    dev_df.pre_tasks = df_predictions_dev.pre_tasks
    task_3_dataloader_dev = create_dataloader(dev_df, model_name_3, padding_3, target_padding, 'task-3')
    predictions_3_dev, references_3_dev = func.generate_task_3(
        model_3, tokenizer_3,
        task_3_dataloader_dev, target_len=target_padding,
        device=device
    )

    bertscore, bleuscore, rougescore = func.compute_score_task_3(predictions_3_dev, references_3_dev)
    random_index = random.randint(0, len(predictions_3_dev) - 1)
    print(f'BERT score (prec, rec, f1): {bertscore}')
    print(f'Bleu score (bleu, prec1, prec2, prec3, prec4): {bleuscore}')
    print(f'Rouge score (1, 2, L): {rougescore}')
    print()
    print('*** Random example: ')
    print(f'Original @ [{random_index}]: {references_3_dev[random_index]}')
    print(f'Generated @ [{random_index}]: {predictions_3_dev[random_index]}')
    print()

    df_predictions_dev['generated_text'] = pd.Series(predictions_3_dev)
    
    saving_path = output_file + '_dev.csv'
    df_predictions_dev.to_csv(saving_path)
    print(f'saving_path: {saving_path}')


print('TASK 3 PREDICTION (TEST) GROUNDTRUTH')
task_3_dataloader_test = create_dataloader(test_df, model_name_3, padding_3, target_padding, 'task-3')
predictions_3_test_1, references_3_test = func.generate_task_3(
    model_3, tokenizer_3,
    task_3_dataloader_test, target_len=target_padding,
    device=device
)

bertscore, bleuscore, rougescore = func.compute_score_task_3(predictions_3_test_1, references_3_test)
random_index = random.randint(0, len(predictions_3_test_1) - 1)
print(f'BERT score (prec, rec, f1): {bertscore}')
print(f'Bleu score (bleu, prec1, prec2, prec3, prec4): {bleuscore}')
print(f'Rouge score (1, 2, L): {rougescore}')
print()
print('*** Random example: ')
print(f'Original @ [{random_index}]: {references_3_test[random_index]}')
print(f'Generated @ [{random_index}]: {predictions_3_test_1[random_index]}')
print()

print('TASK 3 PREDICTION (TEST) PREDICTED')
df_predictions_test = adding_previous_tasks(df_predictions_test)
test_df.pre_tasks = df_predictions_test.pre_tasks
task_3_dataloader_test = create_dataloader(test_df, model_name_3, padding_3, target_padding, 'task-3')
predictions_3_test_2, references_3_test = func.generate_task_3(
    model_3, tokenizer_3,
    task_3_dataloader_test, target_len=target_padding,
    device=device
)

bertscore, bleuscore, rougescore = func.compute_score_task_3(predictions_3_test_2, references_3_test)
random_index = random.randint(0, len(predictions_3_test_2) - 1)
print(f'BERT score (prec, rec, f1): {bertscore}')
print(f'Bleu score (bleu, prec1, prec2, prec3, prec4): {bleuscore}')
print(f'Rouge score (1, 2, L): {rougescore}')
print()
print('*** Random example: ')
print(f'Original @ [{random_index}]: {references_3_test[random_index]}')
print(f'Generated @ [{random_index}]: {predictions_3_test_2[random_index]}')
print()


df_predictions_test['generated_text_gt'] = pd.Series(predictions_3_test_1)
df_predictions_test['generated_text_pred'] = pd.Series(predictions_3_test_2)

saving_path = output_file + '_test.csv'
df_predictions_test.to_csv(saving_path)
print(f'saving_path: {saving_path}')
