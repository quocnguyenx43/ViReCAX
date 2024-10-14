import pandas as pd

import torch
from torch.utils.data import DataLoader

import argparse as arg
from rich.console import Console
import warnings

import my_datasets as dst


#####################
console = Console(record=True)
warnings.filterwarnings("ignore")


### Params 
parser = arg.ArgumentParser(description="Params")

parser.add_argument("--task", type=str, default='task-1')
parser.add_argument("--model_type", type=str, default='simple')
parser.add_argument("--model_name", type=str, default='vinai/phobert-base')
parser.add_argument("--source_len", type=int, default=200)
parser.add_argument("--target_len", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--fine_tune", action="store_true", default=True)

# for LSTM
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=1)

# for CNN
parser.add_argument("--num_channels", type=int, default=64)
parser.add_argument("--kernel_size", type=int, default=64)
parser.add_argument("--padding", type=int, default=64)

args = parser.parse_args()
args = vars(args)


# Model params 
params = {}
if args['model_type'] == 'lstm':
    params['hidden_size'] = args['hidden_size']
    params['num_layers'] = args['num_layers']
elif args['model_type'] == 'cnn':
    params['num_channels'] = args['num_channels']
    params['kernel_size'] = args['kernel_size']
    params['padding'] = args['padding']


try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except RuntimeError as e:
    device = 'cpu'
saving_path = './models/' + args['task'].replace('-', '_') + '/' + args['model_type'] + '_' + args['model_name'].split('/')[-1]

args['device'] = device
args['saving_path'] = saving_path


### Read data
train_df = pd.read_csv('./data/preprocessed/train_preprocessed.csv')
dev_df = pd.read_csv('./data/preprocessed/dev_preprocessed.csv')
test_df = pd.read_csv('./data/preprocessed/test_preprocessed.csv')

train_df.dropna(subset=['explanation'], inplace=True)
dev_df.dropna(subset=['explanation'], inplace=True)
test_df.dropna(subset=['explanation'], inplace=True)

args['train_shape'] = train_df.shape
args['dev_shape'] = dev_df.shape
args['test_shape'] = test_df.shape


### Dataset
train_dataset = dst.RecruitmentDataset(
    train_df, tokenizer_name=args['model_name'],
    padding_len=args['source_len'], target_len=args['target_len'],
    task=args['task'],
)
dev_dataset = dst.RecruitmentDataset(
    dev_df, tokenizer_name=args['model_name'],
    padding_len=args['source_len'], target_len=args['target_len'],
    task=args['task'],
)
test_dataset = dst.RecruitmentDataset(
    test_df, tokenizer_name=args['model_name'],
    padding_len=args['source_len'], target_len=args['target_len'],
    task=args['task'],
)


### Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=args['batch_size'], shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)


### Printing args
p_args = args.copy()
if p_args['task'] == 'task-3':
    p_args.pop('model_type')
    p_args.pop('fine_tune')
    p_args.pop('hidden_size')
    p_args.pop('num_layers')
    p_args.pop('num_channels')
    p_args.pop('kernel_size')
    p_args.pop('padding')
else:
    p_args.pop('target_len')
    if p_args['model_type'] == 'lstm':
        p_args.pop('num_channels')
        p_args.pop('kernel_size')
        p_args.pop('padding')
    elif p_args['model_type'] == 'cnn':
        p_args.pop('hidden_size')
        p_args.pop('num_layers')
    else:
        p_args.pop('hidden_size')
        p_args.pop('num_layers')
        p_args.pop('num_channels')
        p_args.pop('kernel_size')
        p_args.pop('padding') 

print()
for key, value in p_args.items():
    if p_args['task'] in ['task-1', 'task-2'] and key == 'source_len':
        print(f'padding_len: {value}')
        continue
    print(f'{key}: {value}')
print()