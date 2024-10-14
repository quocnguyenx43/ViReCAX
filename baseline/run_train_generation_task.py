import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import models as md
import functions as func
import my_import as imp


###### Model
tokenizer = AutoTokenizer.from_pretrained(imp.args['model_name'])
generation_model = AutoModelForSeq2SeqLM.from_pretrained(imp.args['model_name'])


###### Change device & setting criterion, optimizer
generation_model = generation_model.to(imp.args['device'])
optimizer = optim.Adam(generation_model.parameters(), lr=imp.args['learning_rate'])


###### Training
print('Training ...')
func.train_task_3(
    generation_model, optimizer, tokenizer,
    epochs=imp.args['epochs'],
    train_dataloader=imp.train_dataloader,
    dev_dataloader=imp.dev_dataloader,
    target_len=imp.args['target_len'],
    saving_path=imp.args['saving_path'],
    device=imp.args['device']
)
