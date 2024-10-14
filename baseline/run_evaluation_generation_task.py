import os
import random
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import models as md
import functions as func
import my_import as imp


### Find weight model path (lastest epoch)
task_folder = imp.args['task'].replace('-', '_')
prefix_file_name = imp.args['model_type'] + '_' + imp.args['model_name'].split('/')[-1]
all_model_files = os.listdir('./models/' + task_folder + '/')

lastest = 0
for file in all_model_files:
    if file.startswith(prefix_file_name):
        a = int(file.split('_')[-1].split('.')[0])
        if a > lastest:
            lastest = a

lastest = str(lastest)
model_weight_path = './models/' + task_folder + '/' + prefix_file_name + '_' + lastest + '.pth'


###### Model
tokenizer = AutoTokenizer.from_pretrained(imp.args['model_name'])
generation_model = AutoModelForSeq2SeqLM.from_pretrained(imp.args['model_name'])


###### Loading weights
generation_model = generation_model.to(imp.args['device'])
generation_model.load_state_dict(torch.load(model_weight_path))
print(f'model_weight_path: {model_weight_path}')
print('Loading model weight successfully!\n')


###### Evaluating on dev set
print('Evaluation on dev set:')
predictions, references = func.generate_task_3(
    generation_model, tokenizer,
    imp.dev_dataloader, target_len=imp.args['target_len'],
    device=imp.args['device']
)
bertscore, bleuscore, rougescore = func.compute_score_task_3(predictions, references)
random_index = random.randint(0, len(predictions) - 1)
print(f'BERT score (prec, rec, f1): {bertscore}')
print(f'Bleu score (bleu, prec1, prec2, prec3, prec4): {bleuscore}')
print(f'Rouge score (1, 2, L): {rougescore}')
print()
print('*** Random example: ')
print(f'Original @ [{random_index}]: {references[random_index]}')
print(f'Generated @ [{random_index}]: {predictions[random_index]}')
print()
# pd.DataFrame({
#     'original': references,
#     'prediction': predictions,
# }).to_csv('./results/generation_outputs/{prefix_file_name}_inferences_dev_task_3.csv')


###### Evaluating on test test
print('Evaluation on test set:')
predictions, references = func.generate_task_3(
    generation_model, tokenizer,
    imp.test_dataloader, target_len=imp.args['target_len'],
    device=imp.args['device']
)
bertscore, bleuscore, rougescore = func.compute_score_task_3(predictions, references)
# bleu = bleu['bleu']
random_index = random.randint(0, len(predictions) - 1)
print(f'BERT score (prec, rec, f1): {bertscore}')
print(f'Bleu score (bleu, prec1, prec2, prec3, prec4): {bleuscore}')
print(f'Rouge score (1, 2, L): {rougescore}')
print()
print('*** Random example: ')
print(f'Original @ [{random_index}]: {references[random_index]}')
print(f'Generated @ [{random_index}]: {predictions[random_index]}')
print()
# pd.DataFrame({
#     'original': references,
#     'prediction': predictions,
# }).to_csv('./results/generation_outputs/{prefix_file_name}_inferences_test_task_3.csv')