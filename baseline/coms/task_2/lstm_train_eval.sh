#!/bin/bash

echo "vinai/phobert-base"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "vinai/phobert-base" --source_len 256 --batch_size 32 --epochs 20
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "vinai/phobert-base" --source_len 256 --batch_size 32

echo "uitnlp/visobert"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32

echo "uitnlp/CafeBERT"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/CafeBERT" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/CafeBERT" --source_len 512 --batch_size 32

echo "xlm-roberta-base"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "xlm-roberta-base" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "xlm-roberta-base" --source_len 512 --batch_size 32

echo "bert-base-multilingual-cased"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "bert-base-multilingual-cased" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "bert-base-multilingual-cased" --source_len 512 --batch_size 32

echo "distilbert-base-multilingual-cased"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "distilbert-base-multilingual-cased" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "distilbert-base-multilingual-cased" --source_len 512 --batch_size 32
