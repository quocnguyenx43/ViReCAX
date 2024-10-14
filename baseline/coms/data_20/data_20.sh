#!/bin/bash


# top 1
echo "task-1 uitnlp/visobert"
python run_train_cls_task.py --task "task-1" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
python run_evaluation_cls_task.py --task "task-1" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32

echo "task-2 uitnlp/visobert"
python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32

# # top 2
# echo "task-1 uitnlp/visobert"
# python run_train_cls_task.py --task "task-1" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
# python run_evaluation_cls_task.py --task "task-1" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32

# echo "task-2 uitnlp/visobert"
# python run_train_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32 --learning_rate 0.001 --epochs 20
# python run_evaluation_cls_task.py --task "task-2" --model_type "lstm" --model_name "uitnlp/visobert" --source_len 512 --batch_size 32