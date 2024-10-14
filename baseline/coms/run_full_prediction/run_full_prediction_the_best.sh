#!/bin/bash

python run_prediction_full.py --path1 "lstm_visobert_13.pth" --source_len_1 512 \
                              --path2 "lstm_distilbert-base-multilingual-cased_19.pth" --source_len_2 512 \
                              --path3 "bartpho-word-base_2.pth" \
                              --only_test True \
                              --output_file "outputs/lstm-viso_cnn-distil_bartpho-word"

python run_prediction_full.py --path1 "lstm_visobert_13.pth" --source_len_1 512 \
                              --path2 "lstm_distilbert-base-multilingual-cased_19.pth" --source_len_2 512 \
                              --path3 "bartpho-syllable-base_2.pth" \
                              --only_test True \
                              --output_file "outputs/lstm-viso_cnn-distil_bartpho-syllable"

python run_prediction_full.py --path1 "lstm_visobert_13.pth" --source_len_1 512 \
                              --path2 "lstm_distilbert-base-multilingual-cased_19.pth" --source_len_2 512 \
                              --path3 "vit5-base_2.pth" \
                              --only_test True \
                              --output_file "outputs/lstm-viso_cnn-distil_vit5"

echo "upload file"
python upload_file_to_drive.py --service_account_file "/kaggle/input/new-service-account/service_account.json" \
                               --folder_id "1H6GRQN3oQ4n2FmTrelQXbfV-wAEreMNM" \
                               --local_folder "outputs/"