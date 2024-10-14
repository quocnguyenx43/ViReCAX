#!/bin/bash

echo "download model files"
gdown --id 1zyJmxlg290cK--Di7A6sweREfg1OUO9R
gdown --id 1YNgX7gQgFcBmsp1OgLOfiuhm7FprOIHd
gdown --id 1KZWrJiz3HoznBgw1RyJoFelGNUrBIHg1
gdown --id 1vRd5d44nZcFPL6HPbleFAA1n2IAhE1b9
gdown --id 1yy3BTE9s9z9pTNz4CPFOoOVoZrGvyG-a
echo "download model files done!"

echo "move model files"
mv task_1-lstm_visobert_13.pth models/task_1/lstm_visobert_13.pth
mv task_2-lstm_distilbert-base-multilingual-cased_19.pth models/task_2/lstm_distilbert-base-multilingual-cased_19.pth
mv vit5-base_2.pth models/task_3/vit5-base_2.pth
mv bartpho-word-base_2.pth models/task_3/bartpho-word-base_2.pth
mv bartpho-syllable-base_2.pth models/task_3/bartpho-syllable-base_2.pth
echo "move model files done"


