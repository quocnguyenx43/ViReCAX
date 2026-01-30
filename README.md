# ViReCAX: Vietnamese - Recruiment - Abnormal - Explanation dataset

This repository contains code for our paper: Vuong-Quoc Nguyen Huynh, Bao-Khanh Truong Phuoc, Anh Hoang Tran, Duong Tien Pham, Son T. Luu, Trong-Hop Do. [Abnormal Job Postings Detection with Explanation on Vietnamese Job Hunter Platforms](https://arxiv.org/) was published at [Updating Journal ...](https://arxiv.org/) journal.

The repository is structured as follows:
  - `baseline`: Contains experiments on offline learning baseline models on the dataset.
  - `system`: Contains experiments on online learning systems with three different strategies.

Please contact this email to access the dataset, to sign the dataset user agreement and then receive the dataset:
  - Mr. Luu Thanh Son at sontl@uit.edu.vn
  - Mr. Nguyen Huynh Vuong Quoc at 20521813@gm.uit.edu.vn

# Abstract

Recruitment is a crucial part of human resources for businesses and organizations. The growth of online recruitment websites has made them a popular way to find potential candidates. However, this increase has also led to a rise in fraudulent job postings, which can result in losses of personal information, assets, and company reputation. As a result, with advancements in AI, there is a need for tools to evaluate the verification of job postings. Hence, in this paper, we introduce the **ViReCAX** (**Vi**etnamese **Rec**ruiment - **A**bnormal - **Ex**planation) dataset, a novel dataset designed for detecting abnormal job postings on Vietnamese job hunter platforms, with an emphasis on providing explanations for the model's decisions. The dataset comprises 12,054 manually annotated job postings, covering three tasks: job posting classification (CLEAN, WARNING, SEEDING), aspect-level verification (POSITIVE, NEGATIVE, NEUTRAL, NOT-MENTIONED), and explanation generation. Inter-annotator agreement scores were moderate for the classification tasks (0.6 and 0.59) and high for the text generation task (BLEU-2 score of 0.74 and BERTScore of 0.73). We also propose baseline models for detecting abnormal job postings, utilizing BERT-based models with LSTM or CNN for classification tasks and sequence-to-sequence models for the generation task. Finally, we propose three different online streaming learning strategies to improve the model's ability to adapt to streaming data scenarios.

# Data Sample

- 10 data samples from train dataset: `./data-sample/train_sample_10.csv`

# Citation
For any usage related to all codes and data used from our repository, please cite our following paper:
```
Updating...
```

For any questions, please contact our corresponding author:
  - Mr. Luu Thanh Son at sontl@uit.edu.vn
  - Mr. Nguyen Huynh Vuong Quoc at 20521813@gm.uit.edu.vn for any question about offline learning baseline models and dataset
  - Mr. Tran Hoang Anh at 20521079@gm.uit.edu.vn for any question about online learning systems
