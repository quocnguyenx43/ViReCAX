import numpy as np
import torch
import torch.nn.functional as F


def create_X(df, task):
    infor_cols = [
        'tiêu đề', 'ngành nghề',
        'mô tả', 'kinh nghiệm', 'học vấn', 'bằng cấp', 'quyền lợi',
        'tên công ty', 'địa chỉ', 'số điện thoại', 'người liên hệ',
        'số lượng tuyển', 'hình thức hợp đồng', 'hình thức trả lương', 'lương tối thiểu', 'lương tối đa', 'giới tính', 'năm sinh', 'tuổi', 'tuổi thấp nhất', 'tuổi cao nhất',
    ]

    # Get data
    pre_tasks = df['pre_tasks']
    X_data = df.drop(['title_aspect', 'desc_aspect', 'company_aspect', 'other_aspect', 'label', 'explanation', 'pre_tasks'], axis=1)

    # Combine
    X_combined = []
    for q, row in X_data.iterrows():
        inp = '[CLS] '
        for idx, content in enumerate(row):
            inp = inp + infor_cols[idx] + ": " + str(content) + " [SEP] "
        if task == 'task-3':
            inp = inp + pre_tasks[q]
        else:
            inp = inp[:-1]
        X_combined.append(inp)

    return X_combined


def create_y(df, task):
    if task == 'task-1':
        label = torch.tensor(df['label'].values)
        label = F.one_hot(label, num_classes=3).float()
    elif task =='task-2':
        label = df[['title_aspect', 'desc_aspect', 'company_aspect', 'other_aspect']].to_numpy()
        label = np.eye(4)[label]
        label = torch.from_numpy(label).float()
    else:
        label = df['explanation'].to_numpy()

    return list(label)


def vit5_encode(source_text, target_text, source_len, target_len, tokenizer):
    source_encoded = tokenizer.batch_encode_plus(
        source_text,
        max_length= source_len,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    source_encoded['input_ids'] = source_encoded['input_ids'].squeeze()
    source_encoded['attention_mask'] = source_encoded['attention_mask'].squeeze()

    target_encoded = tokenizer.batch_encode_plus(
        target_text,
        max_length= target_len,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors='pt'
    )
    target_encoded['input_ids'] = target_encoded['input_ids'].squeeze()
    target_encoded['attention_mask'] = target_encoded['attention_mask'].squeeze()

    return source_encoded, target_encoded