from torch.utils.data import Dataset
from transformers import AutoTokenizer
import utils

    

class RecruitmentDataset(Dataset):
    def __init__(self, df, tokenizer_name, padding_len, task='task-1', target_len=None):

        self.data_x = utils.create_X(df, task)
        self.data_y = utils.create_y(df, task)

        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.padding_len = padding_len
        self.target_len = target_len
        self.task = task

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        item = {}
        x = self.data_x[index]
        y = self.data_y[index]

        if self.task != 'task-3':
            x = x.replace('[CLS]', '').replace('[SEP]', '</s>')
            x = self.tokenizer(x, return_tensors='pt', max_length=self.padding_len, truncation=True, padding='max_length')
            x['input_ids'] = x['input_ids'].squeeze()
            x['attention_mask'] = x['attention_mask'].squeeze()
            if len(x) == 3:
                x['token_type_ids'] = x['token_type_ids'].squeeze()
        else:
            # transform to a list
            if isinstance(index, int):
                x, y  = [x], [y]
                
            x, y = utils.vit5_encode(x, y, self.padding_len, self.target_len, self.tokenizer)

        item.update({'input': x})
        item.update({'label': y})

        return item
    