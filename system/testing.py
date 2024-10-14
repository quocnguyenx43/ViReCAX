# Import some necessary modules
from pymongo import MongoClient
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import torch


def load_config(level, path=''):
    if path == '':
        for i in range(level):
            config_folder = Path(__file__).parent
    else:
        config_folder = Path(__file__).parent + path

    with open(f"{config_folder}\config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

# current_path = Path(__file__).parent
# config = load_config(f"{current_path}\config.yaml")
config=load_config(level = 1)

# Connect to MongoDB

try:
   client = MongoClient('localhost',27017)

   db = client["kafka_test"]
   print("Connected successfully!")
except:  
   print("Could not connect to MongoDB")

# X = np.array(pd.read_csv(r"E:\Learning\Docker_basic\basic_kafka\kltn\data\x_train_200.csv"))
# y = np.array(pd.read_csv(r"E:\Learning\Docker_basic\basic_kafka\kltn\data\y_train_200.csv"))

# # Lặp để nhận tin nhắn
# for i in range(len(X)):
#     input =  X[i][0]
#     cls =  y[i][5]

    # # Create dictionary and ingest data into MongoDB
    # try:
    #    record = {'prediction': cls, 'data': input}
    #    rec_id1 = db["dev"].insert_one(record)
    #    print("Data inserted with record ids", rec_id1)
    # except:
    #    print("Could not insert into MongoDB")

# record = {'version': 1, "acc":0.103949, "avg_pre": 0.343098, \
#           'avg_recall': 0.326604, 'avg_f1': 0.161342}
# rec_id1 = db["acsa_evaluation"].insert_one(record)

# client.close()


# def get_dev(connection_string, database_name, collection_name):
#    # Kết nối tới MongoDB
#    client = MongoClient(connection_string)
   
#    # Chọn database và collection
#    db = client[database_name]
#    collection = db[collection_name]
   
#    t = collection.find_one(sort=[("avg_f1", -1)])
   
#    client.close()
#    return t.get('avg_f1')

# print(get_dev(config['mongo']['server'], config['mongo']['database'], config['mongo']['model_eval']))

# print([1] + torch.tensor([1,2,3,4]).tolist())

# s = "[1,2,3,4]"
# print(type(eval(s)))

# collection = db['dev']

# # Lấy toàn bộ dữ liệu từ collection
# data = list(collection.find())

# # Đóng kết nối
# client.close()

# # Chuyển đổi dữ liệu thành DataFrame
# df = pd.DataFrame(data)
# print(df[['prediction', 'acsa']]['acsa'])
from torch.utils.data import Dataset, DataLoader

class DatasetTok(Dataset):
    def __init__(self, data_x, data_y, label_type, tokenizer, padding_length = 514, exp_padding = 60):
        self.data_x = data_x
        self.data_y = data_y
        self.padding_length = padding_length
        self.exp_padding = exp_padding
        self.tokenizer = tokenizer
        self.label_type = label_type

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        item = {}

        x = self.data_x[index]
        x = x.replace('[CLS]', '')
        x = x.replace('[SEP]', '</s>')
        x = self.tokenizer(x, return_tensors='pt', padding= 'max_length', truncation=True, max_length = self.padding_length)
        x['input_ids'] = x['input_ids'].squeeze()
        x['attention_mask'] = x['attention_mask'].squeeze()
        if len(x) == 3:
          x['token_type_ids'] = x['token_type_ids'].squeeze()
        item.update({'input': x})

        y_cls = self.data_y['prediction'][index]
        y_acsa = self.data_y['acsa'][index]

        if "acsa" in self.label_type:
          one_hot_tensor = F.one_hot(torch.tensor(eval(y_acsa)), 4)
          item.update({'acsa': one_hot_tensor.squeeze(dim = 1).float()})

        if "cls" in self.label_type:
          one_hot_label = F.one_hot(torch.tensor(int(y_cls)), 3)
          item.update({'cls':one_hot_label.float()})

        # if "explaination" in self.label_type:
        #   tokenized_y = self.tokenizer(y[6], return_tensors='pt')
        #   size_y = tokenized_y['input_ids'].size()[1]
        #   padded_y = torch.nn.functional.pad(tokenized_y, (0, self.exp_padding - size_y), value = 1)
        #   item.update({'explaination':padded_y})
        return item

# def getDataloader(data):
#     # Bart 600
#     # Viso 400
#     # Distiled, XLM 500
#     # Phobert 200

#     # Tạo đối tượng dataset
#     train_dataset = DatasetTok(np.array(data['data']), data[['prediction', 'acsa']], ['cls', 'acsa'], padding_length = 500, tokenizer=self.tokenizer, exp_padding = 50)
#     train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
#     return train_dataloader

collection = db['dev']

# Lấy toàn bộ dữ liệu từ collection
data = list(collection.find())

# Đóng kết nối
client.close()

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(data)

from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

train_dataset = DatasetTok(np.array(df['input_data']), df[['prediction', 'acsa']], ['cls', 'acsa'], padding_length = 500, tokenizer=tokenizer, exp_padding = 50)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# count = 0
# for batch in train_dataloader:
#    print(count)
#    count+=1

