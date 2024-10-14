from pymongo import MongoClient
import utils
import pandas as pd
from tqdm import tqdm
import time

mongo_server = utils.config['mongo']['server']
mongo_database = utils.config['mongo']['database']
train = utils.config['mongo']['train']
batch_size = utils.config['experiment']['batch_size']

client = MongoClient(mongo_server)

data = pd.read_csv("{}\data\{}".format(utils.main_path,"train_20_2.csv"))

col = data.columns

y = data[col[0:6]]
X = data[col[6:]]

X = utils.format_data(X)
 
print(X)

num_batch = utils.config['experiment']['number_of_batch']

if num_batch == 0:
    num_batch = len(X)//batch_size


for i in tqdm(range(num_batch)):
    batch = []
    for j in range(i * batch_size, (i+1)*batch_size):
        record = {'input_data': utils.get_input(X.loc[[j]]), 'prediction': str(y.loc[j]['label']), 'acsa': str(list(y[y.columns[0:4]].loc[j]))}
        batch.append(record)
    time.sleep(utils.config['experiment']['delay'])
    result = client[mongo_database][train].insert_many(batch)
    print('Insert successfully')
    



# print(utils.get_input(X.loc[[0]]))
client.close()


