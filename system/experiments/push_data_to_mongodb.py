from pymongo import MongoClient
import utils
import pandas as pd
from tqdm import tqdm

mongo_server = utils.config['mongo']['server']
mongo_database = utils.config['mongo']['database']
dev = utils.config['mongo']['test']

client = MongoClient(mongo_server)

data = pd.read_csv(f"{utils.main_path}\data\preprocessed\\test_preprocessed.csv")
col = data.columns

y = data[col[0:6]]
X = data[col[6:]]

X = utils.format_data(X)

batch = []

for i in tqdm(range(len(X)-1)):
    record = {'input_data': utils.get_input(X.loc[[i]]), 'prediction': str(y.loc[i]['label']), 'acsa': str(list(y[y.columns[0:4]].loc[i]))}
    batch.append(record)



result = client[mongo_database][dev].insert_many(batch)

print('Insert successfully')
# print(utils.get_input(X.loc[[0]]))
client.close()


