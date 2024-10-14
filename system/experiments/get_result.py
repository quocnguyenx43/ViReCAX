from pymongo import MongoClient
import utils
import pandas as pd

mongo_server = utils.config['mongo']['server']
mongo_database = utils.config['mongo']['database']


# Kết nối tới MongoDB
client = MongoClient(mongo_server)

# Chọn database và collection
db = client[mongo_database]
collection = db[utils.config['mongo']['model_eval']]

# Lấy toàn bộ dữ liệu từ collection
test = list(collection.find())

# Đóng kết nối
client.close()

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(test)

df.to_csv(r'E:\Learning\Docker_basic\basic_kafka\kltn\experiments\results\task1_distil_lstm_20_80_5epoch_s2.csv')
