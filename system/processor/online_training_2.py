from pymongo import MongoClient
from bson.objectid import ObjectId
import time
import model_2n
import pandas as pd
import utils

config = utils.load_config(level = 2)
main_path = utils.get_parent_path(level=2)

mongo_server = config['mongo']['server']
mongo_database = config['mongo']['database']
cls_version = utils.get_best_model_version(mongo_server, mongo_database, config['mongo']['model_eval'])
acsa_version = utils.get_best_model_version(mongo_server, mongo_database, config['mongo']['acsa_eval'])
BATCH_SIZE = config['model']['trigger_batch_size']  # Số lượng dữ liệu trong mỗi batch

#Khởi tạo model
models = model_2n.model(config= config, cls_version = cls_version, acsa_version= acsa_version,model_path=f'{main_path}\models')
print(f'loaded cls model version {cls_version}, acsa model version {acsa_version}')

# Kết nối với MongoDB
client = MongoClient(mongo_server)
db = client[mongo_database]
collection = db[config['mongo']['train']]

# Bộ sưu tập để lưu trữ trạng thái
state_collection = db[config['mongo']['offset']]


def get_last_processed_id():
    state = state_collection.find_one({'name': 'last_processed_id'})
    if state and 'value' in state:
        return state['value']
    else:
        return '0'  # Trả về giá trị mặc định nếu không có dữ liệu

def set_last_processed_id(last_id):
    state_collection.update_one(
        {'name': 'last_processed_id'},
        {'$set': {'value': str(last_id)}},  # Lưu dưới dạng chuỗi để dễ dàng so sánh
        upsert=True
    )


def start_training(_batch, config):
    _batch = pd.DataFrame(_batch)[['prediction', 'acsa','input_data']]

    models.training(_batch)
    
    models.reload_model()

def fetch_and_process_data(config):
    last_processed_id = get_last_processed_id()
    query = {'_id': {'$gt': ObjectId(last_processed_id)}} if last_processed_id != '0' else {}
    cursor = collection.find(query).sort('_id', 1)  # Sắp xếp theo ObjectId để đảm bảo thứ tự
 
    batch = []
    last_id_in_batch = None

    for document in cursor:
        batch.append(document)
        last_id_in_batch = document['_id']
        if len(batch) == BATCH_SIZE:

            start_training(batch, config)
            set_last_processed_id(last_id_in_batch)
            batch = []  # Reset batch sau khi xử lý
            

while True:
    fetch_and_process_data(config)
    time.sleep(1)  # Chờ một khoảng thời gian trước khi kiểm tra dữ liệu mới
