from pymongo import MongoClient
from bson.objectid import ObjectId
import time
import model_2n
import pandas as pd
import utils

config = utils.load_config(level=2)
main_path = utils.get_parent_path(level=2)

mongo_server = config['mongo']['server']
mongo_database = config['mongo']['database']
cls_version = utils.get_best_model_version(mongo_server, mongo_database, config['mongo']['model_eval'])
acsa_version = utils.get_best_model_version(mongo_server, mongo_database, config['mongo']['acsa_eval'])
M = config['model']['M_time_trigger_batch_size']
BATCH_SIZE = config['model']['trigger_batch_size']  # Số lượng dữ liệu trong mỗi batch
BATCH_SIZE_2 = M * config['model']['trigger_batch_size']


# Khởi tạo model
models = model_2n.model(config=config, cls_version=cls_version, acsa_version=acsa_version, model_path=f'{main_path}\models')
print(f'loaded cls model version {cls_version}, acsa model version {acsa_version}')

# Kết nối với MongoDB
client = MongoClient(mongo_server)
db = client[mongo_database]
train_collection = db[config['mongo']['train']]

# Bộ sưu tập để lưu trữ trạng thái
state_collection = db[config['mongo']['offset']]
state_collection_2 = db[config['mongo']['offset_2']]

def get_last_processed_id(state_collection):
    state = state_collection.find_one({'name': 'last_processed_id'})
    if state and 'value' in state:
        return state['value']
    else:
        return train_collection.find_one(sort=[("_id", 1)])['_id']  # Trả về giá trị mặc định nếu không có dữ liệu

def set_last_processed_id(state_collection, last_id):
    state_collection.update_one(
        {'name': 'last_processed_id'},
        {'$set': {'value': str(last_id)}},  # Lưu dưới dạng chuỗi để dễ dàng so sánh
        upsert=True
    )
    print('updated')

def start_training(_batch, config):
    _batch = pd.DataFrame(_batch)[['prediction', 'acsa', 'input_data']]
    models.training(_batch)
    models.reload_model()

def fetch_and_process_data(config):
    last_processed_id_1 = get_last_processed_id(state_collection)
    last_processed_id_2 = get_last_processed_id(state_collection_2)

    query = {'_id': {'$gt': ObjectId(last_processed_id_1)}} if last_processed_id_1 != '0' else {}
    cursor = train_collection.find(query).sort('_id', 1)  # Sắp xếp theo ObjectId để đảm bảo thứ tự
    

    batch = []
    last_id_in_batch = None
    last_id_in_batch_2 = None
    batch_count = 0

    for document in cursor:
        batch.append(document)
        last_id_in_batch = document['_id']
        batch_2 = []
        if len(batch) == BATCH_SIZE:
            query_2 = {'_id': {'$gt': ObjectId(last_processed_id_2), '$lte': ObjectId(last_id_in_batch)}} if last_processed_id_2 != '0' else {}
            cursor_2 = train_collection.find(query_2).sort('_id', 1)
            batch_2 = list(cursor_2)
            if len(batch_2) != BATCH_SIZE_2:
            # print(f"last2: {last_processed_id_2}, newest: {last_id_in_batch}")
                print("1---------------")
                print(last_id_in_batch)
                start_training(batch, config)
                # print(f"Len B1: {len(batch)}")
                set_last_processed_id(state_collection, last_id_in_batch)
                batch = []  # Reset batch sau khi xử lý
            else:
                last_id_in_batch_2 = batch_2[-1]['_id']
                print("2----------------")
                print(last_id_in_batch_2)
                start_training(batch_2, config)
                # print(f"Len B1: {len(batch)} - Len B2: {len(batch_2)}")
                set_last_processed_id(state_collection_2, last_id_in_batch_2)
                set_last_processed_id(state_collection, last_id_in_batch_2)

while True:
    fetch_and_process_data(config)
    time.sleep(1)  # Chờ một khoảng thời gian trước khi kiểm tra dữ liệu mới
