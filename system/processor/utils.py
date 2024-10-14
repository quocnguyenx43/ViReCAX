import yaml
from pathlib import Path
from pymongo import MongoClient
import re
import os
import torch
import numpy as np


def get_device():
    """
    Kiểm tra xem CUDA có khả dụng hay không. 
    Nếu có, sử dụng GPU, nếu không, sử dụng CPU.

    Returns:
        torch.device: Thiết bị sẽ được sử dụng (GPU hoặc CPU)
    """
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

def load_config(level, path=''):
    config_folder = Path(__file__)
    if path == '':
        for i in range(level):
            config_folder = config_folder.parent
    else:
        config_folder = Path(__file__).parent + path

    with open(f"{config_folder}\config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_parent_path(level):
    path = Path(__file__)
    for i in range(level):
        path = path.parent
    return path


def check_model_existence(directory, version):
    # Định nghĩa biểu thức chính quy để lấy số nằm giữa ký tự '_' và '.'
    pattern = r'_(\d+)\.'

    # Duyệt qua các tệp trong thư mục
    for filename in os.listdir(directory):
        # Kiểm tra xem tệp có khớp với pattern không
        match = re.search(pattern, filename)
        if match:
            number = int(match.group(1))
            if str(number) == str(version):
                return True
    return False

def get_best_model_version(mongo, database_name, collection_name):
    model_path = f"{get_parent_path(level=2)}\models"
    # Kết nối tới MongoDB
    client = MongoClient(mongo)
    
    # Chọn database và collection
    db = client[database_name]
    collection = db[collection_name]
    
    # Truy vấn record với giá trị "avg_f1" cao nhất
    record = collection.find_one(sort=[("avg_f1", -1)])
    version = record.get('version')
    doc_count = collection.count_documents({})

    while(doc_count != 0):
        if check_model_existence(model_path, version):
            # Đóng kết nối
            client.close()
            break
        else:
            collection.delete_one({"_id": record["_id"]})
            print('deleting')
            record = collection.find_one(sort=[("avg_f1", -1)])
            version = record.get('version')
            doc_count = collection.count_documents({})
        if collection.count_documents({}) == 0:
            print('No record or the model does not exist')
            client.close()
            break
    return version

def my_accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]


def my_recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
    return temp/ y_true.shape[0]


def my_hamming_loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])


def my_precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
    return temp/ y_true.shape[0]


def my_f1_score(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]

