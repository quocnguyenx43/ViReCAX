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
# acsa_version = utils.get_best_model_version(mongo_server, mongo_database, config['mongo']['acsa_eval'])
# cls_version = '53'
acsa_version = '1'

# Khởi tạo model
models = model_2n.model(config=config, cls_version=cls_version, acsa_version=acsa_version, model_path=f'{main_path}\models')
print(f'loaded cls model version {cls_version}, acsa model version {acsa_version}')

models.evaluate(type_eval='test')