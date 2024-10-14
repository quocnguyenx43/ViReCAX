from pymongo import MongoClient
import utils

mongo_server = utils.config['mongo']['server']
mongo_database = utils.config['mongo']['database']
off1 = utils.config['mongo']['offset']
off2 = utils.config['mongo']['offset_2']

client = MongoClient(mongo_server)
try:
    client[mongo_database][off1].delete_many({})
    print("Deleted offset")
except:
    pass

try:
    client[mongo_database][off2].delete_many({})
    print("Deleted offset_2")
except:
    pass
client.close()