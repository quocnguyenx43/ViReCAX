from pyspark.sql import SparkSession
import model_2
from pyspark.sql.functions import from_json
import json
from pyspark.sql.types import StructType, StringType
from kafka import KafkaProducer, KafkaConsumer
import utils

config = utils.load_config(level = 2)
main_path = utils.get_parent_path(level=2)

topic_name = config['kafka']['request']
response_topic = config['kafka']['respond']
model_update_topic = config['kafka']['update']
bootstrap_servers = config['kafka']['server']
mongo_server = config['mongo']['server']
mongo_database = config['mongo']['database']

scala_version = '2.12'
spark_version = '3.0.1'


cls_version = utils.get_best_model_version(mongo_server, mongo_database, config['mongo']['model_eval'])
acsa_version = utils.get_best_model_version(mongo_server, mongo_database, config['mongo']['acsa_eval'])

#Khởi tạo model
models = model_2.model(config= config, cls_version = cls_version, acsa_version= acsa_version,model_path=f'{main_path}\models')
print(f'loaded cls model version {cls_version}, acsa model version {acsa_version}')

#Khởi tạo spark session
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:2.8.0'
]

spark = SparkSession.builder.master("local").appName("kafka-example").config("spark.jars.packages", ",".join(packages)).getOrCreate()

print("spark session created")

# Khởi tạo Kafka producer
producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda x: json.dumps(x).encode('utf-8'))

# Đọc dữ liệu từ Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", bootstrap_servers) \
    .option("subscribe", topic_name) \
    .load()

print('reading stream')

# Định nghĩa schema cho DataFrame
schema = StructType() \
    .add("id", StringType()) \
    .add("data", StringType())  # Đặt kiểu dữ liệu phù hợp cho dữ liệu

# Chuyển đổi JSON thành các cột trong DataFrame
df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
    .select("key" , from_json("value", schema).alias("json")) \
    .select("key" ,"json.id", "json.data")

def deep_learning_model(batch_df, batch_id):
    # Đối với mỗi batch, lấy các giá trị và áp dụng mô hình
    for row in batch_df.collect():
        id = row['id']
        correlation_id = row['key']
        value = row['data']
        prediction = models.predict(value)
        # print(f"Key: {key}, ID: {correlation_id} ,Prediction: {prediction}")
        # Tạo dictionary để gửi về Kafka
        result = {
            'id': id,
            'prediction': prediction[0],
            'acsa': str(prediction[1]),
            'data': value
        }
        # print(f'{correlation_id}, {id}, {prediction}')
        # Gửi dữ liệu về Kafka
        producer.send(response_topic, key=str(correlation_id).encode('utf-8'), value=result)

# Sử dụng foreachBatch để áp dụng deep_learning_model trên mỗi batch
query = df.writeStream \
    .outputMode("append") \
    .foreachBatch(deep_learning_model) \
    .start()

# Chờ cho query kết thúc
query.awaitTermination()