from kafka import KafkaProducer, KafkaConsumer, TopicPartition
import asyncio
import preprocess
import websockets
import json
import uuid
import time
import utils
from pymongo import MongoClient

config = utils.load_config(level = 2)

request_topic = config['kafka']['request']
response_topic = config['kafka']['respond']
kafka_server = config['kafka']['server']
mongo_server = config['mongo']['server']
mongo_database = config['mongo']['database']
mongo_latency = config['mongo']['latency']
mongo_report = config['mongo']['report']
DEFAULT_PORT = config['websockets']['port']

client = MongoClient(mongo_server)
db =client[mongo_database]

def error_callback(exc):
    raise Exception('Error while sending data to kafka: {0}'.format(str(exc)))

class Server:
    clients = set()

    def __init__(self) -> None:
        self.port = 8765

    async def send(self, websocket, correlation_id, start_time):
        consumer = KafkaConsumer(
            bootstrap_servers=kafka_server,
            auto_offset_reset='latest',  # Change to latest to start from latest messages
            enable_auto_commit=True,
            group_id='my-group',
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        consumer.subscribe([response_topic])
        
        try:
            while True:
                for message in consumer:
                    # print(str(message.key))
                    if str(message.key) == correlation_id:
                        prediction = str(message.value['prediction']) + ' - ' + str(message.value['acsa'])

                        await websocket.send(f"<{correlation_id}>{message.value['id']}: {json.dumps(prediction)}")
                        end_time = time.time()
                        latency = end_time - start_time
                        print(f'Latency: {latency}')
                        record = {'correlation_id': correlation_id, 'id': message.value['id'], 'latency': latency}
                        rec_id1 = db[mongo_latency].insert_one(record)
                        return  # Exit after sending the message
        except websockets.exceptions.ConnectionClosedOK:
            print("Client disconnected")
        finally:
            consumer.close()

    async def receive(self, websocket):
        producer = KafkaProducer(
            bootstrap_servers=kafka_server, 
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        try:
            if producer.bootstrap_connected():
                print("Connected to Kafka successfully!")
            else:
                print("Failed to connect to Kafka")

            while True:
                raw_response = await websocket.recv()
                # print("bruh")
                respond = eval(raw_response)
                # print(respond)
                correlation_id = str(uuid.uuid4())  # Generate a unique correlation_id
                input_data = preprocess.get_input(preprocess.crawl_full(preprocess.API_DETAIL, preprocess.API_USER, str(respond['id'])))
                if respond['type'] == 'request':
                    message = {
                        'id': str(respond['id']),
                        'data': input_data
                    }
                    # print(message)
                    # producer.send(request_topic, key=correlation_id.encode('utf-8'), value=message).add_errback(error_callback)
                    producer.send(request_topic, key=respond['correlation_id'].encode('utf-8'), value=message).add_errback(error_callback)
                    start_time = time.time()
                    print(f"Sent to Kafka!")

                    # Create a task to send the response back to the correct client
                    send_task = asyncio.create_task(self.send(websocket, respond['correlation_id'], start_time))
                    await send_task  # Wait for the send task to complete

                if respond['type'] == 'report':
                    respond['acsa'] = str(respond['acsa'][1:])
                    respond['prediction'] = str(respond['prediction'])
                    respond.update({'data': input_data})
                    rec_id1 = db[mongo_report].insert_one(respond)

        except websockets.exceptions.ConnectionClosedOK:
            print("Client disconnected")
        finally:
            producer.close()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await self.receive(websocket)
        finally:
            self.clients.remove(websocket)

    async def communicate(self):
        async with websockets.serve(self.handler, "", DEFAULT_PORT):
            await asyncio.Future()

if __name__ == "__main__":
    print("Server's running")
    server = Server()
    asyncio.run(server.communicate())
