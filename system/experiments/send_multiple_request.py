import asyncio
import websockets
import json
import uuid
import time

from pymongo import MongoClient


mongo_latency = 'respond_time_exp'


client = MongoClient('mongodb://localhost:27017/')
db = client['kafka_test']


SERVER_URI = "ws://localhost:8765"
NUMBER_OF_REQUESTS = 10  # Number of requests to send
TIME_SEND = 10
REQUESTS_PER_SECOND = NUMBER_OF_REQUESTS / TIME_SEND  # Maximum number of requests per second

class WebSocketRequest:
    def __init__(self, request_id):
        self.request_id = request_id
        self.correlation_id = str(uuid.uuid4())
        self.websocket = None  # Initialize websocket connection attribute

    async def connect_to_server(self):
        try:
            self.websocket = await websockets.connect(SERVER_URI, ping_interval=70, ping_timeout=10)
            print("Connected to server.")
        except Exception as e:
            print(f"Connection to server failed: {str(e)}")

    async def send_request(self):
        client_wait = time.time()
        while(self.websocket is None or self.websocket.closed):
            print("No connection to server. Attempting to reconnect...")
            await self.connect_to_server()

        try:
            message = {
                'type': 'request',
                'id': self.request_id,
                'correlation_id': self.correlation_id
            }

            await self.websocket.send(json.dumps(message))
            print(f"Sent: {json.dumps(message)}")

            while True:
                response = await self.websocket.recv()
                print(f"Received: {response}")
                if response:
                    client_wait = time.time() - client_wait
                    record = {'correlation_id': self.correlation_id, 'id': self.request_id, 'latency': client_wait}
                    rec_id1 = db[mongo_latency].insert_one(record)
                    break  # Exit the loop when response is received


        except websockets.exceptions.ConnectionClosedError:
            print("Connection closed unexpectedly, attempting to reconnect...")
            if self.websocket:
                await self.websocket.close()
            await self.connect_to_server()  # Retry connecting to server
            await self.send_request()  # Retry sending the request
        except Exception as e:
            print(f"Error occurred: {str(e)}")

async def run_test():
    start_time = time.time()
    tasks = []
    total_time = time.time()
    # Create WebSocketRequest objects
    requests = [WebSocketRequest(69353673) for i in range(NUMBER_OF_REQUESTS)]
    time_f = time.time()
    # Schedule sending requests at REQUESTS_PER_SECOND rate
    for request in requests:
        # await request.connect_to_server()  # Ensure each request has a connection before sending
        task = asyncio.create_task(request.send_request())
        tasks.append(task)
        time_now = time.time()
        # print(time_now - time_f)
        time_f = time_now
        # Wait between each request
        await asyncio.sleep(1 / REQUESTS_PER_SECOND)

    await asyncio.gather(*tasks)
    total_time = time.time() - total_time
    print(total_time)

if __name__ == "__main__":
    asyncio.run(run_test())
