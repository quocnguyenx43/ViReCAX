from kazoo.client import KazooClient
from kazoo.exceptions import KazooException

def check_zookeeper_status(host):
    zk = KazooClient(hosts=host)
    try:
        zk.start(timeout=10)
        if zk.connected:
            print("Zookeeper is running")
        else:
            print("Failed to connect to Zookeeper")
    except KazooException as e:
        print(f"Error connecting to Zookeeper: {e}")
    finally:
        zk.stop()

check_zookeeper_status("127.0.0.1:9092")
