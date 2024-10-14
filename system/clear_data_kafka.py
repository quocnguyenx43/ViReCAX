import shutil
import os

def delete_files(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"All content deleted: {folder_path}")
    except OSError as e:
        print(f"Can't deleted all content {folder_path}: {e}")

delete_files("D:\Software\Kafka\kafka-logs")
delete_files("D:\Software\Kafka\zookeeper-data")

os.makedirs("D:\Software\Kafka\kafka-logs")
os.makedirs("D:\Software\Kafka\zookeeper-data")