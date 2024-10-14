import os
#Create topics
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic train")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic test")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic prediction")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic request")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic report")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic model_update")

#Describe topics
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic train")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic test")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic prediction")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic request")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic report")
os.system("D:/Software/Kafka/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic model_update")