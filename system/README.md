
# Project Documentation

## 1. Dependencies

Before you start, you need to install the following software:
-   Apache Kafka (Scala 2.12): [Apache Kafka](https://kafka.apache.org/)
	-	Open kafka_folder\config\server.properties and kafka_folder\config\zookeeper.properties
	-	Change the path of log.dir and dataDir to your local directory. Example: logs-dirs=C:/kafka/kafka-logs for server.properties and dataDir=C:/kafka/zookeeper-data for zookeeper.properties.
	-	Then you should create 2 empty folders for them in Kafka's folder.
-   Apache Spark: [Downloads | Apache Spark](https://spark.apache.org/downloads.html)
	- Choose Spark release 3.0.1 (Sep 02 2020) and package type Pre-built for Apache Hadoop 2.7
	- Download and extract on any folder you want.
	- Open System Environment Variables window and select Environment Variables -> New... -> Variable name: JAVA_HOME and Variable value: C:\Program Files\Java\jdk1.8.0_271 -> OK.
	- Continue to add a new Environment Variables with Variable name: SPARK_HOME and Variable value: your_spark_folder\spark-3.0.1-bin-hadoop2.7 -> OK
	- Continue to add a new Environment Variables with Variable name: HADOOP_HOME and Variable value: your_spark_folder\spark-3.0.1-bin-hadoop2.7 -> OK
	- In User variables for PC (Admin) page, edit the "Path" variable, add %JAVA_HOME%\bin, %SPARK_HOME%\bin, %HADOOP_HOME%\bin -> OK
-   MongoDB: [MongoDB: The Developer Data Platform | MongoDB](https://www.mongodb.com/)
## 2. Project Files and Directories

Below is a description of the main directories in the project:

-   **src/**: Contains the project files and directories:
	-  **config.yaml**: Configuration file, contains setting for MongoDB, Kafka, Online Learning process, Experiments, ...
	- **extension/**: Contains the files needed to create Chrome Extension. Developed using JavaScript, HTML, and CSS according to Manifest V3.
    -   **client_handler/**: Contains files used to collect data from job postings when requested by users and send it back to the processing system. It displays predictions from the model and receives feedback from users when predictions are incorrect to improve online learning. 
    -   **processor/**: Contains the main processing files, including:
        -   __model_.py_*: Files for managing the training and evaluation process of machine learning models, focusing on two main tasks: Classification (CLS) and Aspect-based Sentiment Analysis (ACSA). These files encompass steps from data preparation, optimization to model evaluation, aiding administrators in deploying and managing machine learning models in real-world applications.
        -   **online_training_*.py**: Files that manage the online learning process.
        -   **predict.py**: File for predicting results for streaming data sent by users.
-   **data/**: Contains labeled and processed data.
-   **models/**: Contains models' parameters (.pth file).
-   **experiments/**: Contains experimental results and necessary files for experiments:
    -   **train_to_mongo.py**: Pushes CSV training data into the MongoDB collection named "train."
    -   **send_multiple_request.py**: Sends multiple requests within a specified time frame to test the system's resilience.

## 3. Running the Project
### Install dependencies
By running this line: 

    pip install -r requirements.txt

### 3.1. Setting Up Kafka

Start Zookeeper by executing the following command in CMD:

    kafka_folder/bin/windows/zookeeper-server-start.bat kafka_folder/config/zookeeper.properties

Next, start the Kafka server in a new CMD window with the command:

    kafka_folder/bin/windows/kafka-server-start.bat kafka_folder/config/server.properties

Then, create the necessary topics for the Kafka server using the command:

    kafka_folder/bin/windows/kafka-topics.bat --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic created_topic

You need to replace "created_topic" with the necessary topic names for this project, including the topics: train, test, prediction, request, report, model_update. After that, describe these topics with the command:

    kafka_folder/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic created_topic

In subsequent runs, if you encounter an error where the Kafka server cannot start, simply delete the `kafka-logs` and `zookeeper-data` directories from the main Kafka directory, and create two new empty directories with the same names. When shutting down Zookeeper and the Kafka server, use the command:

    kafka_folder/bin/windows/kafka-topics.bat --describe --bootstrap-server 127.0.0.1:9092 --topic created_topic

### 3.2. Initializing Collections for MongoDB

In MongoDB, you need to create a database named "kafka_test" and the following collections:

-   acsa_evaluation: contains results of ACSA task on online learning process.
-   current_model: current running model version.
-   dev: contains devset for evaluation.
-   evaluate_test: contains results of any task on testset.
-   model_evaluation: contains results of CLS task on online learning process.
-   offset: id of the last online learning batch (strategy 1 and 2).
-   offset_2: id of the last online learning batch (strategy 3).
-   prediction: contains predictions for clients' requests.
-   respond_time: contains respond time of each request.
-   test: contain testset for evaluation.
-   train: contains trainset for online learning.
-   user_report: contains reports of clients.

Additionally, you can rename the collections or database as per your preference in the `config.yaml` file located in the root directory of the project.
Also, you need to create the version 1 result for choosing model and online learning. You can find the examples in \models\v1_dev, with task_1_.json file for CLS task (model_evaluation collection), and task_2_.json files for ACSA task (acsa_evaluation collection).
You also need to rename models' parameters (.pth file) follow the pattern model_*version*.pth for CLS task, and acsa_*version*.pth for ACSA task. For example, you need to create the model_1.pth file corresponding to the result of version 1 in model_evaluation collection.

### 3.3. Running the Chrome Extension

Open Chrome, go to Manage Extensions or `chrome://extensions/`. Select "Load unpacked," choose the `extension` folder in the project, and click "Select Folder" to load the extension into Chrome.

### 3.4. Running the Main Processors

**You need to 

#### 3.4.1. Sending and Receiving Streaming Data

After starting Kafka and the extension, you need to run `client_handler.py` in the `client_handler` directory and `predict.py` in the `processor` directory. For `predict.py`, wait until the models are fully loaded and the protocols for Spark and Kafka are established.

Once the files are running, they will continue indefinitely until you intervene, simply by pressing CTRL + C. When both `client_handler.py` and `predict.py` are running, you can visit the website [https://muaban.net/viec-lam](https://muaban.net/viec-lam), select one of the currently listed jobs, click on the Element Data Collector extension, and press "Send Data" to send data back to the client handler. After the data is sent, editable predictions will appear. If you find the result incorrect, you can choose another result and click "Report" to send the data back to the system.

#### 3.4.2. Online Learning Process

Run the `online_learning_*.py` file to initiate the online learning process, where the numbers [1, 2, 3] in `*` represent the online learning strategies you want to use. Detailed descriptions can be found in the corresponding documentation.

The online learning process will be triggered when a predetermined amount of new data appears in the "train" collection of MongoDB. The default value is 128 data samples, with a special value of 4 for strategy 3, and the small training batches will be 32, with the number of epochs set to 5. You can change these values in the `config.yaml` file in the root directory, along with the model you wish to use, padding, and the type of task to be performed, etc.

It’s important to note that the data used for online learning will be taken from the "train" collection by default. If you wish to use the data reported by users for online learning, simply rename the "train" collection in `config.yaml` to "report."

Moreover, if you delete data from the "train" collection, and the online learning process does not occur even with sufficient batch data, you need to delete the data in the two collections: "offset" and "offset_2" to reset the reading position of the online learning process.

The results of each online learning batch will be stored in the acsa_evaluation and model_evaluation collections for the ACSA and CLS tasks, respectively.

*Attention*: If you need to do the experiment again, you need to delete and re-initialize the models files in \models and results in MongoDB collection corresponding to the task you are doing. Also, delete the offset in offset and offset_2 collection to restart the pointer.
