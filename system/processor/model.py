import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pymongo import MongoClient
from bson.objectid import ObjectId
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import utils

# model_name = "vinai/bartpho-syllable"
# model_name = "uitnlp/visobert"
# model_name = "xlm-roberta-base"
# model_name = "vinai/phobert-base"
# model_name = "bert-base-multilingual-cased"
# model_name  = "distilbert-base-multilingual-cased"


class DatasetTok(Dataset):
    def __init__(self, data_x, data_y, label_type, tokenizer, padding_length = 514, exp_padding = 60):
        self.data_x = data_x
        self.data_y = data_y
        self.padding_length = padding_length
        self.exp_padding = exp_padding
        self.tokenizer = tokenizer
        self.label_type = label_type

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        item = {}

        x = self.data_x[index]
        x = x.replace('[CLS]', '')
        x = x.replace('[SEP]', '</s>')
        x = self.tokenizer(x, return_tensors='pt', padding= 'max_length', truncation=True, max_length = self.padding_length)
        x['input_ids'] = x['input_ids'].squeeze()
        x['attention_mask'] = x['attention_mask'].squeeze()
        if len(x) == 3:
          x['token_type_ids'] = x['token_type_ids'].squeeze()
        item.update({'input': x})

        y_cls = self.data_y['prediction'][index]
        y_acsa = self.data_y['acsa'][index]

        if "acsa" in self.label_type:
          one_hot_tensor = F.one_hot(torch.tensor(eval(y_acsa)), 4)
          item.update({'acsa': one_hot_tensor.squeeze(dim = 1).float()})

        if "cls" in self.label_type:
          one_hot_label = F.one_hot(torch.tensor(int(y_cls)), 3)
          item.update({'cls':one_hot_label.float()})

        # if "explaination" in self.label_type:
        #   tokenized_y = self.tokenizer(y[6], return_tensors='pt')
        #   size_y = tokenized_y['input_ids'].size()[1]
        #   padded_y = torch.nn.functional.pad(tokenized_y, (0, self.exp_padding - size_y), value = 1)
        #   item.update({'explaination':padded_y})
        return item

class CLSModel(nn.Module):
    def __init__(self, num_heads, num_layers, model, num_classes):
        super(CLSModel, self).__init__()

        # Load pre-trained BERT model
        self.model = model
        self.num_classes = num_classes

        # Fully connected layers
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.model.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, input):
        # model embedding
        # model_output = self.model(**input).pooler_output   #Viso, XLM, Bert
        model_output = self.model(**input).last_hidden_state.mean(dim = 1)  #BART, Distiled

        fc1_output = F.relu(self.dropout(self.fc1(model_output)))
        fc2_output = F.relu(self.dropout(self.fc2(fc1_output)))

        # Softmax output for classification
        soft_max_output = F.log_softmax(self.fc3(fc2_output), dim=1)

        return soft_max_output
    
class ACSAModel(nn.Module):
    def __init__(self, num_aspects, model, num_classes_aspect, dropout_rate = 0.4):
        super(ACSAModel, self).__init__()

        # Load pre-trained model
        self.model = model
        self.num_classes_aspect = num_classes_aspect

        # Define Fully connected layers for each aspect
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.model.config.hidden_size, num_classes_aspect)
            for _ in range(num_aspects)
        ])

        # Define Dropout layer
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # model embedding
        # model_output = self.model(**input).pooler_output   #Viso, XLM, Bert
        model_output = self.model(**input).last_hidden_state.mean(dim = 1)  #BART, Distiled

        # Fully connected layers with Dropout for each aspect
        aspect_outputs = [
            self.dropout_layer(F.relu(fc(model_output)))
            for fc in self.fc_layers
        ]

        # Apply Softmax to each aspect output
        aspect_outputs_softmax = [self.softmax(output) for output in aspect_outputs]

        return aspect_outputs_softmax

class model():
    def __init__(self, config, cls_version, model_path, acsa_version=0):
        super(model, self).__init__()

        # Sử dụng hàm
        self.device = utils.get_device()
        print(f"Device being used: {self.device}")
        self.acsa = config['model']['require_acsa']
        # Hyperparameters
        self.training_epochs = config['model']['epoch']
        self.training_batch_size = config['model']['training_batch_size']
        self.num_classes = config['model']['num_class'] 
        self.fine_tune = config['model']['fine_tune']
        self.pad_len = config['model']['padding']
        self.num_classes_aspect = config['model']['num_classes_aspect']
        self.num_aspects = config['model']['num_aspects']
        self.dropout = config['model']['dropout']
        self.model_path = model_path
        self.cls_current_version = cls_version
        self.acsa_current_version = acsa_version
        self.mongo_server = config['mongo']['server']
        self.mongo_database = config['mongo']['database']
        self.mongo_dev = config['mongo']['dev']
        self.mongo_train = config['mongo']['report']
        self.mongo_cls_eval = config['mongo']['model_eval']
        self.mongo_acsa_eval = config['mongo']['acsa_eval']

        print('getting pretrained and tokenizer')
        self.pretrained_model = AutoModel.from_pretrained(config['model']['name'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

        #Create model
        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # Example usage:
        self.cls_model = CLSModel(num_classes= self.num_classes, model=self.pretrained_model, num_heads=0, num_layers=0).to(self.device)
        
        print('model created')
        
        self.criterion = nn.CrossEntropyLoss()

        self.cls_optimizer = optim.Adam(self.cls_model.parameters(), lr=0.0001)

        model_path = f"{self.model_path}\model_{self.cls_current_version}.pth"
        checkpoint = torch.load(model_path)
        self.cls_model.load_state_dict(checkpoint)

        print('parameters loaded')

        if self.acsa:
            self.acsa_model = ACSAModel(num_aspects=self.num_aspects, model = self.pretrained_model, num_classes_aspect = self.num_classes_aspect, dropout_rate = self.dropout).to(self.device)
            self.acsa_optimizer = optim.Adam(self.acsa_model.parameters(), lr=0.0001)
            model_path = f"{self.model_path}\model_acsa_{self.acsa_current_version}.pth"
            checkpoint = torch.load(model_path)
            self.acsa_model.load_state_dict(checkpoint)   

    def preprocess(self, x):
            x = x.replace('[CLS]', '')
            x = x.replace('[SEP]', '</s>')
            x = self.tokenizer(x, return_tensors='pt', padding= 'max_length', truncation=True, max_length = self.pad_len)
            x['input_ids'] = x['input_ids'].to(self.device)
            x['attention_mask'] = x['attention_mask'].to(self.device)
            if len(x) == 3:
                x['token_type_ids'] = x['token_type_ids'].to(self.device)
            return x

    def predict(self, input):
        input = self.preprocess(input)
        if self.acsa:
            self.acsa_model.eval()
        self.cls_model.eval()
        
        with torch.no_grad():
            outputs = self.cls_model(input)
            _,cls_pred = torch.max(outputs, dim=1)
            if self.acsa:
                outputs = self.acsa_model(input)
                _,acsa_pred = torch.max(torch.stack(outputs, dim=0).reshape(-1,self.num_aspects, self.num_classes_aspect).reshape(-1,self.num_aspects, self.num_classes_aspect), dim=2)

                pred = [cls_pred.item()] + acsa_pred.tolist()
            else:
                pred = [cls_pred.item()] + [[-1,-1,-1,-1]]
        return pred
    
    def getDataloader(self, data):
        # Bart 600
        # Viso 400
        # Distiled, XLM 500
        # Phobert 200

        # Tạo đối tượng dataset
        train_dataset = DatasetTok(np.array(data['input_data']), data[['prediction', 'acsa']], ['cls', 'acsa'], padding_length = self.pad_len, tokenizer=self.tokenizer, exp_padding = 50)
        train_dataloader = DataLoader(train_dataset, batch_size=self.training_batch_size, shuffle=True)
        return train_dataloader
    
    def training(self, traindata):

        dataloader = self.getDataloader(traindata)

        # val_loss = []
        # val_acc = []
        train_loss = []
        train_acc = []
        total_step = len(dataloader)

        self.cls_model.train()
        self.cls_optimizer.zero_grad()
        

        for epoch in range(self.training_epochs):
            running_loss = 0.0
            correct = 0.0
            total=0
            for batch_idx, batch in enumerate(dataloader):
                batch_correct = 0


                inputs = batch['input'].to(self.device)
                label = batch['cls'].to(self.device)

                self.cls_optimizer.zero_grad()
                outputs = self.cls_model(inputs).to(self.device)
                loss_label = self.criterion(outputs , label)

                # Print epoch, batch, loss, and accuracy
                print(f"Epoch {epoch + 1}/{self.training_epochs}, Batch {batch_idx + 1}/{len(batch)}, Loss: {loss_label.item():.4f}")
                batch_idx += 1
                loss_label.backward()
                self.cls_optimizer.step()

                running_loss += loss_label.item()
                _,pred = torch.max(outputs, dim=1)
                _,true = torch.max(label, dim=1)
                batch_correct += torch.sum(pred==true).item()
                correct += batch_correct
                total += label.size(0)
                print(f"Accuracy: {(100 * batch_correct / label.size(0)):.2f}%")

            train_acc.append(100 * correct / total)
            train_loss.append(running_loss/total_step)
            print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
    
        if self.acsa:
            dataloader = self.getDataloader(traindata)

            self.acsa_optimizer.zero_grad()
            train_loss = []
            train_acc = []
            total_step = len(dataloader)

            self.acsa_model.train()

            for epoch in range(self.training_epochs):
                running_loss = 0.0
                correct = 0.0
                total=0
                for batch_idx, batch in enumerate(dataloader):
                    batch_correct = 0


                    inputs = batch['input'].to(self.device)
                    label = batch['acsa'].to(self.device)

                    self.acsa_optimizer.zero_grad()
                    outputs = self.acsa_model(inputs)
                    print(torch.stack(outputs, dim=0).reshape(-1, self.num_aspects, self.num_classes_aspect))
                    loss_label = self.criterion(torch.stack(outputs, dim=0).reshape(-1, self.num_aspects, self.num_classes_aspect), label)

                    # Print epoch, batch, loss, and accuracy
                    print(f"Epoch {epoch + 1}/{self.training_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss_label.item():.4f}")
                    batch_idx += 1
                    loss_label.backward()
                    self.acsa_optimizer.step()

                    running_loss += loss_label.item()
                    _,pred = torch.max(torch.stack(outputs, dim=0).reshape(-1,4,4), dim=2)
                    _,true = torch.max(label, dim=2)
                    batch_correct += torch.sum(pred==true).item()
                    correct += batch_correct
                    total += label.size(0)
                    print(f"Accuracy: {(100 * batch_correct / (label.size(0) * self.num_aspects)):.2f}%")

                train_acc.append(100 * correct / total)
                train_loss.append(running_loss/total_step)
                print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / (total * self.num_aspects)):.4f}')


        print('Evaluating task...')
        self.evaluate()


    def save_parameter(self, save_model, dir,model_name):
        torch.save(save_model.state_dict(),f"{dir}\{model_name}.pth")
        print('saved model')

    def get_best_model(self, mongo, database_name, collection_name):
        # Kết nối tới MongoDB
        client = MongoClient(mongo)
        
        # Chọn database và collection
        db = client[database_name]
        collection = db[collection_name]
        
        # Truy vấn record với giá trị "avg_f1" cao nhất
        highest_avg_f1_record = collection.find_one(sort=[("avg_f1", -1)])
        
        # Đóng kết nối
        client.close()
        
        return highest_avg_f1_record

    def reload_model(self):
        best_model_version = self.get_best_model(mongo=self.mongo_server, database_name=self.mongo_database,\
                                                 collection_name=self.mongo_cls_eval).get('version')
        print (f'old CLS: {self.cls_current_version}, new CLS: {best_model_version}')
        if(int(self.cls_current_version) == int(best_model_version)):
            print('No update CLS')
        else:
            model_path = f"{self.model_path}\model_{best_model_version}.pth"
            checkpoint = torch.load(model_path)
            self.cls_model.load_state_dict(checkpoint)
            print(f'reloaded CLS model version {best_model_version}')
        
        if self.acsa:
            best_model_version = self.get_best_model(mongo=self.mongo_server, database_name=self.mongo_database,\
                                                    collection_name=self.mongo_acsa_eval).get('version')
            print (f'old ACSA: {self.cls_current_version}, new ACSA: {best_model_version}')
            if(int(self.cls_current_version) == int(best_model_version)):
                print('No update ACSA')
            else:
                model_path = f"{self.model_path}\model_acsa_{best_model_version}.pth"
                checkpoint = torch.load(model_path)
                self.cls_model.load_state_dict(checkpoint)
                print(f'reloaded ACSA model version {best_model_version}')

    def get_dev(self):
        # Kết nối tới MongoDB
        client = MongoClient(self.mongo_server)
        
        # Chọn database và collection
        db = client[self.mongo_database]
        collection = db[self.mongo_dev]
        
        # Lấy toàn bộ dữ liệu từ collection
        dev = list(collection.find())
        
        # Đóng kết nối
        client.close()
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(dev)
        
        return df
    
    def evaluate(self):

        dev = self.get_dev()
        dataloader = self.getDataloader(dev)

        predictions = torch.tensor([]).to(self.device)
        gt = torch.tensor([]).to(self.device)
        self.cls_optimizer.zero_grad()
        for batch in tqdm(dataloader):
            with torch.no_grad():
                inputs = batch['input'].to(self.device)
                label = batch['cls'].to(self.device)

                outputs = self.cls_model(inputs).to(self.device)
                _,pred = torch.max(outputs, dim=1)
                _,true = torch.max(label, dim=1)

                predictions = torch.cat((predictions, pred), dim = 0)
                gt = torch.cat((gt, true), dim = 0)

        gt = gt.cpu().numpy().astype(int)
        predictions = predictions.cpu().numpy().astype(int)

        # Tính precision, recall, f1-score cho từng class
        precision, recall, f1_score, _ = precision_recall_fscore_support(gt, predictions, average=None)

        # In kết quả
        for i in range(len(precision)):
            print(f"Class {i}: Precision={precision[i]}, Recall={recall[i]}, F1-Score={f1_score[i]}")

        # Tính trung bình các giá trị
        average_precision = sum(precision) / len(precision)
        average_recall = sum(recall) / len(recall)
        average_f1_score = sum(f1_score) / len(f1_score)
        correct = np.sum(predictions==gt)

        print(f"Accuracy: {correct / predictions.shape[0]}")
        print(f"Average Precision: {average_precision}")
        print(f"Average Recall: {average_recall}")
        print(f"Average F1-Score: {average_f1_score}")

        # Kết nối tới MongoDB
        client = MongoClient(self.mongo_server)
        
        # Chọn database và collection
        db = client[self.mongo_database]
        collection = db[self.mongo_cls_eval]
        
        # Lấy record mới nhất dựa trên _id
        version = int(collection.find_one(sort=[("_id", -1)]).get('version')) + 1

        record = {'version': version,'cls_0_f1': f1_score[0], "cls_1_f1": f1_score[1], 'cls_2_f1': f1_score[2], \
                  "acc": correct / predictions.shape[0], "avg_pre": average_precision, \
          'avg_recall': average_recall, 'avg_f1': average_f1_score}
        
        collection.insert_one(record)
        # Đóng kết nối
        client.close()

        self.save_parameter(save_model=self.cls_model, dir = self.model_path, model_name=f"model_{version}")

        if self.acsa:
            dataloader = self.getDataloader(dev)
            predictions = torch.tensor([]).to(self.device)
            gt = torch.tensor([]).to(self.device)
            self.acsa_model.eval()

            for batch in tqdm(dataloader):
                self.acsa_optimizer.zero_grad()
                with torch.no_grad():
                    inputs = batch['input'].to(self.device)
                    label = batch['acsa'].to(self.device)

                    outputs = self.acsa_model(inputs)
                    _,pred = torch.max(torch.stack(outputs, dim=0).reshape(-1,self.num_aspects, self.num_classes_aspect).reshape(-1,self.num_aspects, self.num_classes_aspect), dim=2)
                    _,true = torch.max(label, dim=2)

                    predictions = torch.cat((predictions.to(self.device), pred.to(self.device)), dim = 0)
                    gt = torch.cat((gt, true.to(self.device)), dim = 0)

            gt = gt.cpu().numpy().astype(int)
            predictions = predictions.cpu().numpy().astype(int)

            # print(predictions)

            # Initialize variables to store TP, FP, FN, accuracy, precision, recall, F1-score
            TP = np.zeros((self.num_aspects, self.num_classes_aspect), dtype=int)
            FP = np.zeros((self.num_aspects, self.num_classes_aspect), dtype=int)
            FN = np.zeros((self.num_aspects, self.num_classes_aspect), dtype=int)
            accuracy_per_aspect = np.zeros(self.num_aspects)
            precision_per_aspect = np.zeros(self.num_aspects)
            recall_per_aspect = np.zeros(self.num_aspects)
            f1_per_aspect = np.zeros(self.num_aspects)

            # Calculate TP, FP, FN for each class and aspect
            for aspect in range(self.num_aspects):
                for sentiment_class in range(self.num_classes_aspect):
                    true_positive_mask = (gt[:, aspect] == sentiment_class) & (predictions[:, aspect] == sentiment_class)
                    false_positive_mask = (gt[:, aspect] != sentiment_class) & (predictions[:, aspect] == sentiment_class)
                    false_negative_mask = (gt[:, aspect] == sentiment_class) & (predictions[:, aspect] != sentiment_class)

                    TP[aspect, sentiment_class] = np.sum(true_positive_mask)
                    FP[aspect, sentiment_class] = np.sum(false_positive_mask)
                    FN[aspect, sentiment_class] = np.sum(false_negative_mask)

                    # Create a DataFrame
                    columns = ['Aspect', 'Class', 'Accuracy', 'Precision', 'Recall', 'F1-score']
                    # df = pd.DataFrame(columns=columns)
                    df = []
                    # Initialize a list to store results
                    results = []
                    overall = []

                    avg_acc = 0
                    avg_precision = 0
                    avg_recall = 0
                    avg_f1 = 0

        # Fill the DataFrame with values
            for aspect in range(self.num_aspects):
                sum_class_acc, sum_class_precision, sum_class_recall, sum_class_f1 = 0, 0, 0, 0

                for sentiment_class in range(self.num_classes_aspect):

                    class_acc_per_aspect = self.calculate_accuracy(TP[aspect][sentiment_class], FP[aspect][sentiment_class], FN[aspect][sentiment_class])
                    class_precision_per_aspect = self.calculate_precision(TP[aspect][sentiment_class], FP[aspect][sentiment_class])
                    class_recall_per_aspect = self.calculate_recall(TP[aspect][sentiment_class], FN[aspect][sentiment_class])
                    class_f1_per_aspect = self.calculate_f1_score(TP[aspect][sentiment_class], FP[aspect][sentiment_class], FN[aspect][sentiment_class])

                    sum_class_acc += class_acc_per_aspect
                    sum_class_precision += class_precision_per_aspect
                    sum_class_recall += class_recall_per_aspect
                    sum_class_f1 += class_f1_per_aspect

                    df.append({'Aspect': aspect,
                                    'Class': sentiment_class,
                                    'Accuracy': class_acc_per_aspect,
                                    'Precision': class_precision_per_aspect,
                                    'Recall': class_recall_per_aspect,
                                    'F1-score': class_f1_per_aspect})
                results.append({
                    'Aspect': aspect,
                    'Aspect_Accuracy': sum_class_acc / self.num_classes_aspect,
                    'Aspect_Precision': sum_class_precision / self.num_classes_aspect,
                    'Aspect_Recall': sum_class_recall / self.num_classes_aspect,
                    'Aspect_F1-score': sum_class_f1 / self.num_classes_aspect
                })

                avg_acc += sum_class_acc / self.num_classes_aspect
                avg_precision += sum_class_precision / self.num_classes_aspect
                avg_recall += sum_class_recall / self.num_classes_aspect
                avg_f1 += sum_class_f1 / self.num_classes_aspect


            # Add a row for overall average metrics
            results.append({
                            'Aspect': 'Overall',
                            'Aspect_Accuracy': avg_acc / self.num_aspects,
                            'Aspect_Precision': avg_precision / self.num_aspects,
                            'Aspect_Recall': avg_recall / self.num_aspects,
                            'Aspect_F1-score': avg_f1 / self.num_aspects})

            # Create a DataFrame from the list
            df_aspects = pd.DataFrame(results)
            df_aspects = df_aspects.astype(object)

            # Kết nối tới MongoDB
            client = MongoClient(self.mongo_server)
            
            # Chọn database và collection
            db = client[self.mongo_database]
            collection = db[self.mongo_acsa_eval]
            
            # Lấy record mới nhất dựa trên _id
            version = int(collection.find_one(sort=[("_id", -1)]).get('version')) + 1

            record = {'version': version, "acc": avg_acc / self.num_aspects , "avg_pre": avg_precision / self.num_aspects, \
            'avg_recall': avg_recall / self.num_aspects, 'avg_f1': avg_f1 / self.num_aspects}
            
            collection.insert_one(record)
            # Đóng kết nối
            client.close()
            self.save_parameter(save_model=self.acsa_model, dir = self.model_path, model_name=f"model_acsa_{version}")


    def calculate_accuracy(self,tp, fp, fn):
        return round((tp.sum() + np.finfo(float).eps) / (tp.sum() + fp.sum() + fn.sum() + np.finfo(float).eps), 6)

    def calculate_precision(self,tp, fp):
        return round((tp.sum() + np.finfo(float).eps) / (tp.sum() + fp.sum() + np.finfo(float).eps), 6)

    def calculate_recall(self,tp, fn):
        return round((tp.sum() + np.finfo(float).eps) / (tp.sum() + fn.sum() + np.finfo(float).eps), 6)

    def calculate_f1_score(self, tp, fp, fn):
        precision = self.calculate_precision(tp, fp)
        recall = self.calculate_recall(tp, fn)
        return round(2 * (precision * recall) / (precision + recall + np.finfo(float).eps), 6)