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
from datetime import datetime
from tqdm import tqdm
import utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# model_name = "vinai/bartpho-syllable"
# model_name = "uitnlp/visobert"
# model_name = "xlm-roberta-base"
# model_name = "vinai/phobert-base"
# model_name = "bert-base-multilingual-cased"
# model_name  = "distilbert-base-multilingual-cased"


class DatasetTok(Dataset):
    def __init__(self, data_x, data_y, label_type, tokenizer, tokenizer_name, padding_length = 514, exp_padding = 60):
        self.data_x = data_x
        self.data_y = data_y
        self.padding_length = padding_length
        self.exp_padding = exp_padding
        self.tokenizer = tokenizer
        self.label_type = label_type

        if tokenizer_name == "distilbert-base-multilingual-cased":
            self.pad_tok = '[PAD]'
            self.sep_tok = '[SEP]'
            self.sos_tok = '[CLS]'
        else:
            self.pad_tok = '<pad>'
            self.sep_tok = '</s>'
            self.sos_tok = '<s>'

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        item = {}

        x = self.data_x[index]
        x = x.replace('[CLS]', self.sos_tok)
        x = x.replace('[SEP]', self.sep_tok)
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

    def __init__(self, config):
        super(CLSModel, self).__init__()

        self.num_classes = config['model']['num_class'] # num classes
        self.model_type = config['model']['cls_classifier']
        self.config = config
        self.pretrained_model_name = config['model']['cls_pretrained']
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = config['model']['fine_tune']

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # Adding complex layers 
        if self.model_type == 'lstm':
            self.lstm1 = nn.LSTM(
                self.pretrained_model.config.hidden_size, self.config['model']['lstm_hidden_size'],
                num_layers=self.config['model']['lstm_num_layers'], batch_first=True
            )
            self.lstm2 = nn.LSTM(
                self.config['model']['lstm_hidden_size'], self.config['model']['lstm_hidden_size'],
                num_layers=self.config['model']['lstm_num_layers'], batch_first=True
            )
            self.fc1 = nn.Linear(self.config['model']['lstm_hidden_size'], 512)
        elif self.model_type == 'cnn':
            self.cnn1 = nn.Conv1d(
                1, self.config['model']['cnn_num_channels'],
                kernel_size=self.config['model']['cnn_kernel_size'], padding=self.config['model']['cnn_padding'],
            )
            self.cnn2 = nn.Conv1d(
                self.config['model']['cnn_num_channels'], int(self.config['model']['cnn_num_channels']/2),
                kernel_size=int(self.config['model']['cnn_kernel_size']/2), padding=int(self.config['model']['cnn_padding']/2),
            )
            self.fc1 = nn.Linear(int(self.config['model']['cnn_num_channels']/2), 512)

        # FC
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, input):
        # Pretrained Model
        if self.pretrained_model_name == 'distilbert-base-multilingual-cased':
            model_output = self.pretrained_model(**input).last_hidden_state.mean(dim=1) # Distil
        else:
            model_output = self.pretrained_model(**input).pooler_output # BERT

        if self.model_type == 'lstm':
            lstm_output1, (h_n1, c_n1) = self.lstm1(model_output)
            lstm_output2, (h_n2, c_n2) = self.lstm2(lstm_output1)
            complex_output = lstm_output2
        elif self.model_type == 'cnn':
            cnn_output_1 = F.relu(self.cnn1(model_output.unsqueeze(1)))
            cnn_output_2 = F.relu(self.cnn2(cnn_output_1))
            max_pool_out = F.max_pool1d(cnn_output_2, kernel_size=cnn_output_2.size(2)).squeeze(2)
            complex_output = max_pool_out

        # Linear
        fc1_output = F.relu(self.dropout(self.fc1(complex_output)))
        fc2_output = F.relu(self.dropout(self.fc2(fc1_output)))
        fc3_output = self.fc3(fc2_output)

        # Softmax
        soft_max_output = F.log_softmax(fc3_output, dim=1)

        return soft_max_output
    
class ACSAModel(nn.Module):

    def __init__(self, config):
        super(ACSAModel, self).__init__()

        self.num_aspects = config['model']['num_aspects'] # num_aspects
        self.num_aspect_classes = config['model']['num_classes_aspect'] # num_aspect_classes
        self.model_type = config['model']['acsa_classifier']
        self.config = config
        self.pretrained_model_name = config['model']['acsa_pretrained']
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = config['model']['fine_tune']

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # Adding complex layers 
        if self.model_type == 'lstm':
            self.lstm1 = nn.ModuleList([
                nn.LSTM(
                    self.pretrained_model.config.hidden_size, self.config['model']['lstm_hidden_size'],
                    num_layers=self.config['model']['lstm_num_layers'], batch_first=True
                )
                for _ in range(self.num_aspects)
            ])
            self.lstm2 = nn.ModuleList([
                nn.LSTM(
                    self.config['model']['lstm_hidden_size'], self.config['model']['lstm_hidden_size'],
                    num_layers=self.config['model']['lstm_num_layers'], batch_first=True
                )
                for _ in range(self.num_aspects)
            ])
            size_fc_1 = self.config['model']['lstm_hidden_size']

        elif self.model_type == 'cnn':
            self.cnn1 = nn.ModuleList([
                nn.Conv1d(
                    1, self.config['model']['cnn_num_channels'],
                    kernel_size=self.config['model']['cnn_kernel_size'], padding=self.config['model']['cnn_padding'],
                )
                for _ in range(self.num_aspects)
            ])
            self.cnn2 = nn.ModuleList([
                nn.Conv1d(
                    self.config['model']['cnn_num_channels'], int(self.config['model']['cnn_num_channels']/2),
                    kernel_size=int(self.config['model']['cnn_kernel_size']/2), padding=int(self.config['model']['cnn_padding']/2),
                )
                for _ in range(self.num_aspects)
            ])
            size_fc_1 = int(self.config['model']['cnn_num_channels']/2)
            
        # FCs
        self.fc_layers_1 = nn.ModuleList([
            nn.Linear(size_fc_1, 512)
            for _ in range(self.num_aspects)
        ])
        self.fc_layers_2 = nn.ModuleList([
            nn.Linear(512, self.num_aspect_classes)
            for _ in range(self.num_aspects)
        ])

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=0.4)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # Pretrained Model
        if self.pretrained_model_name == 'distilbert-base-multilingual-cased':
            model_output = self.pretrained_model(**input).last_hidden_state.mean(dim=1) # Distil
        else:
            model_output = self.pretrained_model(**input).pooler_output # BERT

        # LSTM or CNN
        if self.model_type == 'lstm':
            complex_outputs = [
                lstm(model_output)[0] for lstm in self.lstm1
            ]
            complex_outputs = [
                lstm(inp)[0] for lstm, inp in zip(self.lstm2, complex_outputs)
            ]
        elif self.model_type == 'cnn':
            complex_outputs = [
                F.relu(cnn(model_output.unsqueeze(1))) for cnn in self.cnn1
            ]
            complex_outputs = [
                F.relu(cnn(inp)) for cnn, inp in zip(self.cnn2, complex_outputs)
            ]
            complex_outputs = [
                F.max_pool1d(com_out, kernel_size=com_out.size(2)).squeeze(2)
                for com_out in complex_outputs
            ]

        # FCs
        outputs_1 = [
            self.dropout_layer(F.relu(fc(inp))) \
            for fc, inp in zip(self.fc_layers_1, complex_outputs)
        ]
        outputs_2 = [
            self.dropout_layer(F.relu(fc(inp))) \
            for fc, inp in zip(self.fc_layers_2, outputs_1)
        ]

        # Apply Softmax to each aspect output
        aspect_outputs_softmax = [F.log_softmax(output, dim=1) for output in outputs_2]
        # aspect_outputs_softmax = torch.stack(aspect_outputs_softmax)
        # aspect_outputs_softmax = aspect_outputs_softmax.transpose(0, 1)

        return aspect_outputs_softmax
    
class model():
    def __init__(self, config, cls_version, model_path, acsa_version=0):
        super(model, self).__init__()
        self.config = config
        # Sử dụng hàm
        if config['model']['device'] == '':
            self.device = utils.get_device()
        else:
            self.device = config['model']['device']
        print(f"Device being used: {self.device}")
        self.acsa = config['model']['require_acsa']
        self.cls = config['model']['require_cls']
        # Hyperparameters
        self.training_epochs = config['model']['epoch']
        self.training_batch_size = config['model']['training_batch_size']
        self.num_classes = config['model']['num_class'] 
        self.fine_tune = config['model']['fine_tune']
        self.cls_pad_len = config['model']['cls_padding']
        self.acsa_pad_len = config['model']['acsa_padding']
        self.num_classes_aspect = config['model']['num_classes_aspect']
        self.num_aspects = config['model']['num_aspects']
        self.dropout = config['model']['dropout']
        self.model_path = model_path
        self.cls_current_version = cls_version
        self.acsa_current_version = acsa_version
        self.mongo_server = config['mongo']['server']
        self.mongo_database = config['mongo']['database']
        self.mongo_dev = config['mongo']['dev']
        self.mongo_test = config['mongo']['test']
        self.mongo_train = config['mongo']['report']
        self.mongo_cls_eval = config['mongo']['model_eval']
        self.mongo_acsa_eval = config['mongo']['acsa_eval']
        self.mongo_eval_test = config['mongo']['eval_test']
        self.criterion = nn.CrossEntropyLoss()
        
        # #Create model
        # if self.fine_tune == True:
        #     self.pretrained_model.train()
        #     self.pretrained_model.requires_grad_(True)
        # else:
        #     self.pretrained_model.requires_grad_(False)

        # Example usage:
        
        if self.cls or (self.cls == False and self.acsa == False):
            self.cls_tokenizer = AutoTokenizer.from_pretrained(config['model']['cls_pretrained'])
            print('CLS tokenizer created')
            self.cls_model = CLSModel(config = self.config).to(self.device)
            print('CLS model created')
            self.cls_optimizer = optim.Adam(self.cls_model.parameters(), lr=config['model']['cls_lr'])

            model_path = f"{self.model_path}\model_{self.cls_current_version}.pth"
            checkpoint = torch.load(model_path)
            self.cls_model.load_state_dict(checkpoint)

            print('CLS parameters loaded')

        if self.acsa:
            self.acsa_tokenizer = AutoTokenizer.from_pretrained(config['model']['acsa_pretrained'])
            print('ACSA tokenizer created')
            self.acsa_model = ACSAModel(self.config).to(self.device)
            print('ACSA model created')
            self.acsa_optimizer = optim.Adam(self.acsa_model.parameters(), lr=config['model']['acsa_lr'])
            model_path = f"{self.model_path}\model_acsa_{self.acsa_current_version}.pth"
            checkpoint = torch.load(model_path)
            self.acsa_model.load_state_dict(checkpoint)   
            print('ACSA parameters loaded')

    def preprocess(self, x, type = 'cls'):
            x = x.replace('[CLS]', '<s>')
            x = x.replace('[SEP]', '</s>')
            if type == 'cls':
                x = self.cls_tokenizer(x, return_tensors='pt', padding= 'max_length', truncation=True, max_length = self.cls_pad_len)
            elif type == 'acsa':
                x = self.acsa_tokenizer(x, return_tensors='pt', padding= 'max_length', truncation=True, max_length = self.acsa_pad_len)

            x['input_ids'] = x['input_ids'].to(self.device)
            x['attention_mask'] = x['attention_mask'].to(self.device)
            if len(x) == 3:
                x['token_type_ids'] = x['token_type_ids'].to(self.device)
            return x

    def predict(self, input):
        
        if self.acsa:
            self.acsa_model.eval()
            acsa_input = self.preprocess(input, type = 'acsa')
        self.cls_model.eval()

        if self.cls or (self.cls == False and self.acsa == False):
            cls_input = self.preprocess(input, type = 'cls')

        with torch.no_grad():
            try:
                outputs = self.cls_model(cls_input)
                _,cls_pred = torch.max(outputs, dim=1)
            except:
                cls_pred = -1

            try:
                outputs = self.acsa_model(acsa_input).to(self.device)
                acsa_pred = [torch.argmax(aspect_output).item() for aspect_output in outputs[0]]
            except: 
                acsa_pred = [[-1,-1,-1,-1]]

            pred = [cls_pred.item()] + [acsa_pred]
        return pred
    
    def getDataloader(self, data, task='cls', batch = ''):
        # Bart 600
        # Viso 400
        # Distiled, XLM 500
        # Phobert 200

        if batch == 'full':
            batch_size = len(data) // 10
        else:
            batch_size = self.training_batch_size

        # Tạo đối tượng dataset
        if task == 'cls':
            train_dataset = DatasetTok(np.array(data['input_data']), data[['prediction', 'acsa']], ['cls', 'acsa'], padding_length = self.cls_pad_len, tokenizer=self.cls_tokenizer, tokenizer_name=self.config['model']['cls_pretrained'], exp_padding = 50)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        elif task == 'acsa':
            # Tạo đối tượng dataset
            train_dataset = DatasetTok(np.array(data['input_data']), data[['prediction', 'acsa']], ['cls', 'acsa'], padding_length = self.acsa_pad_len, tokenizer=self.acsa_tokenizer, tokenizer_name=self.config['model']['acsa_pretrained'], exp_padding = 50)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

        return train_dataloader
    
    def training(self, traindata):
        if self.cls or (self.cls == False and self.acsa == False):
            dataloader = self.getDataloader(traindata, task = 'cls')

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

            print('CLS trained')

        if self.acsa:
            dataloader = self.getDataloader(traindata, task = 'acsa')

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
                    start_batch = datetime.now()
                    batch_correct = 0


                    inputs = batch['input'].to(self.device)
                    label = batch['acsa'].to(self.device)

                    self.acsa_optimizer.zero_grad()
                    outputs = self.acsa_model(inputs)

                    loss1 = self.criterion(outputs[0], label[:, 0, :]) # title
                    loss2 = self.criterion(outputs[1], label[:, 1, :]) # desc
                    loss3 = self.criterion(outputs[2], label[:, 2, :]) # comp
                    loss4 = self.criterion(outputs[3], label[:, 3, :]) # other
                    loss = loss1 + loss2 + loss3 + loss4

                    loss.backward()
                    self.acsa_optimizer.step()
                    outputs = torch.stack(outputs)
                    outputs = outputs.transpose(0, 1)


                    # Print epoch, batch, loss, and accuracy
                    print(f"Epoch {epoch + 1}/{self.training_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                    batch_idx += 1

                    # _,pred = torch.max(torch.stack(tuple(outputs), dim=0).reshape(-1,4,4), dim=2)
                    # _,true = torch.max(label, dim=2)
                    _, pred = torch.max(outputs, dim=2)
                    _, true = torch.max(label, dim=2)

                    batch_correct += torch.sum(pred==true).item()
                    correct += batch_correct
                    total += label.size(0)
                    print(f"Accuracy: {(100 * batch_correct / (label.size(0) * self.num_aspects)):.2f}%")
                    delay = datetime.now() - start_batch
                    print(f"Delay: {delay.total_seconds()}")
                    # torch.cuda.empty_cache()
                train_acc.append(100 * correct / total)
                train_loss.append(running_loss/total_step)
                print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / (total * self.num_aspects)):.4f}')

            print('ACSA trained')
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
            self.cls_current_version = best_model_version
            model_path = f"{self.model_path}\model_{best_model_version}.pth"
            checkpoint = torch.load(model_path)
            self.cls_model.load_state_dict(checkpoint)
            print(f'reloaded CLS model version {best_model_version}')
        
        if self.acsa:
            best_model_version = self.get_best_model(mongo=self.mongo_server, database_name=self.mongo_database,\
                                                    collection_name=self.mongo_acsa_eval).get('version')
            print (f'old ACSA: {self.acsa_current_version}, new ACSA: {best_model_version}')
            if(int(self.acsa_current_version) == int(best_model_version)):
                print('No update ACSA')
            else:
                self.acsa_current_version = best_model_version
                model_path = f"{self.model_path}\model_acsa_{best_model_version}.pth"
                checkpoint = torch.load(model_path)
                self.acsa_model.load_state_dict(checkpoint)
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

    def get_test(self):
        # Kết nối tới MongoDB
        client = MongoClient(self.mongo_server)
        
        # Chọn database và collection
        db = client[self.mongo_database]
        collection = db[self.mongo_test]
        
        # Lấy toàn bộ dữ liệu từ collection
        test = list(collection.find())
        
        # Đóng kết nối
        client.close()
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(test)
        
        return df
    
    def evaluate(self, type_eval = 'dev'):
        if type_eval == 'dev':
            data = self.get_dev()
        else:
            data = self.get_test()

        if self.cls or (self.cls == False and self.acsa == False):
            print(f"Evaluating {type_eval} CLS...")
            
            dataloader = self.getDataloader(data)

            predictions = torch.tensor([]).to(self.device)
            gt = torch.tensor([]).to(self.device)

            self.cls_model.eval()

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

            acc = accuracy_score(gt, predictions)
            f1 = f1_score(gt, predictions, average='macro')
            prec = precision_score(gt, predictions, average='macro')
            recall = recall_score(gt, predictions, average='macro')
            
            print(
                f'Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}'
            )

            # Kết nối tới MongoDB
            client = MongoClient(self.mongo_server)
            
            # Chọn database và collection
            db = client[self.mongo_database]
            if type_eval == 'dev':
                collection = db[self.mongo_cls_eval]
                            # Lấy record mới nhất dựa trên _id
                version = int(collection.find_one(sort=[("_id", -1)]).get('version')) + 1

                record = {'version': version, 'task': 'cls', 'pretrained' : self.config['model']['cls_pretrained'],\
                          'classifier': self.config['model']['cls_classifier'], "acc": acc, "avg_pre": prec, \
                'avg_recall': recall, 'avg_f1': f1}
                
                collection.insert_one(record)
                # Đóng kết nối
                client.close()
                self.save_parameter(save_model=self.cls_model, dir = self.model_path, model_name=f"model_{version}")
            else:
                collection = db[self.mongo_eval_test]                
                record = {'version': self.cls_current_version, 'task': 'cls', 'pretrained' : self.config['model']['cls_pretrained'],\
                          'classifier': self.config['model']['cls_classifier'], "acc": acc, "avg_pre": prec, \
                'avg_recall': recall, 'avg_f1': f1}
                
                collection.insert_one(record)
                # Đóng kết nối
                client.close()

            

        if self.acsa:
            print(f'Evaluating {type_eval} ACSA...')
            dataloader = self.getDataloader(data, task = 'acsa')
            predictions = torch.tensor([]).to(self.device)
            gt = torch.tensor([]).to(self.device)
            self.acsa_model.eval()

            for batch in tqdm(dataloader):

                with torch.no_grad():
                    inputs = batch['input'].to(self.device)
                    label = batch['acsa'].to(self.device)

                    outputs = self.acsa_model(inputs)
                    outputs = torch.stack(outputs)
                    outputs = outputs.transpose(0, 1)
                    # _,pred = torch.max(torch.stack(outputs, dim=0).reshape(-1,self.num_aspects, self.num_classes_aspect).reshape(-1,self.num_aspects, self.num_classes_aspect), dim=2)
                    _,pred = torch.max(outputs, dim=2)

                    _,true = torch.max(label, dim=2)


                    # predictions = torch.cat((predictions.to(self.device), pred.to(self.device)), dim = 0)
                    # gt = torch.cat((gt, true.to(self.device)), dim = 0)
                    predictions = np.array(pred.tolist())
                    gt = np.array(true.tolist())
            # gt = gt.cpu().numpy().astype(int)
            # predictions = predictions.cpu().numpy().astype(int)

            # print(f"{predictions}\n {gt}")

            zero_one_loss = np.any(gt != predictions, axis=1).mean()
            hamming_loss = utils.my_hamming_loss(gt, predictions)
            emr = np.all(predictions == gt, axis=1).mean()

            acc = []
            prec, f1, recall = [], [], []

            for i in range(4):
                acc.append(accuracy_score(gt[:, i], predictions[:, i]))

                prec.append(precision_score(gt[:, i], predictions[:, i], average='macro'))
                f1.append(f1_score(gt[:, i], predictions[:, i], average='macro'))
                recall.append(recall_score(gt[:, i], predictions[:, i], average='macro'))

            acc = np.mean(acc)

            prec = np.mean(prec)
            recall = np.mean(recall)
            f1 = np.mean(f1)

            print(
                f'0/1 Loss: {zero_one_loss:.4f}, Hamming Loss: {hamming_loss:.4f}, EMR: {emr:.4f}, ' \
                + f'Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}'
            )

            # Kết nối tới MongoDB
            client = MongoClient(self.mongo_server)
            
            # Chọn database và collection
            db = client[self.mongo_database]

            if type_eval == 'dev':
                collection = db[self.mongo_acsa_eval]
                
                # Lấy record mới nhất dựa trên _id
                version = int(collection.find_one(sort=[("_id", -1)]).get('version')) + 1

                record = {'version': version, 'task': 'acsa', 'pretrained' : self.config['model']['acsa_pretrained'],\
                          'classifier': self.config['model']['acsa_classifier'], "acc": acc, "avg_pre": prec, \
                'avg_recall': recall, 'avg_f1': f1}
                
                collection.insert_one(record)
                # Đóng kết nối
                client.close()
                self.save_parameter(save_model=self.acsa_model, dir = self.model_path, model_name=f"model_acsa_{version}")
            else:
                collection = db[self.mongo_eval_test]                
                record = {'version': self.acsa_current_version, 'task': 'acsa', 'pretrained' : self.config['model']['acsa_pretrained'],\
                          'classifier': self.config['model']['acsa_classifier'], "acc": acc, "avg_pre": prec, \
                'avg_recall': recall, 'avg_f1': f1}
                
                collection.insert_one(record)
                # Đóng kết nối
                client.close()
