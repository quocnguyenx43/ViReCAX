import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


### For task 1
class SimpleCLSModel(nn.Module):

    def __init__(self, pretrained_model_name, fine_tune=False, num_classes=3):
        super(SimpleCLSModel, self).__init__()

        self.num_classes = num_classes # num classes
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = fine_tune

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # FC
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.pretrained_model.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, input):
        # Pretrained Model
        if self.pretrained_model_name == 'distilbert-base-multilingual-cased':
            model_output = self.pretrained_model(**input).last_hidden_state.mean(dim=1) # Distil
        else:
            model_output = self.pretrained_model(**input).pooler_output # BERT

        # Linear
        fc1_output = F.relu(self.dropout(self.fc1(model_output)))
        fc2_output = F.relu(self.dropout(self.fc2(fc1_output)))
        fc3_output = F.relu(self.dropout(self.fc3(fc2_output)))

        # Softmax
        soft_max_output = F.log_softmax(self.fc4(fc3_output), dim=1)

        return soft_max_output
    

### LSTM & CNN Task 1
class ComplexCLSModel(nn.Module):

    def __init__(self, model_type, params,
                 pretrained_model_name,
                 fine_tune=False, num_classes=3):
        super(ComplexCLSModel, self).__init__()

        self.num_classes = num_classes # num classes
        self.model_type = model_type
        self.params = params
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = fine_tune

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # Adding complex layers 
        if self.model_type == 'lstm':
            self.lstm1 = nn.LSTM(
                self.pretrained_model.config.hidden_size, self.params['hidden_size'],
                num_layers=self.params['num_layers'], batch_first=True
            )
            self.lstm2 = nn.LSTM(
                self.params['hidden_size'], self.params['hidden_size'],
                num_layers=self.params['num_layers'], batch_first=True
            )
            self.fc1 = nn.Linear(self.params['hidden_size'], 512)
        elif self.model_type == 'cnn':
            self.cnn1 = nn.Conv1d(
                1, self.params['num_channels'],
                kernel_size=self.params['kernel_size'], padding=self.params['padding'],
            )
            self.cnn2 = nn.Conv1d(
                self.params['num_channels'], int(self.params['num_channels']/2),
                kernel_size=int(self.params['kernel_size']/2), padding=int(self.params['padding']/2),
            )
            self.fc1 = nn.Linear(int(self.params['num_channels']/2), 512)

        # FC
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

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


### For task 2
class SimpleAspectModel(nn.Module):

    def __init__(self, pretrained_model_name, fine_tune=False, num_aspects=4, num_aspect_classes=4):
        super(SimpleAspectModel, self).__init__()

        self.num_aspects = num_aspects # num_aspects
        self.num_aspect_classes = num_aspect_classes # num_aspect_classes
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = fine_tune

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # FCs
        self.fc_layers_1 = nn.ModuleList([
            nn.Linear(self.pretrained_model.config.hidden_size, 512)
            for _ in range(num_aspects)
        ])
        self.fc_layers_2 = nn.ModuleList([
            nn.Linear(512, 256)
            for _ in range(num_aspects)
        ])
        self.fc_layers_3 = nn.ModuleList([
            nn.Linear(256, num_aspect_classes)
            for _ in range(num_aspects)
        ])

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=0.4)
        

    def forward(self, input):
        # Pretrained Model
        if self.pretrained_model_name == 'distilbert-base-multilingual-cased':
            model_output = self.pretrained_model(**input).last_hidden_state.mean(dim=1) # Distil
        else:
            model_output = self.pretrained_model(**input).pooler_output # BERT

        # Fully connected layers with Dropout for each aspect
        outputs_1 = [
            self.dropout_layer(F.relu(fc(model_output)))
            for fc in self.fc_layers_1
        ]
        outputs_2 = [
            self.dropout_layer(F.relu(fc(inp))) \
            for fc, inp in zip(self.fc_layers_2, outputs_1)
        ]
        outputs_3 = [
            self.dropout_layer(F.relu(fc(inp))) \
            for fc, inp in zip(self.fc_layers_3, outputs_2)
        ]
        # 4 x batch_size x dim

        # Apply Softmax to each aspect output
        aspect_outputs_softmax = [F.log_softmax(output, dim=1) for output in outputs_3]
        # aspect_outputs_softmax = torch.stack(aspect_outputs_softmax)
        # aspect_outputs_softmax = aspect_outputs_softmax.transpose(0, 1)

        return aspect_outputs_softmax


### LSTM & CNN Task 2
class ComplexAspectModel(nn.Module):

    def __init__(self, model_type, params,
                 pretrained_model_name,
                 fine_tune=False,
                 num_aspects=4, num_aspect_classes=4):
        super(ComplexAspectModel, self).__init__()

        self.num_aspects = num_aspects # num_aspects
        self.num_aspect_classes = num_aspect_classes # num_aspect_classes
        self.model_type = model_type
        self.params = params
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.fine_tune = fine_tune

        if self.fine_tune == True:
            self.pretrained_model.train()
            self.pretrained_model.requires_grad_(True)
        else:
            self.pretrained_model.requires_grad_(False)

        # Adding complex layers 
        if self.model_type == 'lstm':
            self.lstm1 = nn.ModuleList([
                nn.LSTM(
                    self.pretrained_model.config.hidden_size, self.params['hidden_size'],
                    num_layers=self.params['num_layers'], batch_first=True
                )
                for _ in range(num_aspects)
            ])
            self.lstm2 = nn.ModuleList([
                nn.LSTM(
                    self.params['hidden_size'], self.params['hidden_size'],
                    num_layers=self.params['num_layers'], batch_first=True
                )
                for _ in range(num_aspects)
            ])
            size_fc_1 = self.params['hidden_size']
        elif self.model_type == 'cnn':
            self.cnn1 = nn.ModuleList([
                nn.Conv1d(
                    1, self.params['num_channels'],
                    kernel_size=self.params['kernel_size'], padding=self.params['padding'],
                )
                for _ in range(num_aspects)
            ])
            self.cnn2 = nn.ModuleList([
                nn.Conv1d(
                    self.params['num_channels'], int(self.params['num_channels']/2),
                    kernel_size=int(self.params['kernel_size']/2), padding=int(self.params['padding']/2),
                )
                for _ in range(num_aspects)
            ])
            size_fc_1 = int(self.params['num_channels']/2)

        # FCs
        self.fc_layers_1 = nn.ModuleList([
            nn.Linear(size_fc_1, 512)
            for _ in range(num_aspects)
        ])
        self.fc_layers_2 = nn.ModuleList([
            nn.Linear(512, num_aspect_classes)
            for _ in range(num_aspects)
        ])

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=0.4)


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
