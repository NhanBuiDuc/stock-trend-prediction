import torch.nn as nn
from configs.config import config as cf
import torch
import math
import benchmark as bm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np


class Model:
    def __init__(self, parameters=None, name=None, num_feature=None, model_type=None, full_name=None):
        self.num_feature = num_feature
        self.model_type = model_type
        self.structure = None
        self.name = name
        self.parameters = parameters
        self.train_stop_lr = None
        self.train_stop_epoch = None
        self.state_dict = None
        self.pytorch_timeseries_model_type_dict = cf["pytorch_timeseries_model_type_dict"]
        self.tensorflow_timeseries_model_type_dict = cf["tensorflow_timeseries_model_type_dict"]
        if self.name is not None:
            self.construct_structure()

    def construct_structure(self):

        if self.model_type == self.pytorch_timeseries_model_type_dict[1]:
            parameters = self.parameters["model"]
            self.structure = Movement(self.num_feature, **parameters)

        elif self.model_type == self.pytorch_timeseries_model_type_dict[2]:
            pass
        elif self.model_type == self.pytorch_timeseries_model_type_dict[3]:
            pass
        elif self.model_type == self.pytorch_timeseries_model_type_dict[4]:
            self.parameters = self.parameters["model"]
            self.structure = LSTM(self.num_feature, **self.parameters)
        elif self.model_type == self.pytorch_timeseries_model_type_dict[5]:
            self.parameters = self.parameters["model"]
            self.structure = bm.GRU_bench_mark(self.num_feature, **self.parameters)
        elif self.model_type == self.pytorch_timeseries_model_type_dict[6]:
            self.parameters = self.parameters["model"]
            self.structure = TransformerClassifier(self.num_feature, **self.parameters)
        elif self.model_type == self.tensorflow_timeseries_model_type_dict[1]:
            self.parameters = self.parameters["model"]
            self.structure = svm_classifier(self.num_feature, **self.parameters)
        elif self.model_type == self.tensorflow_timeseries_model_type_dict[2]:
            self.parameters = self.parameters["model"]
            self.structure = rf_classifier(self.num_feature, **self.parameters)
        elif self.model_type == self.tensorflow_timeseries_model_type_dict[3]:
            self.parameters = self.parameters["model"]
            self.structure = xgb_classifier(self.num_feature, **self.parameters)

    def load_check_point(self, model_type, model_name):

        # self.construct_structure()
        if model_type in self.pytorch_timeseries_model_type_dict.values():
            check_point = torch.load('./models/' + model_name + ".pth")
            self = check_point["model"]
            self.structure.load_state_dict(check_point['state_dict'])
        else:
            check_point = torch.load('./models/' + model_name + ".pkl")
            self = check_point["model"]
        return self

    def predict(self, x):
        if self.model_type in self.pytorch_timeseries_model_type_dict.values():
            y = self.structure(x)
            return y
        elif self.model_type in self.tensorflow_timeseries_model_type_dict.values():
            y = self.structure.predict(x)
            return y


class LSTM(nn.Module):
    def __init__(self, num_feature, **param):
        super().__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        if self.data_mode == 0:
            self.lstm = nn.LSTM(input_size=5, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)
            self.fc1 = nn.Linear(200, 1)
        elif self.data_mode == 1:
            self.lstm = nn.LSTM(input_size=39, hidden_size=self.hidden_size, num_layers=self.num_layers)
            self.lstm = nn.LSTM(input_size=39, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)
            self.fc1 = nn.Linear(200, 1)
        elif self.data_mode == 2:
            self.lstm = nn.LSTM(input_size=807, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)
            self.fc1 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(self.drop_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_stock, x_news):
        if self.data_mode == 0:
            x_stock = x_stock[:, :, :5]
            batch = x_stock.shape[0]
            lstm_out, (h_n, c_n) = self.lstm(x_stock)  # self-attention over the input sequence
            x = h_n.reshape(batch, -1)
            x = self.fc1(x)
            x = self.sigmoid(x)
            return x
        elif self.data_mode == 1:
            batch = x_stock.shape[0]
            lstm_out, (h_n, c_n) = self.lstm(x_stock)  # self-attention over the input sequence
            x = h_n.reshape(batch, -1)
            x = self.fc1(x)
            x = self.sigmoid(x)
            return x
        elif self.data_mode == 2:
            batch = x_stock.shape[0]
            x = torch.concat([x_stock, x_news], dim=2)
            lstm_out, (h_n, c_n) = self.lstm(x)  # self-attention over the input sequence
            x = h_n.reshape(batch, -1)
            x = self.fc1(x)
            x = self.sigmoid(x)
            return x


class Movement(nn.Module):
    def __init__(self, num_feature, **param):
        super(Movement, self).__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        # self.autoencoder = Autoencoder(1, self.window_size, **self.conv1D_param)
        self.autoencoder = Autoencoder(self.num_feature, self.window_size, **self.conv1D_param)
        self.autoencoder_2 = Autoencoder(61, self.window_size, **self.conv1D_param)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.bn_1 = nn.BatchNorm1d(num_features=826)
        self.bn_2 = nn.BatchNorm1d(num_features=854)
        self.bn_3 = nn.BatchNorm1d(num_features=1134)
        self.bn_4 = nn.BatchNorm1d(num_features=1708)
        self.sigmoid = nn.Sigmoid()
        self.soft_max = nn.Softmax(dim=1)
        self.drop_out = nn.Dropout(self.drop_out)
        self.linear_1 = nn.Linear(1708, 500)
        self.linear_2 = nn.Linear(500, 100)
        self.linear_3 = nn.Linear(100, 1)

        self.linear_4 = nn.Linear(1163, 1)
        self.lstm = nn.LSTM(self.conv1D_param["output_size"], hidden_size=self.lstm_hidden_layer_size,
                            num_layers=self.lstm_num_layer,
                            batch_first=True)

        # self.lstm = nn.LSTM(self.num_feature, hidden_size=self.lstm_hidden_layer_size,
        #                     num_layers=self.lstm_num_layer,
        #                     batch_first=True)
        self.lstm_1 = nn.LSTM(59, hidden_size=self.lstm_hidden_layer_size,
                              num_layers=self.lstm_num_layer,
                              batch_first=True)
        self.lstm_2 = nn.LSTM(81, hidden_size=self.lstm_hidden_layer_size,
                              num_layers=self.lstm_num_layer,
                              batch_first=True)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm_1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x1 = x.clone()
        x_c = x.clone()
        x = self.autoencoder(x)
        x = torch.concat([x, x1], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_1(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.drop_out(x)
        x = self.relu(x)
        x1 = x.clone()
        lstm_out, (h_n, c_n) = self.lstm_1(x)
        # x = h_n.permute(1, 2, 0).reshape(batchsize, -1)
        x = h_n.permute(1, 2, 0)
        x = torch.concat([x, x1], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_2(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.relu(x)
        x1 = x.clone()
        x = self.autoencoder_2(x)
        x = torch.concat([x, x1], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_3(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.drop_out(x)
        x = self.relu(x)
        x1 = x.clone()
        lstm_out, (h_n, c_n) = self.lstm_2(x)
        # x = h_n.permute(1, 2, 0).reshape(batchsize, -1)
        x = h_n.permute(1, 2, 0)
        x = torch.concat([x, x1, x_c], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_4(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.relu(x)
        x = self.drop_out(x)
        x = x.reshape(batchsize, -1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, num_feature, window_size, output_size, **param):
        super(Autoencoder, self).__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        self.window_size = window_size
        self.main_layer = nn.ModuleList()
        self.sub_small_layer = nn.ModuleList()
        self.sub_big_layer = nn.ModuleList()
        self.output_size = output_size
        self.conv1D_type_dict = {
            1: "spatial",
            2: "temporal"
        }
        if self.conv1D_type_dict[self.type] == "spatial":
            # Spatial: 1d convolution will apply on num_feature dim, input and output chanels = window size
            # 1D feature map will slide, containing info of the different features of the same time step
            # Multiple 1D feature maps means multiple relevance between features, for each time step
            # The receptive field is calculated on num_feature(window_size, n)

            self.num_layer = \
                int((math.log((((self.num_feature - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1)) + 1)) / (
                    math.log(self.dilation_base))) + 1

            # Applying 1dconv among input lenght dim
            for i in range(self.num_layer):
                dilation = self.dilation_base ** i  # exponentially increasing dilation
                padding = (self.dilation_base ** i) * (self.kernel_size - 1)
                layer = CausalDilatedConv1d(in_channels=self.window_size,
                                            out_channels=self.window_size,
                                            kernel_size=self.kernel_size,
                                            dilation=dilation,
                                            padding=padding)
                self.main_layer.append(layer)

            for i in range(self.sub_small_num_layer):
                padding = (self.sub_small_kernel_size - 1)
                layer = CausalConv1d(self.window_size, self.window_size, kernel_size=self.sub_small_kernel_size,
                                     padding=padding)
                self.sub_small_layer.append(layer)
            for i in range(self.sub_big_num_layer):
                padding = (self.sub_big_kernel_size - 1)
                layer = CausalConv1d(self.window_size, self.window_size, kernel_size=self.sub_big_kernel_size,
                                     padding=padding)
                self.sub_big_layer.append(layer)

            # output_dim = (input_dim - kernel_size + 2 * padding) / stride + 1

            self.receptive_field_size = int((self.num_feature + (self.kernel_size - 1) * sum(
                self.dilation_base ** i for i in range(self.num_layer))) / self.max_pooling_kernel_size)

            self.small_sub_receptive_field_size = int((self.num_feature - self.sub_small_kernel_size + 2 * (
                    self.sub_small_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.big_sub_receptive_field_size = int((self.num_feature - self.sub_big_kernel_size + 2 * (
                    self.sub_big_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.receptive_field_size = int(
                self.receptive_field_size + self.small_sub_receptive_field_size + self.big_sub_receptive_field_size)
        elif self.conv1D_type_dict[self.type] == "temporal":
            # Temporal: 1d convolution will apply on window_size dim, input and output chanels = input_feature size
            # 1D feature map will slide, containing info of the same feature consecutively among the time step
            # Multiple 1D feature maps means changes day after daye of 1 feature, for each feature
            # The receptive field is calculated on num_feature(window_size, n)

            self.num_layer = int(
                (math.log((((self.window_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1)) + 1)) / (
                    math.log(self.dilation_base))) + 1

            # Applying 1dconv among input lenght dim
            for i in range(self.num_layer):
                dilation = self.dilation_base ** i  # exponentially increasing dilation
                padding = (self.dilation_base ** i) * (self.kernel_size - 1)
                layer = CausalDilatedConv1d(in_channels=self.num_feature,
                                            out_channels=self.num_feature,
                                            kernel_size=self.kernel_size,
                                            dilation=dilation,
                                            padding=padding)
                self.main_layer.append(layer)

            for i in range(self.sub_small_num_layer):
                padding = (self.sub_small_kernel_size - 1)
                layer = CausalConv1d(self.num_feature, self.num_feature, kernel_size=self.sub_small_kernel_size,
                                     padding=padding)
                self.sub_small_layer.append(layer)
            for i in range(self.sub_big_num_layer):
                padding = (self.sub_big_kernel_size - 1)
                layer = CausalConv1d(self.num_feature, self.num_feature, kernel_size=self.sub_big_kernel_size,
                                     padding=padding)
                self.sub_big_layer.append(layer)

            # output_dim = (input_dim - kernel_size + 2 * padding) / stride + 1

            self.receptive_field_size = int((self.window_size + (self.kernel_size - 1) * sum(
                self.dilation_base ** i for i in range(self.num_layer))) / self.max_pooling_kernel_size)

            self.small_sub_receptive_field_size = int((self.window_size - self.sub_small_kernel_size + 2 * (
                    self.sub_small_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.big_sub_receptive_field_size = int((self.window_size - self.sub_big_kernel_size + 2 * (
                    self.sub_big_kernel_size - 1 + 1)) / self.max_pooling_kernel_size)
            self.receptive_field_size = int(
                self.receptive_field_size + self.small_sub_receptive_field_size + self.big_sub_receptive_field_size) - 1

        self.maxpool = nn.MaxPool1d(kernel_size=self.max_pooling_kernel_size)
        self.linear_1 = nn.Linear(self.receptive_field_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch = x.shape[0]
        if self.conv1D_type_dict[self.type] == "temporal":
            x = x.permute(0, 2, 1)
        x1 = x.clone()
        x2 = x.clone()
        x3 = x.clone()
        for layer in self.main_layer:
            x1 = layer(x1)
        x1 = self.maxpool(x1)
        for layer in self.sub_small_layer:
            x2 = layer(x2)
        x2 = self.maxpool(x2)
        for layer in self.sub_big_layer:
            x3 = layer(x3)
        x3 = self.maxpool(x3)
        concat = torch.cat([x1, x2, x3], dim=2)
        out = self.linear_1(concat)
        return out


class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(CausalDilatedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, num_feature, **param):
        super().__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        self.stock_transformer = nn.Transformer(d_model=39, nhead=13,
                                                num_encoder_layers=self.num_encoder_layers,
                                                dim_feedforward=self.dim_feedforward, dropout=self.dropout)
        # Define LSTM parameters
        self.hidden_size = 128
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=39, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.news_transformer = nn.Transformer(d_model=768, nhead=64,
                                               num_encoder_layers=self.num_encoder_layers,
                                               dim_feedforward=self.dim_feedforward, dropout=self.dropout)

        if self.data_mode == 0:
            self.fc1 = nn.Linear(39 * self.window_size, 1)
        elif self.data_mode == 1:
            self.fc1 = nn.Linear(729 * self.window_size, 1)
        elif self.data_mode == 2:
            model_list = self.ensembled_model
            if model_list["svm"] != -1:
                svm = Model()
                data_mode = model_list["svm"]
                model_name = f'svm_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(data_mode)}'
                self.svm = svm.load_check_point("svm", model_name)
            if model_list["random_forest"] != -1:
                rfc = Model()
                data_mode = model_list["random_forest"]
                model_name = f'random_forest_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(data_mode)}'
                self.rfc = rfc.load_check_point("random_forest", model_name)
            if model_list["xgboost"] != -1:
                xgboost = Model()
                data_mode = model_list["xgboost"]
                model_name = f'xgboost_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(data_mode)}'
                self.xgboost = xgboost.load_check_point("xgboost", model_name)
            if model_list["lstm"] != -1:
                lstm = Model()
                data_mode = model_list["lstm"]
                model_name = f'lstm_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(data_mode)}'
                self.lstm = lstm.load_check_point("lstm", model_name).structure

            self.fc1 = nn.Linear(256, 10)
            if model_list["news"] != -1:
                self.fc2 = nn.Linear(768 * self.window_size, 1)

            count = sum(1 for value in self.ensembled_model.values() if value != -1)

            self.fc3 = nn.Linear(count, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(self.dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_stock, x_news):
        if self.data_mode == 0:
            x_stock = x_stock[:, :, :5]
            batch = x_stock.shape[0]
            x_stock = self.stock_transformer(x_stock, x_stock)  # self-attention over the input sequence
            x = x_stock.reshape(batch, -1)
            x = self.fc1(x)
            x = self.sigmoid(x)
            return x
        elif self.data_mode == 1:
            batch = x_stock.shape[0]
            x_stock = self.stock_transformer(x_stock, x_stock)  # self-attention over the input sequence
            x = x_stock.reshape(batch, -1)
            x = self.fc1(x)
            x = self.sigmoid(x)
            return x
        elif self.data_mode == 2:
            batch = x_stock.shape[0]
            outputs = []
            if self.ensembled_model["svm"] != -1:
                if self.ensembled_model["svm"] == 0:
                    svm_pred = self.svm.predict(x_stock[:, -1:, :5].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                elif self.ensembled_model["svm"] == 1:
                    svm_pred = self.svm.predict(x_stock[:, -1:, :].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                elif self.ensembled_model["svm"] == 2:
                    data = torch.cat([x_stock, x_news], dim=2)
                    svm_pred = self.svm.predict(data[:, -1:, :].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                svm_pred = self.drop_out(svm_pred)
                outputs.append(svm_pred)
            if self.ensembled_model["random_forest"] != -1:
                if self.ensembled_model["random_forest"] == 0:
                    rfc_pred = self.rfc.predict(x_stock[:, -1:, :5].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                elif self.ensembled_model["random_forest"] == 1:
                    rfc_pred = self.rfc.predict(x_stock[:, -1:, :].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                elif self.ensembled_model["random_forest"] == 2:
                    data = torch.cat([x_stock, x_news], dim=2)
                    rfc_pred = self.rfc.predict(data[:, -1:, :].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                rfc_pred = self.drop_out(rfc_pred)
                outputs.append(rfc_pred)
            if self.ensembled_model["xgboost"] != -1:
                if self.ensembled_model["xgboost"] == 0:
                    xgboost_pred = self.xgboost.predict(x_stock[:, -1:, :5].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                elif self.ensembled_model["xgboost"] == 1:
                    xgboost_pred = self.xgboost.predict(x_stock[:, -1:, :].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                elif self.ensembled_model["xgboost"] == 2:
                    data = torch.cat([x_stock, x_news], dim=2)
                    xgboost_pred = self.xgboost.predict(data[:, -1:, :].cpu().detach().numpy().reshape(batch, -1)).to(
                        "cuda").unsqueeze(1)
                xgboost_pred = self.drop_out(torch.tensor(xgboost_pred, dtype=torch.float32))
                outputs.append(xgboost_pred)

            if self.ensembled_model["lstm"] != -1:
                lstm_pred = self.lstm(x_stock, x_news)
                lstm_pred = self.drop_out(lstm_pred)
                outputs.append(lstm_pred)

            if self.ensembled_model["news"] != -1:
                x_news = x_news.view(batch, -1)
                x_news = self.fc2(x_news)
                x_news = self.drop_out(x_news)
                outputs.append(x_news)
            if outputs:
                concat = torch.cat(outputs, dim=1)
                x = concat.clone().detach().requires_grad_(True)
            else:
                x = x_news

            x = self.fc3(x)
            x = self.sigmoid(x)
            return x


class PredictPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


class svm_classifier:
    def __init__(self, num_feature, **param):
        super().__init__()
        self.__dict__.update(param)
        self.sklearn_model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, coef0=self.coef0,
                                 class_weight=self.class_weight)

    def predict(self, x):
        output = self.sklearn_model.predict(x)
        output = torch.from_numpy(output)
        return output

    def fit(self, x, y):
        self.sklearn_model.fit(x, y)
    def score(self, x, y):
        return self.sklearn_model.score(x,y)

class rf_classifier:
    def __init__(self, num_feature, **param):
        super().__init__()
        self.__dict__.update(param)
        self.sklearn_model = RandomForestClassifier(
            n_estimators=self.n_estimators,  # Number of trees in the forest
            criterion=self.criterion,  # Splitting criterion (can be 'gini' or 'entropy')
            max_depth=self.max_depth,  # Maximum depth of the tree
            min_samples_leaf=self.min_samples_leaf,  # Minimum number of samples required to be at a leaf node
        )

    def predict(self, x):
        output = self.sklearn_model.predict(x)
        output = torch.from_numpy(output)
        return output

    def fit(self, x, y):
        return self.sklearn_model.fit(x, y)
    def score(self, x, y):
        return self.sklearn_model.score(x,y)

class xgb_classifier:
    def __init__(self, num_feature, **param):
        super().__init__()
        self.__dict__.update(param)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators = self.n_estimators,  # Number of trees in the ensemble
            objective = self.objective,  # Objective function for binary classification
            max_depth=self.max_depth,  # Maximum depth of each tree
            # Learning rate (step size shrinkage)
            learning_rate=self.learning_rate,
            subsample=self.subsample,  # Subsample ratio of the training instances
            # Subsample ratio of columns when constructing each tree
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,  # L1 regularization term on weights
            reg_lambda=self.reg_lambda,  # L2 regularization term on weights
            random_state=self.random_state  # Random seed for reproducibility
        )

    def predict(self, x):
        # x = x.cpu().detach().numpy()
        output = self.xgb_model.predict(x)
        output = torch.from_numpy(output)
        return output

    def fit(self, x, y):
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        # Fit the model with the DMatrix
        self.xgb_model.fit(x, y)
    def score(self, x, y):
        return self.xgb_model.score(x,y)