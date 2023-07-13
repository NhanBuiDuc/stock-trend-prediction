import torch.nn as nn
from config import config as cf
import torch
import model as m
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import math


class Assemble_1(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()

        model_name = cf["alpha_vantage"]["symbol"] + "_" + "movement_1"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_data_features = checkpoint['features']
        self.forecasting_model = m.Movement_1(
            input_size=len(self.forecasting_data_features),
            window_size=cf["model"]["movement_1"]["window_size"],
            lstm_hidden_layer_size=cf["model"]["movement_1"]["lstm_hidden_layer_size"],
            lstm_num_layers=cf["model"]["movement_1"]["lstm_num_layers"],
            output_steps=cf["model"]["movement_1"]["output_steps"],
            kernel_size=4,
            dilation_base=3
        )

        model_name = cf["alpha_vantage"]["symbol"] + "_" + "magnitude_1"
        checkpoint = torch.load('./models/' + model_name)
        self.magnitude_model_features = checkpoint['features']
        self.magnitude_model = m.Magnitude_1(
            input_size=len(self.magnitude_model_features),
            window_size=cf["model"]["magnitude_1"]["window_size"],
            lstm_hidden_layer_size=cf["model"]["magnitude_1"]["lstm_hidden_layer_size"],
            lstm_num_layers=cf["model"]["magnitude_1"]["lstm_num_layers"],
            output_steps=cf["model"]["magnitude_1"]["output_steps"],
            kernel_size=4,
            dilation_base=3
        )
        self.linear_1 = nn.Linear(2, 10)
        self.linear_2 = nn.Linear(10, 20)
        self.linear_3 = nn.Linear(20, 10)
        self.linear_4 = nn.Linear(20, 10)
        self.linear_5 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.linear_2 = nn.Linear(2, 1)
        # Adding dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]
        latest_data_point = x[:, -1, 0].unsqueeze(1)
        # Run the short-term and long-term forecasting models
        prob = self.forecasting_model(x)
        delta = self.magnitude_model(x)

        direction = (prob[:, :1] > 0.5).float()
        direction = torch.where(direction == 0, -1, 1)
        delta = (direction * delta) * latest_data_point
        # # real_value = latest_data_point + delta
        delta = torch.concat([latest_data_point, delta], dim=1)
        delta = self.linear_1(delta)
        delta = self.dropout(delta)
        delta = self.relu(delta)
        delta = self.linear_2(delta)
        delta = self.dropout(delta)
        delta = self.relu(delta)
        delta = self.linear_3(delta)
        delta = self.dropout(delta)
        delta = self.relu(delta)
        delta = self.linear_4(delta)
        delta = self.dropout(delta)
        delta = self.relu(delta)
        delta = self.linear_5(delta)
        delta = self.dropout(delta)
        delta = self.relu(delta)
        return delta


class Movement_1(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 10)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(320, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.drop_out(x)
        # x[:, :1] = self.tanh(x[:, :1])
        # x[:, 1:] = self.relu(x[:, 1:])
        x = self.sigmoid(x)
        return x


class Magnitude_1(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(320, 100)
        self.linear_2 = nn.Linear(100, 50)
        self.linear_3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.drop_out(x)
        x = self.linear_2(x)
        x = self.drop_out(x)
        x = self.tanh(x)
        x = self.linear_3(x)
        return x


class Movement_3(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 10)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(320, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.drop_out(x)
        # x[:, :1] = self.tanh(x[:, :1])
        # x[:, 1:] = self.relu(x[:, 1:])
        x = self.sigmoid(x)
        return x


class Magnitude_3(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 10)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(320, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.drop_out(x)
        # x[:, :1] = self.tanh(x[:, :1])
        # x[:, 1:] = self.relu(x[:, 1:])
        x = self.sigmoid(x)
        return x


class Movement_7(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 10)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(320, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.drop_out(x)
        # x[:, :1] = self.tanh(x[:, :1])
        # x[:, 1:] = self.relu(x[:, 1:])
        x = self.sigmoid(x)
        return x


class Magnitude_7(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 10)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(320, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.drop_out(x)
        # x[:, :1] = self.tanh(x[:, :1])
        # x[:, 1:] = self.relu(x[:, 1:])
        x = self.sigmoid(x)
        return x


class Movement_14(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 10)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(320, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.drop_out(x)
        # x[:, :1] = self.tanh(x[:, :1])
        # x[:, 1:] = self.relu(x[:, 1:])
        x = self.sigmoid(x)
        return x


class Magnitude_14(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, kernel_size,
                 dilation_base):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)

        self.lstm = nn.LSTM(2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 10)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(320, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_out = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.drop_out(x)
        # x[:, :1] = self.tanh(x[:, :1])
        # x[:, 1:] = self.relu(x[:, 1:])
        x = self.sigmoid(x)
        return x


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
        super(nn.Module, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class CausalDilatedConvNet(nn.Module):
    def __init__(self, window_size, input_channels, out_channels, kernel_size, dilation_base):
        super(CausalDilatedConvNet, self).__init__()
        self.temporal_dilation_layers = nn.ModuleList()
        self.spatial_dilation_layers = nn.ModuleList()
        self.causal_1d_layers = nn.ModuleList()
        self.causal_full_layers = nn.ModuleList()
        self.window_size = window_size
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.dilation_base = dilation_base
        self.output_size = input_channels
        self.kernel_size = kernel_size
        self.max_pooling_size = 2
        # Calculate the number of layers
        self.num_layers = int(
            (math.log((((self.window_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1)) + 1)) / (
                math.log(self.dilation_base))
        ) + 1

        self.tcd_receptive_field_size = int(((self.window_size + (self.kernel_size - 1) * sum(
            self.dilation_base ** i for i in range(self.num_layers)))) / (self.max_pooling_size))
        self.scd_receptive_field_size = int((self.input_channels + (self.kernel_size - 1) * sum(
            self.dilation_base ** i for i in range(self.num_layers))) / (self.max_pooling_size))
        # self.c1d_receptive_field_size = int(self.input_channels - 3 + 1)/ (self.max_pooling_size)
        self.c1d_receptive_field_size = int((self.input_channels - 3 + 1) / (self.max_pooling_size))
        self.f1d_receptive_field_size = int(
            (self.input_channels + (self.input_channels - 1) * (2 - 1)) / (self.max_pooling_size ** 1))
        self.receptive_field_size = int(
            self.scd_receptive_field_size + self.c1d_receptive_field_size + self.f1d_receptive_field_size)

        # Applying 1dconv among input lenght dim
        for i in range(self.num_layers):
            dilation = self.dilation_base ** i  # exponentially increasing dilation
            padding = (self.dilation_base ** i) * (kernel_size - 1)
            layer = CausalDilatedConv1d(in_channels=self.input_channels,
                                        out_channels=self.input_channels,
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        padding=padding)
            self.temporal_dilation_layers.append(layer)
        # Applying 1dconv among input channels dim
        for i in range(self.sub_small_num_layer):

            layer = CausalDilatedConv1d(in_channels=self.window_size,
                                        out_channels=self.window_size,
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        padding=padding)
            self.spatial_dilation_layers.append(layer)
        for i in range(1):
            kernel_size = 3
            padding = 0
            layer = CausalConv1d(self.window_size, self.window_size, kernel_size=kernel_size, padding=padding)
            self.causal_1d_layers.append(layer)
        for i in range(1):
            kernel_size = self.input_channels
            padding = kernel_size - 1
            layer = CausalConv1d(self.window_size, self.window_size, kernel_size=kernel_size, padding=padding)
            self.causal_full_layers.append(layer)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.output_size)
        self.maxpool = nn.MaxPool1d(kernel_size=self.max_pooling_size)
        self.linear_1 = nn.Linear(self.receptive_field_size, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch = x.shape[0]
        x1 = x.clone()
        x2 = x.clone()
        x3 = x.clone()
        for layer in self.spatial_dilation_layers:
            x1 = layer(x1)
        x1 = self.maxpool(x1)
        # for layer in self.temporal_dilation_layers:
        #     x1 = layer(x1)
        for layer in self.causal_1d_layers:
            x2 = layer(x2)
        x2 = self.maxpool(x2)
        for layer in self.causal_full_layers:
            x3 = layer(x3)
        x3 = self.maxpool(x3)
        concat = torch.cat([x1, x2, x3], dim=2)
        out = self.linear_1(concat)
        out = self.relu(out)
        return out


class Diff_1(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps

        self.kernel_size = 4
        self.dilation_base = 3

        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size,
                                                input_channels=self.input_size,
                                                out_channels=self.window_size,
                                                kernel_size=self.kernel_size,
                                                dilation_base=self.dilation_base)
        self.lstm = nn.LSTM(input_size=20, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.relu = nn.ReLU()

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
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear(x)
        x = x.clone()
        x = self.relu(x[:, 2:])
        return x


def find_divisor(a):
    # Start with a divisor of 2 (the smallest even number)
    divisor = 2
    # Keep looping until we find a divisor that works
    while True:
        # Check if a is divisible by the current divisor
        if a % divisor == 0:
            return divisor
        # If not, increment the divisor and try again
        divisor += 1
