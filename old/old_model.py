    
class LSTM_Classifier_1(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=2, batch_first=True)

        self.linear_2 = nn.Linear(14 , 2)
        self.tanh_2 = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)

        self.sigmoid_3 = nn.Sigmoid()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

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

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1:, :].reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.dropout_2(x)
        # x = self.sigmoid_3(x)
        x = self.softmax_3(x)

        return x
    
class LSTM_Classifier_7(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=2, batch_first=True)

        self.linear_2 = nn.Linear(28 , 2)
        self.tanh_2 = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)

        self.sigmoid_3 = nn.Sigmoid()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

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

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.dropout_2(x)
        # x = self.sigmoid_3(x)
        x = self.softmax_3(x)

        return x

class LSTM_Classifier_14(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=2, batch_first=True)

        self.linear_2 = nn.Linear(28 , 2)
        self.tanh_2 = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)

        self.sigmoid_3 = nn.Sigmoid()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

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

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.dropout_2(x)
        # x = self.sigmoid_3(x)
        x = self.softmax_3(x)

        return x
    
# class Movement_3(nn.Module):
#     def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.linear_1 = nn.Linear(14, 1)
#         self.sigmoid_1 = nn.Sigmoid()
#         self.tanh_1 = nn.Tanh()
#         self.dropout_1 = nn.Dropout(0.2)

#         self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

#         self.linear_2 = nn.Linear(196 , 3)
#         self.tanh_2 = nn.Tanh()

#         self.relu_3 = nn.ReLU()
#         self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                  nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                  nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                  nn.init.orthogonal_(param)

#     def forward(self, x):
#         batchsize = x.shape[0]

#         x = self.linear_1(x)
#         x = self.sigmoid_1(x)
#         x = self.dropout_1(x)
#         x = self.tanh_1(x)

#         lstm_out, (h_n, c_n) = self.lstm(x)
#         x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

#         x = self.linear_2(x)
#         # x = self.tanh_2(x)
#         x = x.clone()
#         x[:, :2] = self.softmax_3(x[:, :2])
#         x[:, 2:] = self.relu_3(x[:, 2:])
#         return x
    
# class Movement_7(nn.Module):
#     def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.linear_1 = nn.Linear(14, 1)
#         self.sigmoid_1 = nn.Sigmoid()
#         self.tanh_1 = nn.Tanh()
#         self.dropout_1 = nn.Dropout(0.2)

#         self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

#         self.linear_2 = nn.Linear(196 , 3)
#         self.tanh_2 = nn.Tanh()

#         self.relu_3 = nn.ReLU()
#         self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                  nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                  nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                  nn.init.orthogonal_(param)

#     def forward(self, x):
#         batchsize = x.shape[0]

#         x = self.linear_1(x)
#         x = self.sigmoid_1(x)
#         x = self.dropout_1(x)
#         x = self.tanh_1(x)

#         lstm_out, (h_n, c_n) = self.lstm(x)
#         x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

#         x = self.linear_2(x)
#         # x = self.tanh_2(x)
#         x = x.clone()
#         x[:, :2] = self.softmax_3(x[:, :2])
#         x[:, 2:] = self.relu_3(x[:, 2:])
#         return x
 
# class Movement_14(nn.Module):
#     def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.linear_1 = nn.Linear(14, 1)
#         self.sigmoid_1 = nn.Sigmoid()
#         self.tanh_1 = nn.Tanh()
#         self.dropout_1 = nn.Dropout(0.2)

#         self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

#         self.linear_2 = nn.Linear(196 , 3)
#         self.tanh_2 = nn.Tanh()

#         self.relu_3 = nn.ReLU()
#         self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                  nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                  nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                  nn.init.orthogonal_(param)

#     def forward(self, x):
#         batchsize = x.shape[0]

#         x = self.linear_1(x)
#         x = self.sigmoid_1(x)
#         x = self.dropout_1(x)
#         x = self.tanh_1(x)

#         lstm_out, (h_n, c_n) = self.lstm(x)
#         x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

#         x = self.linear_2(x)
#         # x = self.tanh_2(x)
#         x = x.clone()
#         x[:, :2] = self.softmax_3(x[:, :2])
#         x[:, 2:] = self.relu_3(x[:, 2:])
#         return x
