import torch.nn as nn
import torch


class LSTM_bench_mark(nn.Module):
    def __init__(self, num_feature, **param):
        super(LSTM_bench_mark, self).__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(self.drop_out)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layer, 1)

        self.lstm = nn.LSTM(self.num_feature, hidden_size=self.lstm_hidden_layer_size,
                            num_layers=self.lstm_num_layer,
                            batch_first=True)
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_1(x)
        x = self.sigmoid(x)
        return x


# Define GRU model
class GRU_bench_mark(nn.Module):
    def __init__(self, num_feature, **param):
        super(GRU_bench_mark, self).__init__()
        self.__dict__.update(param)
        self.gru = nn.GRU(num_feature, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_step)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

