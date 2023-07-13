import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

import json
import util as u
import dataset as dts
from configs.price_config import price_cf as config
from model import PredictPriceLSTM 


class Predict_Stock_Price:
    
    def __init__(self):
        self.model = PredictPriceLSTM(input_size=config["model"]["input_size"], 
                                      hidden_layer_size=config["model"]["lstm_size"], 
                                      num_layers=config["model"]["num_lstm_layers"], 
                                      output_size=1, 
                                      dropout=config["model"]["dropout"])
        self.model = self.model.to(config["training"]["device"])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)
        self.scaler = dts.MyMinMaxScaler()
        self.newsymbol = self.readjson('configs/symbolconfig.json')
        config['alpha_vantage']['symbol'] = self.newsymbol
        # define path to model, dataset
        self.data_path = 'csv/'
        self.models_path= 'models/price_predict/'

    def readjson(self, path):
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        # match data['symbol']:
        #     case 'APPLE':
        #         symbol = 'AAPL'
        #     case 'AMAZON':
        #         symbol = 'AMZN'
        #     case 'MICROSOFT':
        #         symbol = 'MSFT'
        #     case 'GOOGLE':
        #         symbol = 'GOOGL'
        #     case 'TESLA':
        #         symbol = 'TSLA'
        symbol=''
        if data['symbol'] == 'AAPL':
            symbol = 'AAPL'
        elif data['symbol'] == 'AMZN':
            symbol = 'AMZN'
        elif data['symbol'] == 'GOOGL':
            symbol = 'GOOGL'
        elif data['symbol'] == 'MSFT':
            symbol = 'MSFT'
        elif data['symbol'] == 'TSLA':
            symbol = 'TSLA'
        return symbol
    
    def get_data_df(self, new_data=True):
        stock_key = config['alpha_vantage']['symbol']
        file_name = str(stock_key) + '.csv'
        if new_data == True:
            df = u.download_stock_csv_file(self.data_path, file_name, symbol=stock_key, window_size=14)
        elif not u.file_exist(self.data_path, file_name) and (new_data == False):
            df = u.download_stock_csv_file(self.data_path, file_name, symbol=stock_key, window_size=14)   
        
        df = pd.read_csv(self.data_path + file_name)
        start_day = pd.to_datetime('2018-01-01')
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.loc[df['date'] > '2018-01-01']

        
        data_date = df['date'].tolist()
        data_close_price = df['adjusted close'].values
        num_data_points = len(df)
        return data_date, data_close_price, num_data_points
    
    def prepare_data_x(self, x, window_size):
        n_row = x.shape[0] - window_size + 1
        output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides = (x.strides[0], x.strides[0]))
        return output[:-1], output[-1]

    def prepare_data_y(self, x, window_size):
        output = x[window_size:]
        return output

    def prepare_data(self, normalized_data_close_price, config):
        data_x, data_x_unseen = self.prepare_data_x(normalized_data_close_price, 
                                                    window_size=config["data"]["window_size"])
        data_y = self.prepare_data_y(normalized_data_close_price, 
                                     window_size=config["data"]["window_size"])

        # split dataset

        split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
        # from 0 - 80%
        data_x_train = data_x[:split_index]
        # from 80% - 100%
        data_x_val = data_x[split_index:]
        # from 0 - 80%
        data_y_train = data_y[:split_index]
        # from 80% - 100%
        data_y_val = data_y[split_index:]
        
        return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen
    
    def run_epoch(self, dataloader, is_training=True):
        epoch_loss = 0

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        for idx, (x, y) in enumerate(dataloader):
            if is_training:
                self.optimizer.zero_grad()

            batchsize = x.shape[0]

            x = x.to(config["training"]["device"])
            y = y.to(config["training"]["device"])

            out = self.model(x)
            loss = self.criterion(out.contiguous(), y.contiguous())

            if is_training:
                loss.backward()
                self.optimizer.step()

            epoch_loss += (loss.detach().item() / batchsize)

        lr = self.scheduler.get_last_lr()[0]

        return epoch_loss, lr
    
    def training_model(self, dataset_train, dataset_val, is_training=True):
        train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
        val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)
        
        for epoch in range(config["training"]["num_epoch"]):
            loss_train, lr_train = self.run_epoch(train_dataloader, is_training=is_training)
            loss_val, lr_val = self.run_epoch(val_dataloader)
            self.scheduler.step()
            
            print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
                    .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))

        # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

        train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
        val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

        self.model.eval()
                
        model_name = config['alpha_vantage']['symbol'] + '_price.pth'
        torch.save(self.model.state_dict(), self.models_path + model_name)
        
    def predict_model(self, data_x_unseen):
        self.model.eval()
        #print(self.model.eval())
        x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
        prediction = self.model(x)
        prediction = prediction.cpu().detach().numpy()
        prediction = self.scaler.inverse_transform(prediction)[0]
        return prediction
    
    def run_model(self):
        data_date, data_close_price, num_data_points = self.get_data_df(new_data=False)
        normalized_data_close_price = self.scaler.fit_transform(data_close_price)
        split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen = self.prepare_data(normalized_data_close_price, config)
        dataset_train = dts.PredictPrice_TimeSeriesDataset(data_x_train, data_y_train)
        dataset_val = dts.PredictPrice_TimeSeriesDataset(data_x_val, data_y_val)
        if config['training']['is_training'] == True:
            self.training_model(dataset_train=dataset_train, dataset_val=dataset_val, is_training=False)
            model_name = config['alpha_vantage']['symbol'] + '_price.pth'
            self.model.load_state_dict(torch.load(self.models_path + model_name))
            predict_next_day = self.predict_model(data_x_unseen)
        else:
            model_name = config['alpha_vantage']['symbol'] + '_price.pth'
            self.model.load_state_dict(torch.load(self.models_path + model_name))
            predict_next_day = self.predict_model(data_x_unseen)
        return predict_next_day
    
    
if __name__ == "__main__":
    
    pred = Predict_Stock_Price()

    
    # cho configValue = giá trị đọc file price_config và gắn giá trị symbol
    # config = configValue
    price = pred.run_model()
    print(price)

    # 'asdasdak'
    # config[][] = ''

    
        
        