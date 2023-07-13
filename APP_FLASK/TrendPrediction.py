from model import Model
import util as u
from datetime import datetime, timedelta
import NLP.util as nlp_u
from APP_FLASK import util
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
import json
from news_api import download_news_with_api
import pandas as pd

class Predictor:
    def __init__(self):
        self.data_folder = f"./csv/"
        self.stock_list = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
        # self.stock_list = ["AAPL", "AMZN"]
        self.window_size_list = [3, 7, 14]
        self.output_size_list = [3, 7, 14]
        self.scaler = MinMaxScaler(feature_range=(-100, 100))
        self.path_to_des = './APP_WEB/static/file/prediction.json'
        self.config_dict = {
            "AAPL": {
                "svm": [[7,1], [7,1], [14,1]],
                "random_forest": [[7,1], [7,2], [14,0]],
                "xgboost": [[7,1], [7,1], [14,1]],
                "lstm": [[7,0], [7,0], [14,2]],
                "ensembler": [[7,2], [7,2], [14,2]],
            },
            "AMZN": {
                "svm": [[7,2], [14,1], [7,2]],
                "random_forest": [[7,1], [14,1], [7,0]],
                "xgboost": [[7,2], [14,1], [7,1]],
                "lstm": [[7,2], [14,1], [7,2]],
                "ensembler": [[7,2], [14,2], [7,2]],
            },
            "GOOGL": {
                "svm": [[14,2], [14,1], [14,1]],
                "random_forest": [[14,1], [14,2], [14,0]],
                "xgboost": [[14,0], [14,2], [14,0]],
                "lstm": [[14,0], [14,2], [14,0]],
                "ensembler": [[14,2], [14,2], [14,2]],
            },
            "MSFT": {
                "svm": [[3,0], [7,2], [7,2]],
                "random_forest": [[3,2], [7,1], [7,1]],
                "xgboost": [[3,2], [7,1], [7,1]],
                "lstm": [[3,2], [7,2], [7,1]],
                "ensembler": [[3,2], [7,2], [7,2]],
            },
            "TSLA": {
                "svm": [[14,2], [14,1], [14,1]],
                "random_forest": [[14,1], [14,0], [14,0]],
                "xgboost": [[14,1], [14,2], [14,1]],
                "lstm": [[14,1], [14,2], [14,2]],
                "ensembler": [[14,2], [14,2], [14,2]],
            }
        }
        

        self.pytorch_timeseries_model_type_dict = [
            "ensembler",
            "lstm",
        ]
        self.tensorflow_timeseries_model_type_dict = [
            "svm",
            "random_forest",
            "xgboost"

        ]
        self.svm_data_mode_dict = {
            3: 0,
            7: 1,
            14: 2,
        }
        self.rf_data_mode_dict = {
            3: 0,
            7: 1,
            14: 2,
        }
        self.xg_data_mode_dict = {
            3: 0,
            7: 1,
            14: 2,
        }
        # Realtime, so from date is from the latest date of the CSV file, to date is current date.
        # for symbol in self.stock_list:
        #     download_news_with_api(symbol)
    
    def batch_predict(self, symbol, model_type_list, window_size, output_step):
        result = {}
        for model_type in model_type_list:
            result[model_type] = self.predict(symbol, model_type, window_size, output_step)
        print(result)
        return result
    
    # def predict(self, symbol, model_type, output_step):
    #     match model_type:
    #         case 'svm':
    #             data_mode = self.svm_data_mode_dict[output_step]
    #             model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'
    #         case 'xgboost':
    #             data_mode = self.xg_data_mode_dict[output_step]
    #             model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'
    #         case 'random_forest':
    #             data_mode = self.rf_data_mode_dict[output_step]
    #             model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'
        
    #     model = Model(model_type=model_type)
    #     model = model.load_check_point(model_type, model_name)

    #     price_data, stock_data, news_data = self.prepare_data(symbol, window_size)
    #     X = np.concatenate((price_data, stock_data, news_data), axis=1)


    #     if data_mode == 0:
    #         if model_type == "transformer":
    #             stock_tensor = torch.tensor(stock_data).float().to("cuda")
    #             news_tensor = torch.tensor(news_data).float().to("cuda")
    #             model.structure.to("cuda")
    #             model.data_mode = 0
    #             output = model.structure(stock_tensor, news_tensor)
    #         else:
    #             tensor_data = torch.tensor(price_data)
    #             output = model.predict(tensor_data)
    #     elif data_mode == 1:
    #         if model_type == "transformer":
    #             stock_tensor = torch.tensor(stock_data).float().to("cuda")
    #             news_tensor = torch.tensor(news_data).float().to("cuda")
    #             model.structure.to("cuda")
    #             model.data_mode = 1
    #             output = model.structure(stock_tensor, news_tensor)
    #         else:
    #             tensor_data = torch.tensor(stock_data)
    #             output = model.predict(tensor_data)
    #     else:
    #         if model_type == "transformer":
    #             stock_tensor = torch.tensor(stock_data).float().to("cuda")
    #             news_tensor = torch.tensor(news_data).float().to("cuda")
    #             model.structure.to("cuda")
    #             model.data_mode = 2
    #             output = model.structure(stock_tensor, news_tensor)
    #         else:
    #             tensor_data = torch.tensor(X)
    #             output = model.predict(tensor_data)

    #     threshold = 0.5
    #     converted_output = torch.where(output >= threshold, torch.tensor(1), torch.tensor(0))
        
    #     if torch.all(converted_output == 1):
    #         output_json = {
    #             f'{model_type}_{symbol}_w{window_size}_o{output_step}': "UP"
    #         }
    #     elif torch.all(converted_output == 0):
    #         output_json = {
    #             f'{model_type}_{symbol}_w{window_size}_o{output_step}': "DOWN"
    #         }     
    #     return output_json

    def predict_1(self, day, symbol, model_type, output_step):

        if output_step == 3:
            data_mode = self.config_dict[symbol][model_type][0][1]
            window_size = self.config_dict[symbol][model_type][0][0]
        elif output_step == 7:
            data_mode = self.config_dict[symbol][model_type][1][1]
            window_size = self.config_dict[symbol][model_type][1][0]
        elif output_step == 14:
            data_mode = self.config_dict[symbol][model_type][2][1]
            window_size = self.config_dict[symbol][model_type][2][0]
        model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'


        price_data, stock_data, news_data = self.prepare_data(day, symbol, window_size, output_step)
        X = np.concatenate((stock_data, news_data), axis=2)
        model = Model(model_type=model_type)
        model = model.load_check_point(model_type, model_name)
        if data_mode == 0:
            if model_type == "ensembler"or model_type == "lstm":
                stock_tensor = torch.tensor(stock_data).float().to("cuda")
                news_tensor = torch.tensor(news_data).float().to("cuda")
                model.structure.to("cuda")
                model.data_mode = 0
                output = model.structure(stock_tensor, news_tensor)
            else:
                tensor_data = torch.tensor(price_data)[:, -1, :]
                output = model.predict(tensor_data)
        elif data_mode == 1:
            if model_type == "ensembler" or model_type == "lstm":
                stock_tensor = torch.tensor(stock_data).float().to("cuda")
                news_tensor = torch.tensor(news_data).float().to("cuda")
                model.structure.to("cuda")
                model.data_mode = 1
                output = model.structure(stock_tensor, news_tensor)
            else:
                tensor_data = torch.tensor(stock_data)[:, -1, :]
                output = model.predict(tensor_data)
        else:
            if model_type == "ensembler" or model_type == "lstm":
                stock_tensor = torch.tensor(stock_data).float().to("cuda")
                news_tensor = torch.tensor(news_data).float().to("cuda")
                model.structure.to("cuda")
                model.data_mode = 2
                output = model.structure(stock_tensor, news_tensor)
            else:
                tensor_data = torch.tensor(X)[:, -1, :]
                output = model.predict(tensor_data)

        # threshold = 0.5
        # converted_output = torch.where(output >= threshold, torch.tensor(1), torch.tensor(0))
        output = output.to("cuda")
        if torch.is_tensor(output):
            # Output is a tensor
            threshold = torch.tensor(0.5, device=output.device)
            converted_output = torch.where(output >= threshold, torch.tensor(1).to("cuda"), torch.tensor(0).to("cuda"))
        else:
            # Output is a scalar
            threshold = 0.5
            if output >= threshold:
                converted_output = torch.tensor(1).to("cuda")
            else:
                converted_output = torch.tensor(0).to("cuda")
        '''
        format data
        '''
        if model_type == "svm":
            model_print_name = 'svm'
        elif model_type == "random_forest":
            model_print_name = 'random'
        elif model_type == "xgboost":
            model_print_name = 'xgboost'
        elif model_type == "lstm":
            model_print_name = 'lstm'
        elif model_type == "ensembler":
            model_print_name = 'ensembler'     
        if torch.all(converted_output == 1):
            output_json = 'UP'
        elif torch.all(converted_output == 0):
            output_json = 'DOWN'    
        return output_json


    def prepare_data(self, end, symbol, window_size, output_step):
        
        # Convert the end date string to a datetime object
        end_date = datetime.strptime(end, "%Y-%m-%d")
        # Calculate the start date by subtracting 30 days from the end date
        start_date = end_date - timedelta(days=window_size * 5)
        start = start_date.strftime("%Y-%m-%d")
        stock_df = u.prepare_stock_dataframe(symbol, window_size, start, end, new_data=False)
        
        # Filter the DataFrame based on the date range
        stock_df = stock_df.loc[start_date:end_date]
        stock_data = stock_df.values[-window_size:]
        price_data = stock_data[:, :5]
        #nlp_u.update_news_url(symbol)
        news_data = util.prepare_news_data(stock_df, symbol, window_size, start_date, end_date, output_step, 5, 500, False)
        news_data = news_data[0]
        news_data = news_data[-1, :, :]
        news_data = np.expand_dims(news_data, axis=0)
        stock_data = self.scaler.fit_transform(stock_data)
        stock_data = np.expand_dims(stock_data, axis=0)
        price_data = self.scaler.fit_transform(price_data)
        price_data = np.expand_dims(price_data, axis=0)
        return price_data, stock_data, news_data

    def fetch_prediction(self):
        df = u.prepare_stock_dataframe("AAPL", 3, "2022-07-01", datetime.now().strftime("%Y-%m-%d"), new_data=False)
        df = df.reset_index().rename(columns={'index': 'date'})
        date_column = df['date']
        # Convert the date column to datetime objects
        date_objects = pd.to_datetime(date_column)
        # Filter the dates from 2020
        filtered_dates = date_objects[(date_objects >= pd.to_datetime("2022-07-01")) & (date_objects <= pd.to_datetime("2023-07-01"))].sort_values(ascending=False)

        # Convert the date column to a list of formatted date strings
        standard_date = filtered_dates.apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")).tolist()
        # for day in standard_date:
        #     for symbol in self.stock_list:
        #         for model_type in self.tensorflow_timeseries_model_type_dict:
        #                 for output_step in self.output_size_list:
        #                     prediction = self.predict_1(day, symbol, model_type, output_step)
        #                     close_data = df[df['date'] == day]['close'].values[0]  # Get the corresponding close data
        #                     prediction['current'] = close_data  # Add 'current' key-value pair
        #                     if pd.to_datetime(day) + pd.DateOffset(days=output_step) in df['date']:
        #                         actual = df[df['date'] == pd.to_datetime(day) + pd.DateOffset(days=output_step)]['close'].values[0]
        #                     else:
        #                         actual = ""
        #                     prediction['actual'] = actual
        #                     result.append(prediction)

        # for stock in self.stock_list:
        #     for window_size in self.window_size_list:
        #         u.prepare_stock_dataframe(stock, window_size, "2022-07-01", datetime.now().strftime("%Y-%m-%d"), new_data=True)
        result = {}
        for symbol in self.stock_list:
            result[symbol] = {}
            for day in standard_date:
                result[symbol][day] = {}
                result[symbol][day]["current"] = str(df[df['date'] == day]['close'].values[0])
                for output_step in self.output_size_list:
                    result[symbol][day][str(output_step)] = {}
                    actual = None
                    actual_day = pd.to_datetime(day) + pd.DateOffset(days=output_step)
                    while actual is None:
                        if actual_day > df['date'].max():  # Check if actual_day is out of bounds
                            actual = ""  # Set actual as empty string
                        elif actual_day.dayofweek >= 5:  # Skip Saturday and Sunday
                            actual_day += pd.DateOffset(days=1)
                        elif actual_day in df['date'].values:
                            actual = df[df['date'] == actual_day]['close'].values[0]
                        else:
                            actual_day += pd.DateOffset(days=1)
                    result[symbol][day][str(output_step)]["actual"] = str(actual)
                    for model_type in self.tensorflow_timeseries_model_type_dict:
                        prediction = self.predict_1(day, symbol, model_type, output_step)
                        # prediction = {model_type: "UP"}

                        # prediction['actual'] = actual
                        result[symbol][day][str(output_step)][model_type] = str(prediction)
                    for model_type in self.pytorch_timeseries_model_type_dict:
                        prediction = self.predict_1(day, symbol, model_type, output_step)
                        result[symbol][day][str(output_step)][model_type] = str(prediction)   
        with open(self.path_to_des, "w") as json_file:
            # Write the dictionary to the JSON file
            json.dump(result, json_file)
        
        return json_file


'''
APPLE:
RF: 3-3-0 
SVM: 3-7-1
XG: 3-3-0

MSFT:
RF: 14-3-0
SVM:14-3-0
XG:7-7-2

TSLA:
RF: 14-7-2
SVM: 7-14-2
XG: 14-3-2

GOOGL:
RF: 14-7-1
SVM: 7-7-2
XG: 14-7-1

AMZN:
RF: 14-7-2
SVM:14-7-1
XG:14-14-1

'''