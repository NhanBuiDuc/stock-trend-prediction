from trainer.trainer import Trainer, check_best_loss, is_early_stop

import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from model import Model
from configs.lstm_config import lstm_cf as cf
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import util as u
from dataset import PriceAndIndicatorsAndNews_TimeseriesDataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json
import numpy as np
import os
from torch.utils.data import DataLoader
import datetime
import NLP.util as nlp_u
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, StratifiedShuffleSplit
from loss import FocalLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class lstm_trainer(Trainer):
    def __init__(self, new_data=True, full_data=False, mode="train"):
        super(lstm_trainer, self).__init__()
        self.__dict__.update(self.cf)
        self.config = cf
        # self.symbol = self.cf["alpha_vantage"]["symbol"]
        self.model_type = "lstm"
        self.__dict__.update(self.config["model"])
        self.__dict__.update(self.config["training"])
        self.test_dataloader = None
        self.valid_dataloader = None
        self.train_dataloader = None
        self.full_data = full_data
        self.num_feature = None
        self.new_data = new_data
        self.model_name = f'{self.model_type}_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(self.data_mode)}'
        self.model_type_dict = self.cf["pytorch_timeseries_model_type_dict"]
        self.model = None
        self.mode = mode
        if self.mode == "train":
            self.prepare_data(self.new_data)
        else:
            self.num_feature = 807
        self.indentify()

    def indentify(self):
        self.model = Model(name=self.model_name, num_feature=self.num_feature, parameters=self.config,
                           model_type=self.model_type)

    def train(self):
        self.mode = "train"
        if "mse" in self.loss:
            criterion = nn.MSELoss()
        elif "mae" in self.loss:
            criterion = nn.L1Loss()
        elif "bce" in self.loss:
            criterion = nn.BCELoss()
        elif "focal" in self.loss:
            criterion = FocalLoss(alpha=0.5, gamma=2)
        if "adam" in self.optimizer:
            optimizer = optim.Adam(self.model.structure.parameters(), lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        elif "sgd" in self.optimizer:
            optimizer = optim.SGD(self.model.structure.parameters(), lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=self.scheduler_step_size, verbose=True)

        self.model.structure.to(self.device)
        if self.best_model:
            best_loss = sys.float_info.max
        else:
            best_loss = sys.float_info.min

        if self.early_stop:
            stop = False
        # Run train valid
        if not self.full_data:
            for epoch in range(self.num_epoch):
                loss_train, lr_train = self.run_epoch(self.model, self.train_dataloader, optimizer, criterion,
                                                      scheduler,
                                                      is_training=True, device=self.device)
                loss_val, lr_val = self.run_epoch(self.model, self.valid_dataloader, optimizer, criterion, scheduler,
                                                  is_training=False, device=self.device)
                loss_test, lr_test = self.run_epoch(self.model, self.test_dataloader, optimizer, criterion, scheduler,
                                                    is_training=False, device=self.device)
                scheduler.step(loss_val)
                if self.best_model:
                    if check_best_loss(best_loss=best_loss, loss=loss_val):
                        best_loss = loss_val
                        patient_count = 0
                        self.model.train_stop_lr = lr_train
                        self.model.train_stop_epoch = epoch

                        self.model.state_dict = self.model.structure.state_dict()
                        self.model.train_stop_epoch = epoch
                        self.model.train_stop_lr = lr_train
                        torch.save({"model": self.model,
                                    "state_dict": self.model.structure.state_dict()
                                    },
                                   "./models/" + self.model.name + ".pth")
                    else:
                        if self.early_stop:
                            stop, patient_count, best_loss, _ = is_early_stop(best_loss=best_loss,
                                                                              current_loss=loss_val,
                                                                              patient_count=patient_count,
                                                                              max_patient=self.patient)
                else:
                    self.model.state_dict = self.model.structure.state_dict()
                    self.model.train_stop_epoch = epoch
                    self.model.train_stop_lr = lr_train
                    torch.save({"model": self.model,
                                "state_dict": self.model.structure.state_dict()
                                },
                               "./models/" + self.model_name + ".pth")

                print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f}, test:{:.6f} | lr:{:.6f}'
                      .format(epoch + 1, self.num_epoch, loss_train, loss_val, loss_test, lr_train))
                if stop:
                    print("Early Stopped At Epoch: {}", epoch + 1)
                    break
        elif self.full_data:
            combined_dataset = ConcatDataset([self.train_dataloader.dataset, self.valid_dataloader.dataset])

            # Create a new data loader using the combined dataset
            combined_dataset = DataLoader(combined_dataset, batch_size=32, shuffle=True)
            for epoch in range(self.num_epoch):
                loss_train, lr_train = self.run_epoch(self.model, combined_dataset, optimizer, criterion, scheduler,
                                                      is_training=True, device=self.device)
                scheduler.step(loss_train)
                if self.best_model:
                    if check_best_loss(best_loss=best_loss, loss=loss_train):
                        best_loss = loss_train
                        patient_count = 0
                        self.model.train_stop_lr = lr_train
                        self.model.train_stop_epoch = epoch

                        self.model.state_dict = self.model.structure.state_dict()
                        self.model.train_stop_epoch = epoch
                        self.model.train_stop_lr = lr_train
                        torch.save({"model": self.model,
                                    "state_dict": self.model.structure.state_dict()
                                    },
                                   "./models/" + self.model_name + ".pth")
                    else:
                        if self.early_stop:
                            stop, patient_count, best_loss, _ = is_early_stop(best_loss=best_loss,
                                                                              current_loss=loss_train,
                                                                              patient_count=patient_count,
                                                                              max_patient=self.patient)
                else:
                    self.model.state_dict = self.model.structure.state_dict()
                    self.model.train_stop_epoch = epoch
                    self.model.train_stop_lr = lr_train
                    torch.save({"model": self.model,
                                "state_dict": self.model.structure.state_dict()
                                },
                               "./models/" + self.model_name + ".pth")

                print('Epoch[{}/{}] | loss train:{:.6f}| lr:{:.6f}'
                      .format(epoch + 1, self.num_epoch, loss_train, lr_train))

                print("patient", patient_count)
                if stop:
                    print("Early Stopped At Epoch: {}", epoch + 1)
                    break
        return self.model

    def grid_train(self, model, train_dataloader, valid_dataloader):
        self.mode = "train"
        if "mse" in self.loss:
            criterion = nn.MSELoss()
        elif "mae" in self.loss:
            criterion = nn.L1Loss()
        elif "bce" in self.loss:
            criterion = nn.BCELoss()
        elif "focal" in self.loss:
            criterion = FocalLoss(alpha=0.5, gamma=2)
        if "adam" in self.optimizer:
            optimizer = optim.Adam(model.structure.parameters(), lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        elif "sgd" in self.optimizer:
            optimizer = optim.SGD(model.structure.parameters(), lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                      patience=self.scheduler_step_size, verbose=True)

        model.structure.to(self.device)
        if self.best_model:
            best_loss = sys.float_info.max
        else:
            best_loss = sys.float_info.min

        if self.early_stop:
            stop = False
        # Run train valid
        if not self.full_data:
            for epoch in range(self.num_epoch):
                loss_train, lr_train = self.run_epoch(model, train_dataloader, optimizer, criterion,
                                                      scheduler,
                                                      is_training=True, device=self.device)
                loss_val, lr_val = self.run_epoch(model, valid_dataloader, optimizer, criterion, scheduler,
                                                  is_training=False, device=self.device)
                # loss_test, lr_test = self.run_epoch(model, test_dataloader, optimizer, criterion, scheduler,
                #                                     is_training=False, device=self.device)
                scheduler.step(loss_val)
                if self.best_model:
                    if check_best_loss(best_loss=best_loss, loss=loss_val):
                        best_loss = loss_val
                        patient_count = 0
                        model.train_stop_lr = lr_train
                        model.train_stop_epoch = epoch

                        model.state_dict = self.model.structure.state_dict()
                        model.train_stop_epoch = epoch
                        model.train_stop_lr = lr_train
                        torch.save({"model": model,
                                    "state_dict": model.structure.state_dict()
                                    },
                                   "./models/" + model.name + ".pth")
                    else:
                        if self.early_stop:
                            stop, patient_count, best_loss, _ = is_early_stop(best_loss=best_loss,
                                                                              current_loss=loss_val,
                                                                              patient_count=patient_count,
                                                                              max_patient=self.patient)
                else:
                    model.state_dict = model.structure.state_dict()
                    model.train_stop_epoch = epoch
                    model.train_stop_lr = lr_train
                    torch.save({"model": model,
                                "state_dict": model.structure.state_dict()
                                },
                               "./models/" + model.name + ".pth")

                print('Epoch[{}/{}] | loss train:{:.6f}, valid:{:.6f}| lr:{:.6f}'
                      .format(epoch + 1, self.num_epoch, loss_train, loss_val, lr_train))
                if stop:
                    print("Early Stopped At Epoch: {}", epoch + 1)
                    break
        elif self.full_data:
            combined_dataset = ConcatDataset([train_dataloader.dataset, valid_dataloader.dataset])

            # Create a new data loader using the combined dataset
            combined_dataset = DataLoader(combined_dataset, batch_size=32, shuffle=True)
            for epoch in range(self.num_epoch):
                loss_train, lr_train = self.run_epoch(model, combined_dataset, optimizer, criterion, scheduler,
                                                      is_training=True, device=self.device)
                scheduler.step(loss_train)
                if self.best_model:
                    if check_best_loss(best_loss=best_loss, loss=loss_train):
                        best_loss = loss_train
                        patient_count = 0
                        model.train_stop_lr = lr_train
                        model.train_stop_epoch = epoch

                        model.state_dict = model.structure.state_dict()
                        model.train_stop_epoch = epoch
                        model.train_stop_lr = lr_train
                        torch.save({"model": model,
                                    "state_dict": model.structure.state_dict()
                                    },
                                   "./models/" + model.name + ".pth")
                    else:
                        if self.early_stop:
                            stop, patient_count, best_loss, _ = is_early_stop(best_loss=best_loss,
                                                                              current_loss=loss_train,
                                                                              patient_count=patient_count,
                                                                              max_patient=self.patient)
                else:
                    model.state_dict = model.structure.state_dict()
                    model.train_stop_epoch = epoch
                    model.train_stop_lr = lr_train
                    torch.save({"model": model,
                                "state_dict": model.structure.state_dict()
                                },
                               "./models/" + model.name + ".pth")

                print('Epoch[{}/{}] | loss train:{:.6f}| lr:{:.6f}'
                      .format(epoch + 1, self.num_epoch, loss_train, lr_train))

                print("patient", patient_count)
                if stop:
                    print("Early Stopped At Epoch: {}", epoch + 1)
                    break
        return model

    def eval(self, model):

        train_dataloader, valid_dataloader, test_dataloader, balancedtest_datataloader = self.prepare_eval_data()
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        save_folder = "./eval/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Format the date and time as a string
        datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Open the file in write mode
        save_path = os.path.join(save_folder, self.model_name + "_eval")
        # Open the file in write mode
        with open(save_path, "a") as f:
            f.write(datetime_str)
            f.write(json.dumps(self.cf, indent=4))
            f.write("\n")
            f.write("Epoch:")
            f.write(json.dumps(model.train_stop_epoch, indent=4))
            f.write("\n")
            f.write("Learning rate:")
            f.write(json.dumps(model.train_stop_epoch, indent=4))
            f.write("\n")
            f.write(json.dumps(self.config, indent=4))
            f.write("\n")

        model.structure.to(self.device)
        for i in range(0, 4, 1):
            if i == 0:
                torch.cuda.empty_cache()
                dataloader = train_dataloader
                print_string = "Train evaluate " + self.model_name
            if i == 1:
                torch.cuda.empty_cache()
                dataloader = valid_dataloader
                print_string = "Valid evaluate " + self.model_name
            elif i == 2:
                torch.cuda.empty_cache()
                dataloader = test_dataloader
                print_string = "Test evaluate " + self.model_name
            elif i == 3:
                torch.cuda.empty_cache()
                dataloader = balancedtest_datataloader
                print_string = "Balanced Test evaluate " + self.model_name
            if "accuracy" or "precision" or "f1" in self.evaluate:
                # Create empty lists to store the true and predicted labels
                true_labels = []
                predicted_labels = []
            total_loss = 0

            target_list = torch.empty(0).to(self.device)
            output_list = torch.empty(0).to(self.device)
            # Iterate over the dataloader
            for x_stock, x_news, labels in dataloader:
                # Move inputs and labels to device
                x_stock = x_stock.to(self.device)
                x_news = x_news.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = model.structure(x_stock, x_news)
                target_list = torch.cat([target_list, labels], dim=0)
                output_list = torch.cat([output_list, outputs], dim=0)
                if "accuracy" or "precision" or "f1" in self.evaluate:
                    predicted = (outputs > 0.5).float()
                    # Append true and predicted labels to the respective lists
                    true_labels.extend(labels.cpu().detach().numpy())
                    predicted_labels.extend(predicted.cpu().detach().numpy())

            if "accuracy" or "precision" or "f1" in self.evaluate:
                # Compute classification report
                target_names = ["DOWN", "UP"]  # Add target class names here
                report = classification_report(true_labels, predicted_labels, target_names=target_names)
                # Create the saving folder if it does not exist
                save_folder = "./eval/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # Open the file in write mode
                save_path = os.path.join(save_folder, self.model_name + "_eval")
                # Open the file in write mode
                with open(save_path, "a") as f:
                    # Write the classification report to the file
                    f.write(print_string + "\n" + "Classification report:\n")
                    f.write(report)
                    # Compute confusion matrix
                    cm = confusion_matrix(true_labels, predicted_labels)
                    # Write the confusion matrix to the file
                    f.write("\nConfusion matrix:\n")
                    f.write(np.array2string(cm))
                    f.write("\n")
                    f.write("-" * 100)
                    f.write("\n")
                # Print a message to confirm that the file was written successfully
                print("Results written to " + self.model_name + "_eval.txt")

            temp_evaluate = np.array(self.evaluate)

            temp_evaluate = temp_evaluate[temp_evaluate != "accuracy"]
            temp_evaluate = temp_evaluate[temp_evaluate != "precision"]
            temp_evaluate = temp_evaluate[temp_evaluate != "f1"]
            for c in temp_evaluate:
                if "mse" in self.evaluate:
                    criterion = nn.MSELoss()
                    c_name = "MSE"
                elif "mae" in self.evaluate:
                    criterion = nn.L1Loss()
                    c_name = "MAE"
                elif "bce" in self.evaluate:
                    criterion = nn.BCELoss()
                    c_name = "BCE"

                target_list = target_list.reshape(-1)
                output_list = output_list.reshape(-1)
                loss = criterion(target_list, output_list)
                loss_str = f"{c_name} loss: {loss.item()}"

                # Create the saving folder if it does not exist
                save_folder = "./eval/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                # Open the file in append mode
                save_path = os.path.join(save_folder, self.model_name + "_eval")
                with open(save_path, "a") as f:
                    # Write the loss to the file
                    f.write(print_string + " " + loss_str + "\n")
                    f.write("-" * 100)
                    f.write("-")
                    f.write("\n")
                # Print a message to confirm that the file was written successfully
                print(f"Loss written to {save_path}.")

    def grid_eval(self, model, dataloader):
        # Get the current date and time

        model.structure.to(self.device)

        if "accuracy" or "precision" or "f1" in self.evaluate:
            # Create empty lists to store the true and predicted labels
            true_labels = []
            predicted_labels = []

        target_list = torch.empty(0).to(self.device)
        output_list = torch.empty(0).to(self.device)
        # Iterate over the dataloader
        for x_stock, x_news, labels in dataloader:
            # Move inputs and labels to device
            x_stock = x_stock.to(self.device)
            x_news = x_news.to(self.device)
            labels = labels.to(self.device)
            # Forward pass
            outputs = model.structure(x_stock, x_news)
            target_list = torch.cat([target_list, labels], dim=0)
            output_list = torch.cat([output_list, outputs], dim=0)
            if "accuracy" in self.evaluate:
                predicted = (outputs > 0.5).float()
                # Append true and predicted labels to the respective lists
                true_labels.extend(labels.cpu().detach().numpy())
                predicted_labels.extend(predicted.cpu().detach().numpy())

        if "accuracy" in self.evaluate:
            # Compute classification report
            target_names = ["DOWN", "UP"]  # Add target class names here
            accuracy = accuracy_score(true_labels, predicted_labels)

        return accuracy

    def grid_search(self):
        results = []

        best_accuracy = 0.0
        best_cases = []
        symbol_list = ["AAPL", "AMZN", "GOOGL", "MSFT","TSLA"]
        for symbol in symbol_list:
            for data_mode in self.param_grid['data_mode']:
                for window_size in self.param_grid['window_size']:
                    for output_size in self.param_grid['output_size']:
                            if data_mode == 2:
                                for string_length in self.param_grid['max_string_length']:
                                    for drop_out in self.param_grid['drop_out']:

                                                train_dataloader, valid_dataloader, test_dataloader, balance_dataloader = self.prepare_gridsearch_data(symbol, data_mode, window_size, output_size, string_length, new_data=True)
                                                num_feature = train_dataloader.dataset.X.shape[-1]

                                                model_name = f'svm_{symbol}_w{window_size}_o{output_size}_d{str(data_mode)}'
                                                print(model_name)
                                                config = self.config
                                                model_config = {
                                                        "num_layers": 10,
                                                        "hidden_size": 20,
                                                        "drop_out": drop_out,
                                                        "window_size": window_size,
                                                        "output_step": output_size,
                                                        "conv1D_param": {
                                                            "type": 1,
                                                            "kernel_size": 4,
                                                            "dilation_base": 3,
                                                            "max_pooling_kernel_size": 2,
                                                            "sub_small_num_layer": 1,
                                                            "sub_big_num_layer": 1,
                                                            "sub_small_kernel_size": 3,
                                                            "sub_big_kernel_size": 30,
                                                            "output_size": 20
                                                        },
                                                        "symbol": symbol,
                                                        "topk": 10,
                                                        "data_mode": data_mode,
                                                        "max_string_length": string_length,
                                                }
                                                config["model"] = model_config

                                                model = Model(name=model_name, num_feature=num_feature, parameters=config,
                                                                model_type=self.model_type,
                                                                full_name=model_name)
                                                model = self.grid_train(model, train_dataloader, valid_dataloader)
                                                train_score = self.grid_eval(model, train_dataloader)
                                                val_score = self.grid_eval(model, valid_dataloader)
                                                test_score = self.grid_eval(model, test_dataloader)
                                                balance_score = self.grid_eval(model, balance_dataloader)
                                                result = {
                                                    "symbol": symbol,
                                                    'output_size': output_size,
                                                    'window_size': window_size,
                                                    'data_mode': data_mode,
                                                    "drop_out": drop_out,
                                                    'max_string_lenght:': string_length,
                                                    'train_score': train_score,
                                                    'val_score': val_score,
                                                    "test_score": test_score,
                                                    "balance_score": balance_score  
                                                }

                                                # Append the current result to best_cases
                                                best_cases.append(result)
                            else:
                                for drop_out in self.param_grid['drop_out']:
                                            string_length = 500

                                            train_dataloader, valid_dataloader, test_dataloader, balance_dataloader = self.prepare_gridsearch_data(symbol, data_mode, window_size, output_size, string_length, new_data=True)
                                                
                                            print(f'{symbol}_o{output_size}_w{window_size}_d{data_mode}')

                                               
                                            model_name = f'svm_{symbol}_w{window_size}_o{output_size}_d{str(data_mode)}'
                                            config = self.config
                                            model_config = {
                                                        "num_layers": 10,
                                                        "hidden_size": 20,
                                                        "drop_out": drop_out,
                                                        "window_size": window_size,
                                                        "output_step": output_size,
                                                        "conv1D_param": {
                                                            "type": 1,
                                                            "kernel_size": 4,
                                                            "dilation_base": 3,
                                                            "max_pooling_kernel_size": 2,
                                                            "sub_small_num_layer": 1,
                                                            "sub_big_num_layer": 1,
                                                            "sub_small_kernel_size": 3,
                                                            "sub_big_kernel_size": 30,
                                                            "output_size": 20
                                                        },
                                                        "symbol": symbol,
                                                        "topk": 10,
                                                        "data_mode": data_mode,
                                                        "max_string_length": string_length,
                                            }
                                            config["model"] = model_config

                                            model = Model(name=model_name, num_feature=self.num_feature, parameters=config,
                                                            model_type=self.model_type,
                                                            full_name=model_name)
                                            model = self.grid_train(model, train_dataloader, valid_dataloader)
                                            train_score = self.grid_eval(model, train_dataloader)
                                            val_score = self.grid_eval(model, valid_dataloader)
                                            test_score = self.grid_eval(model, test_dataloader)
                                            balance_score = self.grid_eval(model, balance_dataloader)
                                            result = {
                                                "symbol": symbol,
                                                'output_size': output_size,
                                                'window_size': window_size,
                                                'data_mode': data_mode,
                                                "drop_out": drop_out,
                                                'max_string_lenght:': string_length,
                                                'train_score': train_score,
                                                'val_score': val_score,
                                                "test_score": test_score,
                                                "balance_score": balance_score  
                                            }

                                            # Append the current result to best_cases
                                            best_cases.append(result)

        results_df = pd.DataFrame(best_cases)

        # Sort the DataFrame by score in descending order
        results_df = results_df.sort_values('val_score', ascending=True)

        # # Drop duplicates based on data_mode, window_size, and output_size, keeping the first occurrence (highest score)
        # results_df = results_df.drop_duplicates(subset=['data_mode', 'window_size', 'output_size'], keep='first')

        results_df.to_csv("lstm_grid_search_results.csv", index=False)
        return results_df
    
    def prepare_gridsearch_data(self, symbol, data_mode, window_size, output_step, string_length, new_data):

        model_name = f'{self.model_type}_{symbol}_w{window_size}_o{output_step}_d{str(data_mode)}'
        file_paths = [
            './dataset/X_train_' + model_name + '.npy',
            './dataset/y_train_' + model_name + '.npy',
            './dataset/X_valid_' + model_name + '.npy',
            './dataset/y_valid_' + model_name + '.npy',
            './dataset/X_test_' + model_name + '.npy',
            './dataset/y_test_' + model_name + '.npy',
            './dataset/X_balance_test_' + model_name + '.npy',
            './dataset/y_balance_test_' + model_name + '.npy'
        ]
        if any(not os.path.exists(file_path) for file_path in file_paths) or new_data:
            df = u.prepare_stock_dataframe(symbol, window_size, self.start, self.end, False)
            num_data_points = df.shape[0]
            data_date = df.index.strftime("%Y-%m-%d").tolist()

            # Split train-val and test dataframes
            trainval_test_split_index = int(num_data_points * self.cf["data"]["train_test_split_size"])
            trainval_df = df.iloc[:trainval_test_split_index]
            test_df = df.iloc[trainval_test_split_index:]
            print("Train date from: " + trainval_df.index[0].strftime("%Y-%m-%d") + " to " + trainval_df.index[-1].strftime("%Y-%m-%d"))
            print("Test from: " + test_df.index[0].strftime("%Y-%m-%d") + " to " + test_df.index[-1].strftime("%Y-%m-%d"))


            # Prepare X data for train-val dataframe
            X_trainval, y_trainval = u.prepare_timeseries_dataset(trainval_df.to_numpy(), window_size=window_size,
                                                                output_step=output_step, dilation=1)
            dataset_slicing = X_trainval.shape[2]

            # Prepare X data for test dataframe
            X_test, y_test = u.prepare_timeseries_dataset(test_df.to_numpy(), window_size=window_size,
                                                        output_step=output_step, dilation=1)
            if data_mode == 2:
                news_X_trainval, _ = nlp_u.prepare_splited_news_data(trainval_df, symbol, window_size, self.start, self.end,
                                                            output_step, self.topk, string_length, True, False)
                X_trainval = np.concatenate((X_trainval, news_X_trainval), axis=2)
                news_X_test, _ = nlp_u.prepare_splited_news_data(test_df, symbol, window_size, self.start, self.end,
                                                            output_step, self.topk, string_length, False, False)
                X_test = np.concatenate((X_test, news_X_test), axis=2)
            self.num_feature = X_trainval.shape[2]
            # Concatenate X_stocks and news_X


            # Split X and y into train and validation datasets
            train_indices, valid_indices = train_test_split(range(X_trainval.shape[0]), test_size=1 - self.cf["data"]["train_val_split_size"], shuffle=True, random_state=42)
            X_train = X_trainval[train_indices]
            X_valid = X_trainval[valid_indices]
            y_train = y_trainval[train_indices]
            y_valid = y_trainval[valid_indices]

            class_0_indices = np.where(y_test == 0)[0]
            class_1_indices = np.where(y_test == 1)[0]
            min_class_count = min(len(class_0_indices), len(class_1_indices))
            balanced_indices = np.concatenate([class_0_indices[:min_class_count], class_1_indices[:min_class_count]])
            X_test_balanced = X_test[balanced_indices]
            y_test_balanced = y_test[balanced_indices]

            # Count the class distribution for each set
            train_class_counts = np.bincount(y_train[:, 0])
            valid_class_counts = np.bincount(y_valid[:, 0])
            test_class_counts = np.bincount(y_test[:, 0])
            test_balanced_class_counts = np.bincount(y_test_balanced[:, 0])

            print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
            print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
            print("Test set - Class 0 count:", test_class_counts[0], ", Class 1 count:", test_class_counts[1])
            print("Balanced Test set - Class 0 count:", test_balanced_class_counts[0], ", Class 1 count:",
                test_balanced_class_counts[1])
            # Save train and validation data
            X_train_file = './dataset/X_train_' + model_name + '.npy'
            X_valid_file = './dataset/X_valid_' + model_name + '.npy'
            X_test_file = './dataset/X_test_' + model_name + '.npy'
            
            y_train_file = './dataset/y_train_' + model_name + '.npy'
            y_valid_file = './dataset/y_valid_' + model_name + '.npy'
            y_test_file = './dataset/y_test_' + model_name + '.npy'
            X_balance_test_file = './dataset/X_balance_test_' + model_name + '.npy'
            y_balance_test_file = './dataset/y_balance_test_' + model_name + '.npy'
            # if os.path.exists(X_train_file):
            #     os.remove(X_train_file)
            # if os.path.exists(X_valid_file):
            #     os.remove(X_valid_file)
            # if os.path.exists(y_train_file):
            #     os.remove(y_train_file)
            # if os.path.exists(y_valid_file):
            #     os.remove(y_valid_file)

            np.save(X_train_file, X_train)
            np.save(X_valid_file, X_valid)
            np.save(X_test_file, X_test)
            np.save(X_balance_test_file, X_test_balanced)
            np.save(y_train_file, y_train)
            np.save(y_valid_file, y_valid)
            np.save(y_test_file, y_test)
            np.save(y_balance_test_file, y_test_balanced)
        else:
            # Load train and validation data
            X_train = np.load('./dataset/X_train_' + model_name + '.npy', allow_pickle=True)
            y_train = np.load('./dataset/y_train_' + model_name + '.npy', allow_pickle=True)
            X_valid = np.load('./dataset/X_valid_' + model_name + '.npy', allow_pickle=True)
            y_valid = np.load('./dataset/y_valid_' + model_name + '.npy', allow_pickle=True)
            
            self.num_feature = X_train.shape[2]
            # Load full test data
            X_test = np.load('./dataset/X_test_' + model_name + '.npy', allow_pickle=True)
            y_test = np.load('./dataset/y_test_' + model_name + '.npy', allow_pickle=True)
        # Create datasets and dataloaders for train and validation sets
        train_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_train, y_train, 39)
        valid_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_valid, y_valid, 39)
        test_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test, y_test, 39)
        test_balanced_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test_balanced, y_test_balanced, 39)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)
        # Create dataloaders for train, validation, full test, and balanced test
        test_balanced_dataloader = DataLoader(test_balanced_dataset, batch_size=self.batch_size,
                                                shuffle=self.test_shuffle)
        return train_dataloader, valid_dataloader, test_dataloader, test_balanced_dataloader
    def prepare_data(self, new_data):

        file_paths = [
            './dataset/X_train_' + self.model_name + '.npy',
            './dataset/y_train_' + self.model_name + '.npy',
            './dataset/X_valid_' + self.model_name + '.npy',
            './dataset/y_valid_' + self.model_name + '.npy',
            './dataset/X_test_' + self.model_name + '.npy',
            './dataset/y_test_' + self.model_name + '.npy'
        ]
        if any(not os.path.exists(file_path) for file_path in file_paths) or new_data:
            df = u.prepare_stock_dataframe(self.symbol, self.window_size, self.start, self.end, new_data)
            num_data_points = df.shape[0]
            data_date = df.index.strftime("%Y-%m-%d").tolist()

            # Split train-val and test dataframes
            trainval_test_split_index = int(num_data_points * self.cf["data"]["train_test_split_size"])
            trainval_df = df.iloc[:trainval_test_split_index]
            test_df = df.iloc[trainval_test_split_index:]
            print("Train date from: " + trainval_df.index[0].strftime("%Y-%m-%d") + " to " + trainval_df.index[-1].strftime("%Y-%m-%d"))
            print("Test from: " + test_df.index[0].strftime("%Y-%m-%d") + " to " + test_df.index[-1].strftime("%Y-%m-%d"))


            # Prepare X data for train-val dataframe
            X_trainval, y_trainval = u.prepare_timeseries_dataset(trainval_df.to_numpy(), window_size=self.window_size,
                                                                output_step=self.output_step, dilation=1)
            dataset_slicing = X_trainval.shape[2]

            # Prepare X data for test dataframe
            X_test, y_test = u.prepare_timeseries_dataset(test_df.to_numpy(), window_size=self.window_size,
                                                        output_step=self.output_step, dilation=1)
            if self.data_mode == 2:
                news_X_trainval, _ = nlp_u.prepare_splited_news_data(trainval_df, self.symbol, self.window_size, self.start, self.end,
                                                            self.output_step, self.topk, self.max_string_length, True, False)
                X_trainval = np.concatenate((X_trainval, news_X_trainval), axis=2)
                news_X_test, _ = nlp_u.prepare_splited_news_data(test_df, self.symbol, self.window_size, self.start, self.end,
                                                            self.output_step, self.topk, self.max_string_length, False, False)
                X_test = np.concatenate((X_test, news_X_test), axis=2)
            self.num_feature = X_trainval.shape[2]
            # Concatenate X_stocks and news_X


            # Split X and y into train and validation datasets
            train_indices, valid_indices = train_test_split(range(X_trainval.shape[0]), test_size=1 - self.cf["data"]["train_val_split_size"], shuffle=True, random_state=42)
            X_train = X_trainval[train_indices]
            X_valid = X_trainval[valid_indices]
            y_train = y_trainval[train_indices]
            y_valid = y_trainval[valid_indices]

            # Count the class distribution for each set
            train_class_counts = np.bincount(y_train[:, 0])
            valid_class_counts = np.bincount(y_valid[:, 0])
            test_class_counts = np.bincount(y_test[:, 0])

            print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
            print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
            print("Test set - Class 0 count:", test_class_counts[0], ", Class 1 count:", test_class_counts[1])

            # Save train and validation data
            X_train_file = './dataset/X_train_' + self.model_name + '.npy'
            X_valid_file = './dataset/X_valid_' + self.model_name + '.npy'
            X_test_file = './dataset/X_test_' + self.model_name + '.npy'
            y_train_file = './dataset/y_train_' + self.model_name + '.npy'
            y_valid_file = './dataset/y_valid_' + self.model_name + '.npy'
            y_test_file = './dataset/y_test_' + self.model_name + '.npy'

            if os.path.exists(X_train_file):
                os.remove(X_train_file)
            if os.path.exists(X_valid_file):
                os.remove(X_valid_file)
            if os.path.exists(y_train_file):
                os.remove(y_train_file)
            if os.path.exists(y_valid_file):
                os.remove(y_valid_file)

            np.save(X_train_file, X_train)
            np.save(X_valid_file, X_valid)
            np.save(X_test_file, X_test)
            np.save(y_train_file, y_train)
            np.save(y_valid_file, y_valid)
            np.save(y_test_file, y_test)
        else:
            # Load train and validation data
            X_train = np.load('./dataset/X_train_' + self.model_name + '.npy', allow_pickle=True)
            y_train = np.load('./dataset/y_train_' + self.model_name + '.npy', allow_pickle=True)
            X_valid = np.load('./dataset/X_valid_' + self.model_name + '.npy', allow_pickle=True)
            y_valid = np.load('./dataset/y_valid_' + self.model_name + '.npy', allow_pickle=True)
            
            self.num_feature = X_train.shape[2]
            # Load full test data
            X_test = np.load('./dataset/X_test_' + self.model_name + '.npy', allow_pickle=True)
            y_test = np.load('./dataset/y_test_' + self.model_name + '.npy', allow_pickle=True)
        # Create datasets and dataloaders for train and validation sets
        train_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_train, y_train, 39)
        valid_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_valid, y_valid, 39)
        test_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test, y_test, 39)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)

    def prepare_eval_data(self):
        # Load train and validation data
        X_train = np.load('./dataset/X_train_' + self.model_name + '.npy', allow_pickle=True)
        y_train = np.load('./dataset/y_train_' + self.model_name + '.npy', allow_pickle=True)
        X_valid = np.load('./dataset/X_valid_' + self.model_name + '.npy', allow_pickle=True)
        y_valid = np.load('./dataset/y_valid_' + self.model_name + '.npy', allow_pickle=True)

        # Load full test data
        X_test_full = np.load('./dataset/X_test_' + self.model_name + '.npy', allow_pickle=True)
        y_test_full = np.load('./dataset/y_test_' + self.model_name + '.npy', allow_pickle=True)
        
        # Balance the test set
        class_0_indices = np.where(y_test_full == 0)[0]
        class_1_indices = np.where(y_test_full == 1)[0]
        min_class_count = min(len(class_0_indices), len(class_1_indices))
        balanced_indices = np.concatenate([class_0_indices[:min_class_count], class_1_indices[:min_class_count]])
        X_test_balanced = X_test_full[balanced_indices]
        y_test_balanced = y_test_full[balanced_indices]


        dataset_slicing = 39

        # Create datasets for train, validation, full test, and balanced test
        train_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_train, y_train, dataset_slicing)
        valid_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_valid, y_valid, dataset_slicing)
        test_full_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test_full, y_test_full, dataset_slicing)
        test_balanced_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test_balanced, y_test_balanced, dataset_slicing)

        # Print class distribution for all datasets
        train_class_counts = np.bincount(np.squeeze(y_train))
        valid_class_counts = np.bincount(np.squeeze(y_valid))
        test_full_class_counts = np.bincount(np.squeeze(y_test_full))
        test_balanced_class_counts = np.bincount(np.squeeze(y_test_balanced))

        print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
        print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
        print("Full Test set - Class 0 count:", test_full_class_counts[0], ", Class 1 count:", test_full_class_counts[1])
        print("Balanced Test set - Class 0 count:", test_balanced_class_counts[0], ", Class 1 count:", test_balanced_class_counts[1])
        # Create dataloaders for train, validation, full test, and balanced test
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        test_full_dataloader = DataLoader(test_full_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)
        test_balanced_dataloader = DataLoader(test_balanced_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)

        return train_dataloader, valid_dataloader, test_full_dataloader,  test_balanced_dataloader
    
    def prepare_grid_eval_data(self, model_name):
        # Load train and validation data
        X_train = np.load('./dataset/X_train_' + model_name + '.npy', allow_pickle=True)
        y_train = np.load('./dataset/y_train_' + model_name + '.npy', allow_pickle=True)
        X_valid = np.load('./dataset/X_valid_' + model_name + '.npy', allow_pickle=True)
        y_valid = np.load('./dataset/y_valid_' + model_name + '.npy', allow_pickle=True)

        # Load full test data
        X_test_full = np.load('./dataset/X_test_' + model_name + '.npy', allow_pickle=True)
        y_test_full = np.load('./dataset/y_test_' + model_name + '.npy', allow_pickle=True)
        
        # Balance the test set
        class_0_indices = np.where(y_test_full == 0)[0]
        class_1_indices = np.where(y_test_full == 1)[0]
        min_class_count = min(len(class_0_indices), len(class_1_indices))
        balanced_indices = np.concatenate([class_0_indices[:min_class_count], class_1_indices[:min_class_count]])
        X_test_balanced = X_test_full[balanced_indices]
        y_test_balanced = y_test_full[balanced_indices]


        dataset_slicing = 39

        # Create datasets for train, validation, full test, and balanced test
        train_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_train, y_train, dataset_slicing)
        valid_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_valid, y_valid, dataset_slicing)
        test_full_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test_full, y_test_full, dataset_slicing)
        test_balanced_dataset = PriceAndIndicatorsAndNews_TimeseriesDataset(X_test_balanced, y_test_balanced, dataset_slicing)

        # Print class distribution for all datasets
        train_class_counts = np.bincount(np.squeeze(y_train))
        valid_class_counts = np.bincount(np.squeeze(y_valid))
        test_full_class_counts = np.bincount(np.squeeze(y_test_full))
        test_balanced_class_counts = np.bincount(np.squeeze(y_test_balanced))

        print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
        print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
        print("Full Test set - Class 0 count:", test_full_class_counts[0], ", Class 1 count:", test_full_class_counts[1])
        print("Balanced Test set - Class 0 count:", test_balanced_class_counts[0], ", Class 1 count:", test_balanced_class_counts[1])
        # Create dataloaders for train, validation, full test, and balanced test
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        test_full_dataloader = DataLoader(test_full_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)
        test_balanced_dataloader = DataLoader(test_balanced_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)

        return train_dataloader, valid_dataloader, test_full_dataloader,  test_balanced_dataloader
    def run_epoch(self, model, dataloader, optimizer, criterion, scheduler, is_training, device):
        epoch_loss = 0

        weight_decay = 0.001
        if is_training:
            model.structure.train()
        else:
            model.structure.eval()

        # create a tqdm progress bar
        dataloader = tqdm(dataloader)
        for idx, (x_stock, x_news, y) in enumerate(dataloader):
            if is_training:
                optimizer.zero_grad()
            batch_size = x_stock.shape[0]
            # print(x.shape)
            x_stock = x_stock.to(device)
            x_news = x_news.to(device)
            y = y.to(device)
            out = model.structure(x_stock, x_news)
            # Compute accuracy
            predictions = torch.argmax(out, dim=1).unsqueeze(1)
            correct = (predictions == y).sum().item()
            accuracy = correct / batch_size  # Multiply by 100 to get percentage

            # Print loss and accuracy
            print("Accuracy: {:.2f}%".format(accuracy))
            loss = criterion(out, y)
            if is_training:
                if loss != torch.nan:
                    torch.autograd.set_detect_anomaly(True)
                    loss.backward()
                    optimizer.step()
                else:
                    print("loss = nan")
            batch_loss = (loss.detach().item())
            epoch_loss += batch_loss
            # update the progress bar
            dataloader.set_description(f"At index {idx:.4f}")

        try:
            lr = scheduler.get_last_lr()[0]

        except:
            lr = optimizer.param_groups[0]['lr']
        return epoch_loss, lr
