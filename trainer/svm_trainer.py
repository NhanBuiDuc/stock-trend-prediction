from trainer.trainer import Trainer

import torch
from torch.utils.data import ConcatDataset
from model import Model
from configs.svm_config import svm_cf as cf
import torch.nn as nn
import util as u
from dataset import PriceAndIndicatorsAndNews_Dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import json
import numpy as np
import os
from torch.utils.data import DataLoader
import datetime
import NLP.util as nlp_u
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from itertools import product
class svm_trainer(Trainer):
    def __init__(self, new_data=True, full_data=False, mode="train"):
        super(svm_trainer, self).__init__()
        self.__dict__.update(self.cf)
        self.config = cf
        self.model_type = "svm"
        self.__dict__.update(self.config["model"])
        self.__dict__.update(self.config["training"])
        self.test_dataloader = None
        self.valid_dataloader = None
        self.train_dataloader = None
        self.full_data = full_data
        self.num_feature = None
        self.new_data = new_data
        self.model_name = f'{self.model_type}_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(self.data_mode)}'
        self.model_type_dict = self.cf["tensorflow_timeseries_model_type_dict"]
        self.model = None
        self.mode = mode
        if self.mode == "train":
            self.prepare_data(self.new_data)
        else:
            self.num_feature = 807
        self.indentify()

    def indentify(self):
        self.model = Model(name=self.model_name, num_feature=self.num_feature, parameters=self.config,
                           model_type=self.model_type,
                           full_name=self.model_name)

    def train(self):
        self.mode = "train"

        if not self.full_data:
            if self.data_mode == 0:
                X_train = self.train_dataloader.dataset.x_price
                y_train = self.train_dataloader.dataset.Y
                X_val = self.valid_dataloader.dataset.x_price
                y_val = self.valid_dataloader.dataset.Y
                self.num_feature = X_train.shape[-1]
            elif self.data_mode == 1:
                X_train = self.train_dataloader.dataset.x_stock
                X_val = self.valid_dataloader.dataset.x_stock
                y_train = self.train_dataloader.dataset.Y
                y_val = self.valid_dataloader.dataset.Y
                self.num_feature = X_train.shape[-1]
            elif self.data_mode == 2:
                X_train = self.train_dataloader.dataset.X
                y_train = self.train_dataloader.dataset.Y
                X_val = self.valid_dataloader.dataset.X
                y_val = self.valid_dataloader.dataset.Y
                self.num_feature = X_train.shape[-1]

            self.model.structure.fit(X_train, y_train)

            torch.save({"model": self.model,
                        "state_dict": []
                        },
                       "./models/" + self.model.name + ".pkl")
        elif self.full_data:
            self.combined_dataset = ConcatDataset([self.train_dataloader.dataset, self.valid_dataloader.dataset])

            if self.data_mode == 0:
                X_train = self.combined_dataset.dataset.x_price
                y_train = self.combined_dataset.dataset.Y
                x_val = self.valid_dataloader.dataset.x_price
                y_val = self.valid_dataloader.dataset.Y
            elif self.data_mode == 1:
                X_train = self.combined_dataset.dataset.x_stock
                x_val = self.combined_dataset.dataset.x_stock
                y_train = self.train_dataloader.dataset.Y
                y_val = self.valid_dataloader.dataset.Y
            elif self.data_mode == 2:
                X_train = self.combined_dataset.dataset.X
                y_train = self.combined_dataset.dataset.Y
                x_val = self.valid_dataloader.dataset.X
                y_val = self.valid_dataloader.dataset.Y

            self.model.structure.fit(X_train, y_train)
            torch.save({"model": self.model,
                        "state_dict": []
                        },
                       "./models/" + self.model.name + ".pkl")


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
                                for C in self.param_grid['C']:
                                        train_dataloader, valid_dataloader, test_dataloader, balance_dataloader = self.prepare_gridsearch_data(symbol, data_mode, window_size, output_size, string_length, new_data=True)

                                        X_train = train_dataloader.dataset.X
                                        X_val = valid_dataloader.dataset.X
                                        X_test = test_dataloader.dataset.X
                                        X_balance_test = balance_dataloader.dataset.X
                                        y_train = train_dataloader.dataset.Y
                                        y_val = valid_dataloader.dataset.Y
                                        y_test = test_dataloader.dataset.Y
                                        y_balance_test = balance_dataloader.dataset.Y
                                        num_feature = X_train.shape[-1]

                                        print(f'{symbol}_o{output_size}_w{window_size}_d{data_mode}_s{string_length}')

                                        X_train_w = X_train
                                        X_val_w = X_val
                                        y_train_o = y_train
                                        y_val_o = y_val
                                        model_name = f'svm_{symbol}_w{window_size}_o{output_size}_d{str(data_mode)}'
                                        config = self.config
                                        model_config = {
                                                    "C": C,
                                                    "kernel": 'poly',
                                                    "degree": 100,
                                                    "gamma": 'scale',
                                                    "coef0": 100,
                                                    "class_weight": {0: 0.5, 1: 0.5},
                                                    "window_size": window_size,
                                                    "output_step": output_size,
                                                    "data_mode": data_mode,
                                                    "topk": 10,
                                                    "symbol": symbol,
                                                    "max_string_length": string_length,
                                        }
                                        config["model"] = model_config
                                        svm_model = SVC(C=C, kernel='poly', gamma='scale', coef0=100, class_weight={0: 0.5, 1: 0.5})

                                        
                                        svm_model.fit(X_train_w, y_train_o)
                                        train_score = svm_model.score(X_train_w, y_train_o)
                                        val_score = svm_model.score(X_val_w, y_val_o)
                                        test_score = svm_model.score(X_test, y_test)
                                        balance_score = svm_model.score(X_balance_test, y_balance_test)
                                        result = {
                                            'symbol': symbol,
                                            'output_size': output_size,
                                            'window_size': window_size,
                                            'data_mode': data_mode,
                                            'C': C,
                                            'gamma': 'scale',
                                            'max_string_lenght': string_length,
                                            'train_score': train_score,
                                            'val_score': val_score,
                                            "test_score": test_score,
                                            "balance_score": balance_score                                    
                                        }
                                        best_cases.append(result)
                                        model = Model(name=model_name, num_feature=num_feature, parameters=config,
                                                        model_type=self.model_type,
                                                        full_name=model_name)
                                        model.structure.sklearn_model = svm_model
                                        torch.save({"model": model,
                                                    "state_dict": []
                                                    },
                                                "./models/" + model.name + ".pkl")
                                                # Check if the current result has the highest score within its own category
                                                # is_highest_score = not any(
                                                #     case['val_score'] > val_score and
                                                #     case['data_mode'] == data_mode and
                                                #     case['window_size'] == window_size and
                                                #     case['output_size'] == output_size
                                                #     for case in best_cases
                                                # )

                                                # if is_highest_score:
                                                #     # Remove cases with lower scores within the same category
                                                #     # Find indices of cases with lower scores within the same category
                                                #     lower_score_indices = [
                                                #         idx for idx, case in enumerate(best_cases) if (
                                                #             case['data_mode'] == data_mode and
                                                #             case['window_size'] == window_size and
                                                #             case['output_size'] == output_size and
                                                #             case['val_score'] < val_score
                                                #         )
                                                #     ]

                                                #     # Remove cases with lower scores by indices
                                                #     for idx in reversed(lower_score_indices):
                                                #         del best_cases[idx]


                                                #     # Append the current result to best_cases
                                                #     best_cases.append(result)
                                                #     model = Model(name=model_name, num_feature=self.num_feature, parameters=config,
                                                #                 model_type=self.model_type,
                                                #                 full_name=model_name)
                                                #     model.structure.sklearn_model = svm_model
                                                #     torch.save({"model": model,
                                                #             "state_dict": []
                                                #             },
                                                #         "./models/" + model.name + ".pkl")
                        else:
                            for C in self.param_grid['C']:
                                    string_length = 500
                                    train_dataloader, valid_dataloader, test_dataloader, balance_dataloader = self.prepare_gridsearch_data(symbol, data_mode, window_size, output_size, string_length, new_data=True)
                                    if data_mode == 0:
                                        X_train = train_dataloader.dataset.x_price
                                        X_val = valid_dataloader.dataset.x_price
                                        X_test = test_dataloader.dataset.x_price
                                        X_balance_test = balance_dataloader.dataset.x_price
                                        y_val = valid_dataloader.dataset.Y
                                        y_train = train_dataloader.dataset.Y
                                        y_test = test_dataloader.dataset.Y
                                        y_balance_test = balance_dataloader.dataset.Y
                                        num_feature = X_train.shape[-1]
                                    elif data_mode == 1:
                                        X_train = train_dataloader.dataset.x_stock
                                        X_val = valid_dataloader.dataset.x_stock
                                        X_test = test_dataloader.dataset.x_stock
                                        X_balance_test = balance_dataloader.dataset.x_stock
                                        y_train = train_dataloader.dataset.Y
                                        y_val = valid_dataloader.dataset.Y
                                        y_test = test_dataloader.dataset.Y
                                        y_balance_test = balance_dataloader.dataset.Y
                                        num_feature = X_train.shape[-1]

                                    print(f'{symbol}_o{output_size}_w{window_size}_d{data_mode}')


                                    X_train_w = X_train
                                    X_val_w = X_val
                                    y_train_o = y_train
                                    y_val_o = y_val
                                    model_name = f'svm_{symbol}_w{window_size}_o{output_size}_d{str(data_mode)}'
                                    config = self.config
                                    model_config = {
                                                "C": C,
                                                "kernel": 'poly',
                                                "degree": 100,
                                                "gamma": 'scale',
                                                "coef0": 100,
                                                "class_weight": {0: 0.5, 1: 0.5},
                                                "window_size": window_size,
                                                "output_step": output_size,
                                                "data_mode": data_mode,
                                                "topk": 10,
                                                "symbol": symbol,
                                                "max_string_length": string_length,
                                    }
                                    config["model"] = model_config
                                    svm_model = SVC(C = C, kernel='poly', gamma='scale', coef0=100, class_weight={0: 0.5, 1: 0.5})

                                    
                                    svm_model.fit(X_train_w, y_train_o)
                                    train_score = svm_model.score(X_train_w, y_train_o)
                                    val_score = svm_model.score(X_val_w, y_val_o)
                                    test_score = svm_model.score(X_test, y_test)
                                    balance_score = svm_model.score(X_balance_test, y_balance_test)
                                    result = {
                                        'symbol': symbol,
                                        'output_size': output_size,
                                        'window_size': window_size,
                                        'data_mode': data_mode,
                                        'C': C,
                                        'gamma': 'scale',
                                        'max_string_lenght': string_length,
                                        'train_score': train_score,
                                        'val_score': val_score,
                                        "test_score": test_score,
                                        "balance_score": balance_score
                                    }
                                    best_cases.append(result)
                                    model = Model(name=model_name, num_feature=num_feature, parameters=config,
                                                        model_type=self.model_type,
                                                        full_name=model_name)
                                    model.structure.sklearn_model = svm_model
                                    torch.save({"model": model,
                                            "state_dict": []
                                            },
                                        "./models/" + model.name + ".pkl")
                            # Check if the current result has the highest score within its own category
                            # is_highest_score = not any(
                            #     case['val_score'] > val_score and
                            #     case['data_mode'] == data_mode and
                            #     case['window_size'] == window_size and
                            #     case['output_size'] == output_size
                            #     for case in best_cases
                            # )

                            # if is_highest_score:
                            #     # Remove cases with lower scores within the same category
                            #     # Find indices of cases with lower scores within the same category
                            #     lower_score_indices = [
                            #         idx for idx, case in enumerate(best_cases) if (
                            #             case['data_mode'] == data_mode and
                            #             case['window_size'] == window_size and
                            #             case['output_size'] == output_size and
                            #             case['val_score'] < val_score
                            #         )
                            #     ]

                                # Remove cases with lower scores by indices
                                # for idx in reversed(lower_score_indices):
                                #     del best_cases[idx]


                                # # Append the current result to best_cases
                                # best_cases.append(result)
                                # model = Model(name=model_name, num_feature=self.num_feature, parameters=config,
                                #             model_type=self.model_type,
                                #             full_name=model_name)
                                # model.structure.sklearn_model = svm_model
                                # torch.save({"model": model,
                                #         "state_dict": []
                                #         },
                                #     "./models/" + model.name + ".pkl")
        results_df = pd.DataFrame(best_cases)
        # Sort the DataFrame by score in descending order
        results_df = results_df.sort_values('val_score', ascending=True)
        # results_df = results_df.drop_duplicates(subset=['data_mode', 'window_size', 'output_size'], keep='first')
        results_df.to_csv(f"svm_grid_search_results.csv", index=False)
        return results_df

    

    def eval(self, model):

        train_dataloader, valid_dataloader, test_dataloader, balancedtest_datataloader = self.prepare_eval_data()
        if self.data_mode == 0:
            train_dataloader.dataset.X = train_dataloader.dataset.x_price
            valid_dataloader.dataset.X = valid_dataloader.dataset.x_price
            test_dataloader.dataset.X = test_dataloader.dataset.x_price
            balancedtest_datataloader.dataset.X = balancedtest_datataloader.dataset.x_price
            self.num_feature = train_dataloader.dataset.X.shape[-1]
        elif self.data_mode == 1:
            train_dataloader.dataset.X = train_dataloader.dataset.x_stock
            valid_dataloader.dataset.X = valid_dataloader.dataset.x_stock
            test_dataloader.dataset.X = test_dataloader.dataset.x_stock
            balancedtest_datataloader.dataset.X = balancedtest_datataloader.dataset.x_stock
            self.num_feature = train_dataloader.dataset.X.shape[-1]
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
            for x, labels in dataloader:
                # Move inputs and labels to device
                # Forward pass
                # x = x.to(self.device)
                x = x.cpu().detach().numpy()
                labels = labels.to(self.device)
                # labels = labels.cpu().detach().numpy()
                outputs = model.predict(x).to(self.device)
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

            # Split train and test sets based on date
            train_split_index = int(num_data_points * self.cf["data"]["train_test_split_size"])  # 70% of data points
            train_valid_date = data_date[:train_split_index]
            test_date = data_date[train_split_index:]

            # Prepare y
            y = u.prepare_data_y_trend(df.to_numpy(), output_step=self.output_step)
            y = np.array(y, dtype=int)

            # Prepare X
            X_stocks = np.array(df.values)[:-self.output_step]

            # Save train, valid, and test datasets
            X_train_file = './dataset/X_train_' + self.model_name + '.npy'
            X_valid_file = './dataset/X_valid_' + self.model_name + '.npy'
            X_test_file = './dataset/X_test_' + self.model_name + '.npy'
            y_train_file = './dataset/y_train_' + self.model_name + '.npy'
            y_valid_file = './dataset/y_valid_' + self.model_name + '.npy'
            y_test_file = './dataset/y_test_' + self.model_name + '.npy'
            if self.data_mode == 2:
                _, news_X = nlp_u.prepare_news_data(df, self.symbol, self.window_size, self.start, self.end,
                                                    self.output_step,
                                                    self.topk, self.max_string_length, new_data)

                news_X = news_X[:-self.output_step]
                # Concatenate X_stocks and news_X
                X = np.concatenate((X_stocks, news_X), axis=1)
            else:
                X = X_stocks
            self.num_feature = X.shape[1]

            # Split X and y into train, valid, and test datasets
            train_indices = np.where(df.index.isin(train_valid_date))[0]
            test_indices = np.where(df.index.isin(test_date))[0][:-self.output_step]
            print("Train date from: " + train_valid_date[0] + " to " + train_valid_date[-1])
            print("Test from: " + test_date[0] + " to " + test_date[-1])
            X_train_valid = X[train_indices]
            X_test = X[test_indices]
            y_train_valid = y[train_indices]
            y_test = y[test_indices]

            # Perform stratified splitting on train and valid datasets
            sss_train = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.cf["data"]["train_val_split_size"],
                                            random_state=42)
            for train_index, valid_index in sss_train.split(X_train_valid, y_train_valid):
                X_train = X_train_valid[train_index]
                X_valid = X_train_valid[valid_index]
                y_train = y_train_valid[train_index]
                y_valid = y_train_valid[valid_index]

            # Print class distribution in train, valid, and test sets
            train_class_counts = np.bincount(y_train)
            valid_class_counts = np.bincount(y_valid)
            test_class_counts = np.bincount(y_test)
            print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
            print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
            print("Test set - Class 0 count:", test_class_counts[0], ", Class 1 count:", test_class_counts[1])

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
            # Load full test data
            X_test = np.load('./dataset/X_test_' + self.model_name + '.npy', allow_pickle=True)
            y_test = np.load('./dataset/y_test_' + self.model_name + '.npy', allow_pickle=True)
        # Create dataloaders for train, valid, and test sets
        train_dataset = PriceAndIndicatorsAndNews_Dataset(X_train, y_train, 39)
        valid_dataset = PriceAndIndicatorsAndNews_Dataset(X_valid, y_valid, 39)
        test_dataset = PriceAndIndicatorsAndNews_Dataset(X_test, y_test, 39)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)
     
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

            # Split train and test sets based on date
            train_split_index = int(num_data_points * self.cf["data"]["train_test_split_size"])  # 70% of data points
            train_valid_date = data_date[:train_split_index]
            test_date = data_date[train_split_index:]

            # Prepare y
            y = u.prepare_data_y_trend(df.to_numpy(), output_step=output_step)
            y = np.array(y, dtype=int)

            # Prepare X
            X_stocks = np.array(df.values)[:-output_step]

            # Save train, valid, and test datasets
            X_train_file = './dataset/X_train_' + model_name + '.npy'
            X_valid_file = './dataset/X_valid_' + model_name + '.npy'
            X_test_file = './dataset/X_test_' + model_name + '.npy'
            X_balance_test_file = './dataset/X_balance_test_' + model_name + '.npy'
            y_train_file = './dataset/y_train_' + model_name + '.npy'
            y_valid_file = './dataset/y_valid_' + model_name + '.npy'
            y_test_file = './dataset/y_test_' + model_name + '.npy'
            y_balance_test_file = './dataset/y_balance_test_' + model_name + '.npy'
            if data_mode == 2:
                _, news_X = nlp_u.prepare_news_data(df, symbol, window_size, self.start, self.end,
                                                    output_step,
                                                    self.topk, string_length, new_data=False)

                news_X = news_X[:-output_step]
                # Concatenate X_stocks and news_X
                X = np.concatenate((X_stocks, news_X), axis=1)
            else:
                X = X_stocks
            self.num_feature = X.shape[1]

            # Split X and y into train, valid, and test datasets
            train_indices = np.where(df.index.isin(train_valid_date))[0]
            test_indices = np.where(df.index.isin(test_date))[0][:-output_step]
            print("Train date from: " + train_valid_date[0] + " to " + train_valid_date[-1])
            print("Test from: " + test_date[0] + " to " + test_date[-1])
            X_train_valid = X[train_indices]
            X_test = X[test_indices]
            y_train_valid = y[train_indices]
            y_test = y[test_indices]

            # Perform stratified splitting on train and valid datasets
            sss_train = StratifiedShuffleSplit(n_splits=1, test_size=1 - self.cf["data"]["train_val_split_size"],
                                            random_state=42)
            for train_index, valid_index in sss_train.split(X_train_valid, y_train_valid):
                X_train = X_train_valid[train_index]
                X_valid = X_train_valid[valid_index]
                y_train = y_train_valid[train_index]
                y_valid = y_train_valid[valid_index]


            # Balance the test set
            class_0_indices = np.where(y_test == 0)[0]
            class_1_indices = np.where(y_test == 1)[0]
            min_class_count = min(len(class_0_indices), len(class_1_indices))
            balanced_indices = np.concatenate([class_0_indices[:min_class_count], class_1_indices[:min_class_count]])
            X_test_balanced = X_test[balanced_indices]
            y_test_balanced = y_test[balanced_indices]


            # Print class distribution for all datasets
            train_class_counts = np.bincount(y_train)
            valid_class_counts = np.bincount(y_valid)
            test_full_class_counts = np.bincount(y_test)
            test_balanced_class_counts = np.bincount(y_test_balanced)

            print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
            print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
            print("Full Test set - Class 0 count:", test_full_class_counts[0], ", Class 1 count:",
                test_full_class_counts[1])
            print("Balanced Test set - Class 0 count:", test_balanced_class_counts[0], ", Class 1 count:",
                test_balanced_class_counts[1])
            if os.path.exists(X_train_file):
                os.remove(X_train_file)
                np.save(X_train_file, X_train)
            if os.path.exists(X_valid_file):
                os.remove(X_valid_file)
                np.save(X_valid_file, X_valid)
            if os.path.exists(X_test_file):
                os.remove(X_test_file)
                np.save(X_test_file, X_test)
            if os.path.exists(X_balance_test_file):
                os.remove(X_balance_test_file)
                np.save(X_balance_test_file, X_test_balanced)                
            if os.path.exists(y_train_file):
                os.remove(y_train_file)
                np.save(y_train_file, y_train)
            if os.path.exists(y_valid_file):
                os.remove(y_valid_file)
                np.save(y_valid_file, y_valid)
            if os.path.exists(y_test_file):
                os.remove(y_test_file)
                np.save(y_test_file, y_test)
            if os.path.exists(y_test_file):
                os.remove(y_test_file)
                np.save(y_balance_test_file, y_test_balanced)
        else:
            # Load train and validation data
            X_train = np.load('./dataset/X_train_' + model_name + '.npy', allow_pickle=True)
            y_train = np.load('./dataset/y_train_' + model_name + '.npy', allow_pickle=True)
            X_valid = np.load('./dataset/X_valid_' + model_name + '.npy', allow_pickle=True)
            y_valid = np.load('./dataset/y_valid_' + model_name + '.npy', allow_pickle=True)
            # Load full test data
            X_test = np.load('./dataset/X_test_' + model_name + '.npy', allow_pickle=True)
            y_test = np.load('./dataset/y_test_' + model_name + '.npy', allow_pickle=True)
            X_test_balanced = np.load('./dataset/X_balance_test_' + model_name + '.npy', allow_pickle=True)
            y_test_balanced = np.load('./dataset/y_balance_test_' + model_name + '.npy', allow_pickle=True)
        # Create dataloaders for train, valid, and test sets
        train_dataset = PriceAndIndicatorsAndNews_Dataset(X_train, y_train, 39)
        valid_dataset = PriceAndIndicatorsAndNews_Dataset(X_valid, y_valid, 39)
        test_dataset = PriceAndIndicatorsAndNews_Dataset(X_test, y_test, 39)
        test_balanced_dataset = PriceAndIndicatorsAndNews_Dataset(X_test_balanced, y_test_balanced, 39)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)
        # Create dataloaders for train, validation, full test, and balanced test
        test_balanced_dataloader = DataLoader(test_balanced_dataset, batch_size=self.batch_size,
                                                shuffle=self.test_shuffle)
        return train_dataloader, valid_dataloader, test_dataloader, test_balanced_dataloader

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

        # Create datasets for train, validation, full test, and balanced test
        train_dataset = PriceAndIndicatorsAndNews_Dataset(X_train, y_train, 39)
        valid_dataset = PriceAndIndicatorsAndNews_Dataset(X_valid, y_valid, 39)
        test_full_dataset = PriceAndIndicatorsAndNews_Dataset(X_test_full, y_test_full, 39)
        test_balanced_dataset = PriceAndIndicatorsAndNews_Dataset(X_test_balanced, y_test_balanced, 39)

        # Create dataloaders for train, validation, full test, and balanced test
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle)
        test_full_dataloader = DataLoader(test_full_dataset, batch_size=self.batch_size, shuffle=self.test_shuffle)
        test_balanced_dataloader = DataLoader(test_balanced_dataset, batch_size=self.batch_size,
                                              shuffle=self.test_shuffle)

        # Print class distribution for all datasets
        train_class_counts = np.bincount(y_train)
        valid_class_counts = np.bincount(y_valid)
        test_full_class_counts = np.bincount(y_test_full)
        test_balanced_class_counts = np.bincount(y_test_balanced)

        print("Train set - Class 0 count:", train_class_counts[0], ", Class 1 count:", train_class_counts[1])
        print("Validation set - Class 0 count:", valid_class_counts[0], ", Class 1 count:", valid_class_counts[1])
        print("Full Test set - Class 0 count:", test_full_class_counts[0], ", Class 1 count:",
              test_full_class_counts[1])
        print("Balanced Test set - Class 0 count:", test_balanced_class_counts[0], ", Class 1 count:",
              test_balanced_class_counts[1])

        return train_dataloader, valid_dataloader, test_full_dataloader, test_balanced_dataloader

        # train_date, valid_date, test_date

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
            predictions = torch.argmax(out, dim=1)
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
