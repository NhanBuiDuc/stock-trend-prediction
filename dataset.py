from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import torch


class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.sd = np.std(x, axis=0)
        normalized_x = (x - self.mu) / self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu


class MyMinMaxScaler():
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit_transform(self, x):
        x_tensor = torch.tensor(x)
        self.min_val = torch.min(x_tensor, dim=0).values
        self.max_val = torch.max(x_tensor, dim=0).values
        normalized_x = (x_tensor - self.min_val) / (self.max_val - self.min_val)
        return normalized_x.numpy()

    def inverse_transform(self, x):
        x_tensor = torch.tensor(x)
        return (x_tensor * (self.max_val - self.min_val)) + self.min_val


class StockAndNews_Dataset(Dataset):
    def __init__(self, x, y, slicing):
        self.scaler = MinMaxScaler(feature_range=(-100, 100))

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        x_stock = x[:, :, :slicing]
        self.x_news = x[:, :, slicing:]

        x_stock = x_stock.reshape((x_stock.shape[0] * x_stock.shape[1], x_stock.shape[2]))
        x_stock = self.scaler.fit_transform(x_stock)
        # Reshape the scaled data back to the original shape
        self.x_stock = x_stock.reshape((x.shape[0], x.shape[1], slicing))
        self.X = np.concatenate((self.x_stock, self.x_news), axis=2)
        self.X = self.X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x_stock)

    def __getitem__(self, idx):
        x_stock = self.x_stock[idx]
        x_news = self.x_news[idx]
        y = self.y[idx]
        return x_stock, x_news, y


class PriceAndIndicatorsAndNews_Dataset(Dataset):
    def __init__(self, x, y, slicing):
        self.scaler = MinMaxScaler(feature_range=(-100, 100))

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x_stock = x[:, :slicing]
        self.x_news = x[:, slicing:]
        self.x_stock = self.scaler.fit_transform(x_stock)
        # Reshape the scaled data back to the original shape
        self.x_price = self.x_stock[:, :5]
        self.x_indicators = self.x_stock[:, 5:slicing]
        self.X = np.concatenate((self.x_stock, self.x_news), axis=1)
        self.X = self.X.astype(np.float32)
        self.Y = y.astype(np.float32)
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return X, Y

class PriceAndIndicatorsAndNews_TimeseriesDataset(Dataset):
    def __init__(self, x, y, slicing):
        self.scaler = MinMaxScaler(feature_range=(-100, 100))

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x_stock = x[:, :, :slicing]
        self.x_news = x[:, :, slicing:]
        x_stock = x_stock.reshape((x_stock.shape[0] * x_stock.shape[1], x_stock.shape[2]))
        x_stock = self.scaler.fit_transform(x_stock)
        self.x_stock = x_stock.reshape((x.shape[0], x.shape[1], slicing))
        # Reshape the scaled data back to the original shape
        self.x_price = self.x_stock[:, :, :5]
        self.x_indicators = self.x_stock[:, :, 5:slicing]
        self.X = np.concatenate((self.x_stock, self.x_news), axis=2)
        self.X = self.X.astype(np.float32)
        self.Y = y.astype(np.float32)
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x_stock = self.x_stock[idx]
        x_news = self.x_news[idx]
        y = self.Y[idx]
        return x_stock, x_news, y




class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.scaler = MyMinMaxScaler(feature_range=(-100, 100))
        self.x = x.astype(np.float32)
        # Reshape the data
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        # Apply the scaler
        x = self.scaler.fit_transform(x)
        self.y = self.scaler.fit_transform(y)
        # Reshape the scaled data back to the original shape
        self.x = x.reshape((self.x.shape[0], self.x.shape[1], self.x.shape[2]))
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Classification_TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.scaler = MyMinMaxScaler(feature_range=(-10, 10))
        self.y = y.astype(np.float32)
        self.x = x.astype(np.float32)
        # Reshape the data
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        # Apply the scaler
        x = self.scaler.fit_transform(x)
        # Reshape the scaled data back to the original shape
        self.x = x.reshape((self.x.shape[0], self.x.shape[1], self.x.shape[2]))
        self.x = self.x.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Classification_Dataset(Dataset):
    def __init__(self, x, y):
        self.scaler = MyMinMaxScaler()
        self.y = y.astype(np.float32)
        x = self.scaler.fit_transform(x)
        self.x = x.astype(np.float32)
        self.num_classes = len(set(y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PredictPrice_TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x,
                           2)  # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
