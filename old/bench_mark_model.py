import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from config import config as cf
from keras.optimizers import Adam
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def bench_mark_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    # create random forest regressor object
    model = RandomForestRegressor()

    # train the regressor using the training data
    model.fit(X_train, y_train)

    # evaluate the regressor on the validation data
    mse = mean_squared_error(y_test, model.predict(X_test))

    # evaluate the regressor on the test data
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mse, mae


def bench_mark_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    # create random forest regressor object
    model = SVR()

    # train the regressor using the training data
    model.fit(X_train, y_train)

    # evaluate the regressor on the validation data
    mse = mean_squared_error(y_test, model.predict(X_test))

    # evaluate the regressor on the test data
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mse, mae


def create_gru_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = Sequential()

    # Add LSTM layers
    model.add(GRU(units=100, input_shape=(14, 5), return_sequences=True))
    model.add(Dense(units=1))
    # Print model summary
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error', 'mean_absolute_error'])

    history_LSTM = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_val, y_val))
    _, mse, mae = model.evaluate(X_test, y_test)
    return model, mse, mae


def create_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = Sequential()

    # Add LSTM layers
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=True, activation='relu'))
    # Add output layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    # Print model summary
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error', 'mean_absolute_error'])

    history_LSTM = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val))
    _, mse, mae = model.evaluate(X_test, y_test)
    return model, mse, mae
