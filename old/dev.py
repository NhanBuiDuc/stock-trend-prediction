from old import utils
from config import config as cf
import numpy as np
import pandas as pd
from dataset import TimeSeriesDataset, Classification_TimeSeriesDataset
import infer
from plot import to_plot


def train_random_tree_classifier_14(data_df, num_data_points, data_date):
    # data_df = utils.get_new_df(data_df, '2018-01-01')

    sma = utils.SMA(data_df['4. close'].values, cf['data']['window_size'])
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], cf['data']['window_size'])
    rsi = utils.RSI(data_df, cf['data']['window_size'])
    vwap = utils.VWAP(data_df, cf['data']['window_size'])
    hma = utils.HMA(data_df['4. close'], cf['data']['window_size'])
    upward = utils.upward(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'],'upward': upward, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    window_size = cf["data"]["window_size"]
    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_14 = utils.prepare_tree_data_y_trend(n_row, close, 14)
    X = X[:-14]
    split_index = int(y_trend_14.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X[:split_index]
    X_test = X[split_index:]
    y_train_first = y_trend_14[:split_index]
    y_test = y_trend_14[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]
    # random_tree_classifier = train.train_random_forest_classfier(X_train, y_train, X_val, y_val, X_test, y_test)
    svm_classifier = train.train_svm_classfier(X_train, y_train, X_val, y_val, X_test, y_test)
    
def train_assemble(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]
    # data loss due to window_size for indicators
    dataset_df = utils.prepare_dataset_and_indicators(data_df, window_size)

    # close_df and dataset_df should be the same
    close_df = pd.DataFrame({'close': dataset_df['close']})
    features = dataset_df.columns.values
    close = close_df.to_numpy()
    n_row = len(dataset_df) - window_size
    # y_real_1 should be less than close_df = window size
    y_real_1 = utils.prepare_timeseries_data_y(
        num_rows=n_row, 
        data=close, 
        window_size=window_size, 
        output_size=1)
    # X should be equal to dataset_df lenght
    X = dataset_df.to_numpy()
    # X_set should be less than X = window size and equal to y_real
    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    loss_data_count = num_data_points - len(X_set)

    # new dates update from the losing datapoints
    dates = data_date[loss_data_count:]

    split_index = int(y_real_1.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_real_1[:split_index]
    y_test = y_real_1[split_index:]

    # data_date lenght must equal X_set Lenght, dates lenght must equal X_train_first lenght

    train_dates_first = dates[:split_index]
    test_dates = dates[split_index:]

    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    
    train_dates = train_dates_first[:split_index]
    val_dates = train_dates_first[split_index:]

    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]
    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_val, y_val)
    dataset_test = TimeSeriesDataset(X_test, y_test)

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_val, y_val)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_assemble_model_1(dataset_train, dataset_val, features)
    infer.evalute_assembly_regression(dataset_val=dataset_val , features = features)
    infer.evalute_assembly_regression(dataset_val=dataset_test, features = features)
    to_plot(dataset_test, dataset_val, y_test, y_val, num_data_points, dates, test_dates, val_dates)
def train_diff_1(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    window_size = cf["model"]["diff_1"]["window_size"]
    max_features = cf["model"]["diff_1"]["max_features"]
    thresh_hold = cf["training"]["diff_1"]["corr_thresh_hold"]
    dataset_df = utils.prepare_dataset_and_indicators(data_df, window_size)

    # prepare y df
    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    n_row = len(dataset_df) - window_size
    # calculate y
    y_diff_1 = utils.prepare_timeseries_data_y_diff(num_rows=n_row, data=close, window_size=window_size)

    # coppy dataframe
    temp_df = dataset_df.copy()[window_size:]
    temp_df["target"] = y_diff_1
    dataset_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                          main_columns=["target"],
                                                          max_columns = max_features,
                                                          threshold=thresh_hold,
                                                          show_heat_map = show_heat_map)
    X = dataset_df.to_numpy()

    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_diff_1.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_diff_1[:split_index]
    y_test = y_diff_1[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]
    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_val, y_val)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_regression_1(dataset_train, dataset_val, features = features, mask = mask)
    infer.evalute_diff_1(dataset_val=dataset_val, features=features)
    infer.evalute_diff_1(dataset_val=dataset_test, features=features)

def train_movement_3(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    window_size = cf["model"]["movement_3"]["window_size"]
    max_features = cf["model"]["movement_3"]["max_features"]
    thresh_hold = cf["training"]["movement_3"]["corr_thresh_hold"]
    dataset_df = utils.prepare_dataset_and_indicators(data_df, window_size)

    # prepare y df
    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    n_row = len(dataset_df) - window_size
    # calculate y
    y_trend_percentage_3 = utils.prepare_timeseries_data_y_trend_percentage(n_row, close, output_size=3)

    # coppy dataframe
    temp_df = dataset_df.copy()[window_size:]
    # temp_df["target_trend_down"] = y_trend_percentage_3[:, :1]
    temp_df["target_increasing"] = y_trend_percentage_3[:, 1:2]
    temp_df["target_percentage"] = y_trend_percentage_3[:, 2:]
    dataset_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                          main_columns=["target_increasing","target_percentage"],
                                                          max_columns = max_features,
                                                          threshold=thresh_hold,
                                                          show_heat_map = show_heat_map)
    X = dataset_df.to_numpy()

    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_trend_percentage_3.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_percentage_3[:split_index]
    y_test = y_trend_percentage_3[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_Movement_3(dataset_train_trend, dataset_val_trend, features, mask)
    infer.evalute_Movement_3(dataset_val=dataset_val_trend, features = features)
    infer.evalute_Movement_3(dataset_val=dataset_test_trend, features = features)
    
def train_movement_7(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["movement_7"]["window_size"]
    max_features = cf["model"]["movement_7"]["max_features"]
    thresh_hold = cf["training"]["movement_7"]["corr_thresh_hold"]
    dataset_df = utils.prepare_dataset_and_indicators(data_df, window_size)

    # prepare y df
    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    n_row = len(dataset_df) - window_size
    # calculate y
    y_trend_percentage_7 = utils.prepare_timeseries_data_y_trend_percentage(n_row, close, output_size=7)

    # coppy dataframe
    temp_df = dataset_df.copy()[window_size:]
    # temp_df["target_trend_down"] = y_trend_percentage_7[:, :1]
    temp_df["target_increasing"] = y_trend_percentage_7[:, 1:2]
    temp_df["target_percentage"] = y_trend_percentage_7[:, 2:]
    dataset_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                          main_columns=["target_increasing", "target_percentage"],
                                                          max_columns = max_features,
                                                          threshold = thresh_hold,
                                                          show_heat_map = show_heat_map)
    X = dataset_df.to_numpy()

    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_trend_percentage_7.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_percentage_7[:split_index]
    y_test = y_trend_percentage_7[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_Movement_7(dataset_train_trend, dataset_val_trend, features, mask)
    infer.evalute_Movement_7(dataset_val=dataset_val_trend, features = features)
    infer.evalute_Movement_7(dataset_val=dataset_test_trend, features = features)
def train_movement_14(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    window_size = cf["model"]["movement_14"]["window_size"]
    max_features = cf["model"]["movement_14"]["max_features"]
    thresh_hold = cf["training"]["movement_14"]["corr_thresh_hold"]
    dataset_df = utils.prepare_dataset_and_indicators(data_df, window_size)

    # prepare y df
    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    n_row = len(dataset_df) - window_size
    # calculate y
    y_trend_percentage_14 = utils.prepare_timeseries_data_y_trend_percentage(n_row, close, output_size= 14)
    count_10 = 0
    count_01 = 0
    for element in y_trend_percentage_14[:, :2]:
        if element[0] == 1 and element[1] == 0:
            count_10 += 1
        else:
            count_01 += 1

    print(f"Number of (1,0) occurrences: {count_10}")
    print(f"Number of (0,1) occurrences: {count_01}")
    # coppy dataframe
    temp_df = dataset_df.copy()[window_size:]
    # temp_df["target_trend_down"] = y_trend_percentage_14[:, :1]
    temp_df["target_increasing"] = y_trend_percentage_14[:, 1:2]
    temp_df["target_percentage"] = y_trend_percentage_14[:, 2:]
    dataset_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                          main_columns=["target_increasing", "target_percentage"],
                                                          max_columns = max_features,
                                                          threshold=thresh_hold,
                                                          show_heat_map = show_heat_map)
    X = dataset_df.to_numpy()

    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_trend_percentage_14.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_percentage_14[:split_index]
    y_test = y_trend_percentage_14[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_Movement_14(dataset_train_trend, dataset_val_trend, features, mask)
    infer.evalute_Movement_14(dataset_val=dataset_val_trend, features = features)
    infer.evalute_Movement_14(dataset_val=dataset_test_trend, features = features)


if __name__ == "__main__":
    data_df, num_data_points, data_dates = utils.download_data_api()
    # data_df, num_data_points, data_dates = utils.get_new_df(data_df, '2018-01-01')
    data_df.set_index('date', inplace=True)
    train_df, valid_df, test_df, train_date, valid_date, test_date = utils.split_train_valid_test_dataframe(data_df, num_data_points, data_dates)
    # data_df = utils.get_new_df(data_df, '2018-01-01')
    # train_random_tree_classifier_14(data_df, num_data_points, data_date)

    train_movement_3(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)
    train_movement_7(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)
    train_movement_14(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)
    train_diff_1(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)
    train_assemble(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)