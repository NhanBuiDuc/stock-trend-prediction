from old import utils
from configs import config as cf
import pandas as pd
from dataset import TimeSeriesDataset, Classification_TimeSeriesDataset
from plot import to_plot
import gridsearch as gs

def  train_assemble(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = False):

    window_size = cf["model"]["assemble_1"]["window_size"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)

    # date modified 
    train_date = train_date[ int(len(train_date) - len(train_df)) :]
    valid_date = valid_date[ int(len(valid_date) - len(valid_df)) :]

    test_date = test_date[ int(len(test_date) - len(test_df)) :]
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_n_row = len(train_close_df) - window_size
    valid_n_row = len(valid_close_df) - window_size
    test_n_row = len(test_close_df) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y(train_n_row, train_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_valid = utils.prepare_timeseries_data_y(valid_n_row, valid_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_test = utils.prepare_timeseries_data_y(test_n_row, test_close_df.to_numpy(), window_size= window_size, output_size=1)

    # date modified 
    train_date = train_date[ int(len(train_date) - len(y_train)) :]
    valid_date = valid_date[ int(len(valid_date) - len(y_valid)) :]
    test_date = test_date[ int(len(test_date) - len(y_test)) :]
    # close_df and dataset_df should be the same
    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_valid, y_valid)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if search:
        train.train_assemble_model_1(dataset_train, dataset_val, features = train_df.columns.values)
    infer.evalute_assembly_regression(dataset_val=dataset_val , features = train_df.columns.values)
    infer.evalute_assembly_regression(dataset_val=dataset_test, features = train_df.columns.values)
    to_plot(dataset_test, dataset_val, y_test, y_valid, num_data_points, data_dates, test_date, valid_date)
def train_diff_1(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    

    window_size = cf["model"]["diff_1"]["window_size"]
    max_features = cf["model"]["diff_1"]["max_features"]
    thresh_hold = cf["training"]["diff_1"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)

    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_n_row = len(train_close_df) - window_size
    valid_n_row = len(valid_close_df) - window_size
    test_n_row = len(test_close_df) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_diff(train_n_row, train_close_df.to_numpy(), window_size)
    y_valid = utils.prepare_timeseries_data_y_diff(valid_n_row, valid_close_df.to_numpy(), window_size = 14)
    y_test = utils.prepare_timeseries_data_y_diff(test_n_row, test_close_df.to_numpy(), window_size = 14)

    # copy dataframe
    # to merge targets to dataframe we need to drop first 14 targets 
    # because It has no X until form a window size of data
    temp_df = train_df.copy()[window_size:]

    temp_df["target"] = y_train[:]

    temp_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                       main_columns=["target"],
                                                       max_columns = max_features,
                                                       threshold=thresh_hold,
                                                       show_heat_map = show_heat_map)
    train_df = train_df[features]
    valid_df = valid_df[features]
    test_df = test_df[features]

    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_valid, y_valid)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_regression_1(dataset_train, dataset_val, features = full_features, mask = mask)
    infer.evalute_diff_1(dataset_val=dataset_val, features=features)
    infer.evalute_diff_1(dataset_val=dataset_test, features=features)

def train_movement_3(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = False):
    
    window_size = cf["model"]["movement_3"]["window_size"]
    max_features = cf["model"]["movement_3"]["max_features"]
    thresh_hold = cf["training"]["movement_3"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)

    full_features = train_df.columns.values
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_n_row = len(train_close_df) - window_size
    valid_n_row = len(valid_close_df) - window_size
    test_n_row = len(test_close_df) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_trend_percentage(train_n_row, train_close_df.to_numpy(), output_size = 3)
    y_valid = utils.prepare_timeseries_data_y_trend_percentage(valid_n_row, valid_close_df.to_numpy(), output_size = 3)
    y_test = utils.prepare_timeseries_data_y_trend_percentage(test_n_row, test_close_df.to_numpy(), output_size = 3)

    # copy dataframe
    # to merge targets to dataframe we need to drop first 14 targets 
    # because It has no X until form a window size of data
    temp_df = train_df.copy()[window_size:]

    temp_df["target_increasing"] = y_train[:, 1:2]
    temp_df["target_percentage"] = y_train[:, 2:]

    temp_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                       main_columns=["target_increasing","target_percentage"],
                                                       max_columns = max_features,
                                                       threshold=thresh_hold,
                                                       show_heat_map = show_heat_map, )
    # train_df = train_df[features]
    # valid_df = valid_df[features]
    # test_df = test_df[features]

    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_valid, y_valid)
    # dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)

    if search:
        gs.gridsearch_movement_3(dataset_train_trend, dataset_val_trend, full_features, mask)
    # infer.evalute_Movement_3(dataset_val=dataset_val_trend, features = features)
    # infer.evalute_Movement_3(dataset_val=dataset_test_trend, features = features)



def train_movement_7(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = False):
    
    window_size = cf["model"]["movement_7"]["window_size"]
    max_features = cf["model"]["movement_7"]["max_features"]
    thresh_hold = cf["training"]["movement_7"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)

    full_features = train_df.columns.values
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})
    
    train_n_row = len(train_close_df) - window_size
    valid_n_row = len(valid_close_df) - window_size
    test_n_row = len(test_close_df) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_trend_percentage(train_n_row, train_close_df.to_numpy(), output_size = 7)
    y_valid = utils.prepare_timeseries_data_y_trend_percentage(valid_n_row, valid_close_df.to_numpy(), output_size = 7)
    y_test = utils.prepare_timeseries_data_y_trend_percentage(test_n_row, test_close_df.to_numpy(), output_size = 7)

    # copy dataframe
    # to merge targets to dataframe we need to drop first 14 targets 
    # because It has no X until form a window size of data
    temp_df = train_df.copy()[window_size:]

    temp_df["target_increasing"] = y_train[:, 1:2]
    temp_df["target_percentage"] = y_train[:, 2:]

    temp_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                       main_columns=["target_increasing","target_percentage"],
                                                       max_columns = max_features,
                                                       threshold=thresh_hold,
                                                       show_heat_map = show_heat_map)
    train_df = train_df[features]
    valid_df = valid_df[features]
    test_df = test_df[features]

    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_valid, y_valid)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)

    if search:
        gs.gridsearch_movement_7(dataset_train_trend, dataset_val_trend, full_features, mask)


def train_movement_14(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = False):
    
    window_size = cf["model"]["movement_14"]["window_size"]
    max_features = cf["model"]["movement_14"]["max_features"]
    thresh_hold = cf["training"]["movement_14"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)

    full_features = train_df.columns.values

    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_n_row = len(train_close_df) - window_size
    valid_n_row = len(valid_close_df) - window_size
    test_n_row = len(test_close_df) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_trend_percentage(train_n_row, train_close_df.to_numpy(), output_size = 14)
    y_valid = utils.prepare_timeseries_data_y_trend_percentage(valid_n_row, valid_close_df.to_numpy(), output_size = 14)
    y_test = utils.prepare_timeseries_data_y_trend_percentage(test_n_row, test_close_df.to_numpy(), output_size = 14)

    # copy dataframe
    # to merge targets to dataframe we need to drop first 14 targets 
    # because It has no X until form a window size of data
    temp_df = train_df.copy()[window_size:]

    temp_df["target_increasing"] = y_train[:, 1:2]
    temp_df["target_percentage"] = y_train[:, 2:]

    temp_df, features, mask = utils.correlation_filter(dataframe=temp_df,
                                                       main_columns=["target_increasing","target_percentage"],
                                                       max_columns = max_features,
                                                       threshold=thresh_hold,
                                                       show_heat_map = show_heat_map)
    train_df = train_df[features]
    valid_df = valid_df[features]
    test_df = test_df[features]

    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_valid, y_valid)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)

    if search:
        gs.gridsearch_movement_14(dataset_train_trend, dataset_val_trend, full_features, mask)

if __name__ == "__main__":
    data_df, num_data_points, data_dates = utils.download_data_api()
    data_df.set_index('date', inplace=True)
    # data_df, num_data_points, data_dates = utils.get_new_df(data_df, '2018-01-01')
    train_df, valid_df, test_df, train_date, valid_date, test_date = utils.split_train_valid_test_dataframe(data_df, num_data_points, data_dates)
    # data_df = utils.get_new_df(data_df, '2023-04-01')

    # train_random_tree_classifier_14(data_df, num_data_points, data_date)

    train_movement_3(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = True)
    train_movement_7(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = False)
    train_movement_14(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = True)
    # train_diff_1(data_df, 
    #                 num_data_points,
    #                 train_df, valid_df,
    #                 test_df, train_date,valid_date, test_date,
    #                 data_dates, show_heat_map = False, is_train = True)
    train_assemble(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, search = True)