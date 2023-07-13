from old import utils
from config import config as cf
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataset import TimeSeriesDataset
from bench_mark_model import bench_mark_random_forest, create_lstm_model, bench_mark_svm, create_gru_model
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib


def train_random_forest(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    # calculate y
    y_train = train_close_df.to_numpy()[1:]
    y_valid = valid_close_df.to_numpy()[1:]
    y_test = test_close_df.to_numpy()[1:]

    # close_df and dataset_df should be the same
    X_train = train_df.to_numpy()[:-1]
    X_valid = valid_df.to_numpy()[:-1]
    X_test = test_df.to_numpy()[:-1]
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)
    
    y_train = scaler.fit_transform(y_train)
    y_valid = scaler.fit_transform(y_valid)
    y_test = scaler.fit_transform(y_test)

    model, mse, mae = bench_mark_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    print("mse", mse)
    print("mae", mae)   
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "random_forest.pkl"
    joblib.dump(model, "./bench_mark_models/" + model_name)
def train_svm(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    # calculate y
    y_train = train_close_df.to_numpy()[1:]
    y_valid = valid_close_df.to_numpy()[1:]
    y_test = test_close_df.to_numpy()[1:]

    # close_df and dataset_df should be the same
    X_train = train_df.to_numpy()[:-1]
    X_valid = valid_df.to_numpy()[:-1]
    X_test = test_df.to_numpy()[:-1]
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)
    
    y_train = scaler.fit_transform(y_train)
    y_valid = scaler.fit_transform(y_valid)
    y_test = scaler.fit_transform(y_test)

    model, mse, mae = bench_mark_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    print("mse", mse)
    print("mae", mae)   
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "svm.pkl"
    joblib.dump(model, "./bench_mark_models/" + model_name)

def train_lstm(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]
    output_step = 1
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

    train_n_row = len(train_close_df) - window_size - output_step
    valid_n_row = len(valid_close_df) - window_size - output_step
    test_n_row = len(test_close_df) - window_size - output_step

    # calculate y
    y_train = utils.prepare_timeseries_data_y(train_n_row, train_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_valid = utils.prepare_timeseries_data_y(valid_n_row, valid_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_test = utils.prepare_timeseries_data_y(test_n_row, test_close_df.to_numpy(), window_size= window_size, output_size=1)

    # date modified 
    train_date = train_date[ int(len(train_date) - len(y_train)) :]
    valid_date = valid_date[ int(len(valid_date) - len(y_valid)) :]
    test_date = test_date[ int(len(test_date) - len(y_test)) :]
    # close_df and dataset_df should be the same
    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)[:-output_step]
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)[:-output_step]
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)[:-output_step]
    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_valid, y_valid)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    model, mse, mae = create_lstm_model(dataset_train.x, dataset_train.y, dataset_val.x, dataset_val.y,
                                            dataset_test.x, dataset_test.y)
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "LSTM"
    tf.keras.models.save_model(model, './bench_mark_models/' + model_name, save_format='h5')

    print("mse", mse)
    print("mae", mae)  

def train_gru(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = 14
    output_step = 1
    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)


    train_df = train_df.iloc[:, :5]
    valid_df = valid_df.iloc[:, :5]
    test_df = test_df.iloc[:, :5]
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_n_row = len(train_close_df) - window_size - output_step
    valid_n_row = len(valid_close_df) - window_size - output_step
    test_n_row = len(test_close_df) - window_size - output_step

    # calculate y
    y_train = utils.prepare_timeseries_data_y(train_n_row, train_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_valid = utils.prepare_timeseries_data_y(valid_n_row, valid_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_test = utils.prepare_timeseries_data_y(test_n_row, test_close_df.to_numpy(), window_size= window_size, output_size=1)

    # date modified 
    train_date = train_date[ int(len(train_date) - len(y_train)) :]
    valid_date = valid_date[ int(len(valid_date) - len(y_valid)) :]
    test_date = test_date[ int(len(test_date) - len(y_test)) :]
    # close_df and dataset_df should be the same
    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)[:-output_step]
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)[:-output_step]
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)[:-output_step]

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_valid, y_valid)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    model, mse, mae = create_gru_model(dataset_train.x, dataset_train.y, dataset_val.x, dataset_val.y,
                                            dataset_test.x, dataset_test.y)
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "GRU"
    tf.keras.models.save_model(model, './bench_mark_models/' + model_name, save_format='h5')

    print("mse", mse)
    print("mae", mae)  

def evaluate_forest(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    # calculate y
    y_train = train_close_df.to_numpy()[1:]
    y_valid = valid_close_df.to_numpy()[1:]
    y_test = test_close_df.to_numpy()[1:]

    # close_df and dataset_df should be the same
    X_train = train_df.to_numpy()[:-1]
    X_valid = valid_df.to_numpy()[:-1]
    X_test = test_df.to_numpy()[:-1]
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)
    
    y_train = scaler.fit_transform(y_train)
    y_valid = scaler.fit_transform(y_valid)
    y_test = scaler.fit_transform(y_test)

    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "random_forest.pkl"

    model = joblib.load('./bench_mark_models./' + model_name)
    mse = mean_squared_error(y_test, model.predict(X_test))

    # evaluate the regressor on the test data
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(model_name + " mse", mse)
    print(model_name + " mae", mae)
def evaluate_svm(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = 14

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    # calculate y
    y_train = train_close_df.to_numpy()[1:]
    y_valid = valid_close_df.to_numpy()[1:]
    y_test = test_close_df.to_numpy()[1:]

    # close_df and dataset_df should be the same
    X_train = train_df.to_numpy()[:-1]
    X_valid = valid_df.to_numpy()[:-1]
    X_test = test_df.to_numpy()[:-1]
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_test = scaler.fit_transform(X_test)
    
    y_train = scaler.fit_transform(y_train)
    y_valid = scaler.fit_transform(y_valid)
    y_test = scaler.fit_transform(y_test)

    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "svm.pkl"

    model = joblib.load('./bench_mark_models./' + model_name)
    mse = mean_squared_error(y_test, model.predict(X_test))

    # evaluate the regressor on the test data
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(model_name + " mse", mse)
    print(model_name + " mae", mae)

def evaluate_lstm(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]
    output_step = 1
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

    train_n_row = len(train_close_df) - window_size - output_step
    valid_n_row = len(valid_close_df) - window_size  - output_step
    test_n_row = len(test_close_df) - window_size - output_step

    # calculate y
    y_train = utils.prepare_timeseries_data_y(train_n_row, train_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_valid = utils.prepare_timeseries_data_y(valid_n_row, valid_close_df.to_numpy(), window_size= window_size, output_size=1)
    y_test = utils.prepare_timeseries_data_y(test_n_row, test_close_df.to_numpy(), window_size= window_size, output_size=1)

    # date modified 
    train_date = train_date[ int(len(train_date) - len(y_train)) :]
    valid_date = valid_date[ int(len(valid_date) - len(y_valid)) :]
    test_date = test_date[ int(len(test_date) - len(y_test)) :]
    # close_df and dataset_df should be the same
    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)[:-output_step]
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)[:-output_step]
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)[:-output_step]

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_valid, y_valid)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "LSTM"
    model = tf.keras.models.load_model('./bench_mark_models./' + model_name)
    
    _, mse, mae = model.evaluate(dataset_test.x, dataset_test.y)
    print(model_name + " mse", mse)
    print(model_name + " mae", mae)

def evaluate_gru(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]
    output_step = 1
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
    model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "GRU"
    model = tf.keras.models.load_model('./bench_mark_models./' + model_name)
    
    _, mse, mae = model.evaluate(dataset_test.x, dataset_test.y)
    print(model_name + " mse", mse)
    print(model_name + " mae", mae)
if __name__ == "__main__":
    data_df, num_data_points, data_dates = utils.download_data_api('2000-5-01', '2023-04-01')
    train_df, valid_df, test_df, train_date, valid_date, test_date = utils.split_train_valid_test_dataframe(data_df, num_data_points, data_dates)
    # train_random_forest(data_df, 
    #                 num_data_points,
    #                 train_df, valid_df,
    #                 test_df, train_date,valid_date, test_date,
    #                 data_dates, show_heat_map = False, is_train = True)
    # train_lstm(data_df, 
    #                 num_data_points,
    #                 train_df, valid_df,
    #                 test_df, train_date,valid_date, test_date,
    #                 data_dates, show_heat_map = False, is_train = True)
    # train_svm(data_df, 
    #                 num_data_points,
    #                 train_df, valid_df,
    #                 test_df, train_date,valid_date, test_date,
    #                 data_dates, show_heat_map = False, is_train = True)
    train_gru(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = True)
    evaluate_forest(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)
    evaluate_svm(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)
    evaluate_lstm(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)
    evaluate_gru(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False)