def train_lstm_classifier_14(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma_14 = utils.SMA(data_df['4. close'].values, window_size)
    sma_7 = utils.SMA(data_df['4. close'].values, 7)
    sma_3 = utils.SMA(data_df['4. close'].values, 3)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    upward = utils.upward(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'upward': upward, 'sma_3' : sma_3, 'sma_7' : sma_7, 'sma_14' : sma_14, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_14 = utils.prepare_data_y_trend(n_row, close, 14)
    X_set = utils.prepare_timeseries_data(X, window_size=window_size)
    split_index = int(y_trend_14.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_14[:split_index]
    y_test = y_trend_14[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_classifier_14(dataset_train_trend, dataset_val_trend)
    infer.evalute_classifier_14(dataset_val=dataset_val_trend)
    infer.evalute_classifier_14(dataset_val=dataset_test_trend)

def train_lstm_classifier_1(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma_14 = utils.SMA(data_df['4. close'].values, window_size)
    sma_7 = utils.SMA(data_df['4. close'].values, 7)
    sma_3 = utils.SMA(data_df['4. close'].values, 3)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    upward = utils.upward(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'upward': upward, 'sma_3' : sma_3, 'sma_7' : sma_7, 'sma_14' : sma_14, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_1 = utils.prepare_data_y_trend(n_row, close, 1)
    X_set = utils.prepare_timeseries_data(X, window_size=window_size)
    split_index = int(y_trend_1.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_1[:split_index]
    y_test = y_trend_1[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_classifier_1(dataset_train_trend, dataset_val_trend)
    infer.evalute_classifier_1(dataset_val=dataset_val_trend)
    infer.evalute_classifier_1(dataset_val=dataset_test_trend)

def train_lstm_classifier_7(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma_14 = utils.SMA(data_df['4. close'].values, window_size)
    sma_7 = utils.SMA(data_df['4. close'].values, 7)
    sma_3 = utils.SMA(data_df['4. close'].values, 3)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    upward = utils.upward(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'upward': upward, 'sma_3' : sma_3, 'sma_7' : sma_7, 'sma_14' : sma_14, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_7 = utils.prepare_data_y_trend(n_row, close, 7)
    X_set = utils.prepare_timeseries_data(X, window_size=window_size)
    split_index = int(y_trend_7.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_7[:split_index]
    y_test = y_trend_7[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_classifier_7(dataset_train_trend, dataset_val_trend)
    infer.evalute_classifier_7(dataset_val=dataset_val_trend)
    infer.evalute_classifier_7(dataset_val=dataset_test_trend)
def train_lstm_classifier_percentage_14(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma_14 = utils.SMA(data_df['4. close'].values, window_size)
    sma_7 = utils.SMA(data_df['4. close'].values, 7)
    sma_3 = utils.SMA(data_df['4. close'].values, 3)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    upward = utils.upward(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'upward': upward, 'sma_3' : sma_3, 'sma_7' : sma_7, 'sma_14' : sma_14, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})

    # dataset_df = dataset_df[15:]
    dataset_df = dataset_df.dropna()
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_percentage_14 = utils.prepare_timeseries_data_y_trend_percentage(n_row, close, 14)
    X_set = utils.prepare_timeseries_data(X, window_size=window_size)
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
        train.train_Movement_14(dataset_train_trend, dataset_val_trend)
    infer.evalute_Movement_14(dataset_val=dataset_val_trend)
    infer.evalute_Movement_14(dataset_val=dataset_test_trend)


def train_assemble(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma_14 = utils.SMA(data_df['4. close'].values, window_size)
    sma_7 = utils.SMA(data_df['4. close'].values, 7)
    sma_3 = utils.SMA(data_df['4. close'].values, 3)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    upward = utils.upward(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'upward': upward, 'sma_3' : sma_3, 'sma_7' : sma_7, 'sma_14' : sma_14, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})

    # dataset_df = dataset_df[15:]
    dataset_df = dataset_df.dropna()
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_real_1 = utils.prepare_timeseries_data_y(n_row, close, window_size, 1)
    X_set = utils.prepare_timeseries_data(X, window_size=window_size)
    split_index = int(y_real_1.shape[0]*cf["data"]["train_split_size"])
    dates = data_date[15:-window_size]
    train_dates_first = dates[:split_index]
    test_dates = dates[split_index:]
    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_real_1[:split_index]
    y_test = y_real_1[split_index:]
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
    if is_train:
        train.train_assemble_model(dataset_train, dataset_val)
    infer.evalute_assembly_regression(dataset_val=dataset_val)
    infer.evalute_assembly_regression(dataset_val=dataset_test)
    to_plot(dataset_test, dataset_val, y_test, y_val, num_data_points, dates, test_dates, val_dates)

