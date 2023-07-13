# def SMA(x, window_size):
#     #output = [sum(row) / len(row) for row in x]
#     i = 0
#     sma = []
#     while i < (len(x) - window_size):
#         window = x[i:i+window_size]
#         window_avg = np.sum(window)/window_size
#         sma.append(window_avg)
#         i += 1   
#     sma = [float('nan')]*(window_size) + sma
#     return sma

# def EMA(x, smoothing, window_size):
#     k = smoothing/(window_size + 1)
#     ema = []
#     ema.append(x[0])
#     i = 1
#     while i < (len(x) - window_size + 1):
#         window_avg = x[i]*k + ema[i-1]*(1-k)
#         ema.append(window_avg)
#         i += 1
#     ema = [float('nan')]*(window_size-1) + ema
#     return ema

# def RSI(df, window_size, ema=True):
#     delta_close = df['4. close'].diff()
#     up = delta_close.clip(lower=0)
#     down = -1 * delta_close.clip(upper=0)
#     if ema == True:
# 	    # Use exponential moving average
#         ma_up = up.ewm(com = window_size - 1, adjust=True, min_periods = window_size).mean()
#         ma_down = down.ewm(com = window_size - 1, adjust=True, min_periods = window_size).mean()
#     else:
#         # Use simple moving average
#         ma_up = up.rolling(window = window_size, adjust=False).mean()
#         ma_down = down.rolling(window = window_size, adjust=False).mean()
        
#     rsi = ma_up / ma_down
#     rsi = 100 - (100/(1 + rsi))
#     return rsi.to_numpy().tolist()

def VWAP(df, window_size):
    close = np.array(df['4. close'])
    vol = np.array(df['6. volume'])
    cum_price_volume = np.cumsum(close * vol)
    cum_volume = np.cumsum(vol)
    vwap = cum_price_volume[window_size-1:] / cum_volume[window_size-1:]
    vwap = vwap.tolist()
    vwap = [float('nan')]*(window_size-1) + vwap
    return vwap

'''
https://oxfordstrat.com/trading-strategies/hull-moving-average/
'''
def WMA(s, window_size):
    wma = s.rolling(window_size).apply(lambda x: ((np.arange(window_size)+1)*x).sum()/(np.arange(window_size)+1).sum(), raw=True)
    return wma

def HMA(s, window_size):
    wma1 = WMA(s, window_size//2)
    wma2 = WMA(s, window_size)
    hma = WMA(wma1.multiply(2).sub(wma2), int(np.sqrt(window_size)))
    return hma.tolist()
  
'''
https://oxfordstrat.com/trading-strategies/hull-moving-average/
'''
def WMA(s, window_size):
    wma = s.rolling(window_size).apply(lambda x: ((np.arange(window_size)+1)*x).sum()/(np.arange(window_size)+1).sum(), raw=True)
    return wma

def HMA(s, window_size):
    wma1 = WMA(s, window_size//2)
    wma2 = WMA(s, window_size)
    hma = WMA(wma1.multiply(2).sub(wma2), int(np.sqrt(window_size)))
    return hma.tolist()

def CMO(s, window_size):
    '''
    Chande Momentum Oscillator
    The CMO is another well-known STI that employs momentum to locate 
    the relative behavior of stock by demonstrating its strengths and weaknesses in a particular period 
    CMO = 100 * ((Su - Sd)/ ( Su + Sd ) )
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo#:~:text=The%20CMO%20indicator%20is%20created,%2D100%20to%20%2B100%20range.
    '''
    price_diff = s - s.shift(1)
    gains = price_diff.copy()
    losses = price_diff.copy()
    # Set all negative values in gains to 0
    gains[gains < 0] = 0
    # Set all positive values in losses to 0
    losses[losses > 0] = 0
    gains_sum = gains.rolling(window=window_size).sum()
    losses_sum = abs(losses.rolling(window=window_size).sum())
    diff = gains_sum - losses_sum
    gains_losses_sum = gains_sum + losses_sum
    cmo = (diff / gains_losses_sum).multiply(100)
    return cmo.tolist()

def ROC(s, window_size):
    '''
    The Price Rate of Change
    ROC is the percentage change between the current price with respect to an earlier closing price n periods ago.
    ROC = [(Today’s Closing Price – Closing Price n periods ago) / Closing Price n periods ago] x 100
    no need to multiply by 100
    '''
    roc = (s - s.shift(window_size)) / s.shift(window_size)
    return roc.tolist()

def DMI(df, window_size):
    '''
    Directional Movement Index
    This STI identifies the direction of price movement by comparing prior highs and lows.
    The directional movement index (DMI) is a technical indicator that measures both 
    the strength and direction of a price movement and is intended to reduce false signals.
    window_size typically = 14 days ----> remember to tranfer dataframe to datatime object
    Note: import pandas_ta 
    func return as dataframe have 3 cols: adx, dmp, dmn we take adx col only
    '''
    dmi = ta.trend.adx(df['2. high'], df['3. low'], df['4. close'], length=window_size)
    #dmi = dmi.iloc[:, 0].tolist() # get adx only
    return dmi
    
def CCI(df, window_size):
    '''
    Commodity Channel Index
    CCI measures the difference between a security's price change and its average price change.
    window_size typically = 20 days
    '''
    cci = ta.momentum.cci(high=df['2. high'], low=df['3. low'], close=df['4. close'], window=window_size, constant=0.015)
    return cci.tolist()

def calculate_money_flow_volume_series(df):
    mfv = df['6. volume'] * (2*df['4. close'] - df['2. high'] - df['3. low']) / \
                                    (df['2. high'] - df['3. low'])
    return mfv
    
def calculate_money_flow_volume(df, window_size):
    return calculate_money_flow_volume_series(df).rolling(window_size).sum()

def CMF(df, window_size=20):
    '''
    Chakin Money Flow
    It measures the amount of Money Flow Volume over a specific period.
    https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf
    window_size typically = 20 days
    '''
    return (calculate_money_flow_volume(df, window_size) / df['6. volume'].rolling(window_size).sum()).tolist()

def MACD(df):
    '''
    Moving Average Convergence Divergence (MACD)
    The MACD is a popular indicator to that is used to identify a security's trend.
    While APO and MACD are the same calculation, MACD also returns two more series
    called Signal and Histogram. The Signal is an EMA of MACD and the Histogram is
    the difference of MACD and Signal.
    This STI is calculated by taking the difference of the EA of 26 days from the EA of 12 days.
    Args:
        close (pd.Series): Series of 'close's
        fast (int): The short period. Default: 12
        slow (int): The long period. Default: 26
        signal (int): The signal period. Default: 9
    Returns:
        pd.DataFrame: macd, histogram, signal columns.
    '''
    return ta.macd(close=df['4. close'])

def SO(df):
    '''
    The Stochastic Oscillator
    It is a range-bound oscillator with two lines moving between 0 and 100.
    The first line (%K) displays the current close in relation to the period's
    high/low range. The second line (%D) is a Simple Moving Average of the %K line.
    The most common choices are a 14 period %K and a 3 period SMA for %D.
    Args:
        high (pd.Series): Series of 'high's
        low (pd.Series): Series of 'low's
        close (pd.Series): Series of 'close's
        k (int): The Fast %K period. Default: 14
        d (int): The Slow %K period. Default: 3
    Returns:
        pd.DataFrame: %K, %D columns.
    '''
    beta = ta.stoch(high=df['2. high'], low=df['3. low'], close=df['4. close'], k=14, d=3)
    nan_df = pd.DataFrame(np.nan, index=range(df.shape[0]-beta.shape[0]), columns=beta.columns)
    return pd.concat([nan_df, beta]).reset_index(drop=True)

def RCI(df, window_size):
    '''
    Rank Correlation Index
    The RCI indicator is used to identify potential changes in
    market sentiment to expose turning points. 
    RCI is the combination of price change data and time change data
    '''
    ranks = df[['2. high', '3. low', '4. close']].rank(axis=1, method='min')
    rank_sums = ranks.sum(axis=1)

    # Calculate the RCI for each row
    rci_values = []
    for i in range(window_size, len(df)):
        start_index = i - window_size
        end_index = i
        # Get the rank sum for the current period
        rank_sum = rank_sums.iloc[end_index - 1]
        # Get the rank sum for the previous period
        prev_rank_sum = rank_sums.iloc[start_index - 1]
        # Calculate the RCI
        rci = (rank_sum - prev_rank_sum) / ((window_size ** 3 - window_size) / 12)
        rci_values.append(rci)

    # Add NaN values to the beginning of the DataFrame to match the length of the input DataFrame
    rci_values = np.concatenate((np.full(window_size - 1, np.nan), rci_values))

    # Create a new DataFrame containing the RCI values
    rci_df = pd.DataFrame(rci_values, index=df.index, columns=['rci'])
    return rci_df
      
def upward(close, window_size):
    # Create a new empty array to hold the results
    comparison_array = []
    close = np.array(close)
    # Loop through each element in the arrays and compare them
    for i in range(len(close) - window_size):
        # if tommorow close price better
        if close[i] > close[i - window_size]:
            comparison_array.append(1)
        else:
            comparison_array.append(0)
    return [float('nan')]*(window_size) + comparison_array

def daily_dataframe(api_key, symbol):
    ts = TimeSeries(key=api_key)
    data, meta_data = ts.get_daily_adjusted(symbol, outputsize="full")
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def sma_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_sma(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def ema_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_ema(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def vwap_dataframe(ti, symbol, window_size):
    try:
        data, meta_data = ti.get_vwap(symbol=symbol, interval="daily")
        df = pd.DataFrame.from_dict(data, orient='index').astype(float)
        return df
    finally:
        api_key = cf["alpha_vantage"]["api_key"]
        function = "VWAP"

        api = api_builder()


def willr_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_willr(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def macd_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_macd(symbol=symbol, interval="daily")
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def stochrsi_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_stochrsi(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def mom_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_mom(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def cmo_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_cmo(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def roc_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_roc(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def minus_di_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_minus_di(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def plus_di_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_plus_di(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def minus_dm_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_minus_dm(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def plus_dm_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_plus_dm(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def bbands_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_bbands(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def ad_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_ad(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def obv_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_obv(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


def aroon_dataframe(ti, symbol, window_size):
    data, meta_data = ti.get_aroon(symbol=symbol, interval="daily", time_period=window_size)
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    return df


   sma_df = sma_dataframe(ti, symbol, window_size)
    ema_df = ema_dataframe(ti, symbol, window_size)
    # vwap_df = vwap_dataframe(ti, symbol, window_size)
    willr_df = willr_dataframe(ti, symbol, window_size)
    macd_df = macd_dataframe(ti, symbol, window_size)
    stochrsi_df = stochrsi_dataframe(ti, symbol, window_size)
    mom_df = mom_dataframe(ti, symbol, window_size)
    cmo_df = cmo_dataframe(ti, symbol, window_size)
    roc_df = roc_dataframe(ti, symbol, window_size)
    minus_di_df = minus_di_dataframe(ti, symbol, window_size)
    plus_di_df = plus_di_dataframe(ti, symbol, window_size)
    minus_dm_df = minus_dm_dataframe(ti, symbol, window_size)
    plus_dm_df = plus_dm_dataframe(ti, symbol, window_size)
    bbands_df = bbands_dataframe(ti, symbol, window_size)
    ad_df = ad_dataframe(ti, symbol, window_size)
    obv_df = obv_dataframe(ti, symbol, window_size)
    aroon_df = aroon_dataframe(ti, symbol, window_size)

    final_df = daily_adjusted_df.append(sma_df)
    final_df = daily_adjusted_df.append(ema_df)
    # final_df = daily_adjusted_df.append(vwap_df)
    final_df = daily_adjusted_df.append(willr_df)
    final_df = daily_adjusted_df.append(macd_df)
    final_df = daily_adjusted_df.append(stochrsi_df)
    final_df = daily_adjusted_df.append(mom_df)
    final_df = daily_adjusted_df.append(cmo_df)
    final_df = daily_adjusted_df.append(roc_df)
    final_df = daily_adjusted_df.append(minus_di_df)
    final_df = daily_adjusted_df.append(plus_di_df)
    final_df = daily_adjusted_df.append(minus_dm_df)
    final_df = daily_adjusted_df.append(plus_dm_df)
    final_df = daily_adjusted_df.append(bbands_df)
    final_df = daily_adjusted_df.append(ad_df)
    final_df = daily_adjusted_df.append(obv_df)
    final_df = daily_adjusted_df.append(aroon_df)
