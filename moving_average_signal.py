import os 
import json
import tensorflow as tf
import datetime
import util
from util import *
import numpy as np
import pandas as pd
from configs.price_config import price_cf as cf
import pandas_ta as ta


# window_size = [10,20,30,50,100,200] for MA
class Signal:
    def __init__(self):
        pass

    def get_start_date(self, date_length=200, end_date=None):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_day = end_date - datetime.timedelta(days=date_length)
        start_day = start_day.strftime(end_date, '%Y-%m-%d')
        return start_day
    
    def get_stock_dataframe(self):
        symbol = cf['alpha_vantage']['symbol']
        api_key = cf["alpha_vantage"]["key"]
        filename = symbol + '.csv'
        
        src_path = 'csv/'
        if not file_exist(src_path, filename):
            df = download_stock_csv_file(src_path, filename, api_key, 14)
        df = pd.read_csv(src_path + filename)
        #df = df.iloc[-date_length:]
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        
        df = pd.DataFrame({
            'date':df['date'],
            '4. close': df['close'],
            '1. open': df['open'],
            '2. high': df['high'],
            '3. low': df['low'],
            '6. volume': df['volume']
        })
        # Moving Average
        for i in [10, 20, 30, 50, 100, 200]:
            sma = SMA(df, i)
            ema = EMA(df, i)
            df = pd.concat([df, sma], axis=1)
            df = pd.concat([df, ema], axis=1)
        hma = HMA(df, 9)
        df = pd.concat([df, hma], axis=1)

        # Oscillators
        # for i in [12, 26]:
        #     macd = MACD(df, i)
        #     df = pd.concat([df, macd], axis=1)
        
        '''
        smi 5 - 20 - 5 
        dm 14
        cfo 9
        cmo len = 7
        er len = auto
        roc len = auto
        stc len = auto
        vwap len = auto
        cmf len = 20
        '''
        # smi = ta.smi(close=df['close'], fast=5, slow=20, signal=5)
        # dm = DM(df, 14)
        # cfo = CFO(df, 9)
        # cmf = CMF(df, 20)
        
        macd = MACD(df, 12)
        rsi = RSI(df, 14)
        stochrsi = STOCHRSI(df, window_size=14)
        cci = CCI(df, 20)
        willr = WILLR(df, 14)
        mom = MOM(df, 10)
        eri = ERI(df, window_size=13)
        bbands = BBANDS(df, 5)
        uo = UO(df, 14)
        df = pd.concat([df, rsi], axis=1)
        df = pd.concat([df, macd], axis=1)
        df = pd.concat([df, stochrsi], axis=1)
        df = pd.concat([df, willr], axis=1)
        df = pd.concat([df, cci], axis=1)
        df = pd.concat([df, mom], axis=1)
        df = pd.concat([df, eri], axis=1)
        df = pd.concat([df, bbands], axis=1)
        df = pd.concat([df, uo], axis=1)
        return df
    
    
    def ema_signal1(self, df, short, long):
        ema_fast = df['EMA_' + str(short)]
        ema_slow = df['EMA_' + str(long)]
        signal = np.zeros(len(df))
        for i in range(1, len(df)):
            if ema_fast.iloc[i] != np.nan and ema_slow.iloc[i] != np.nan :
                if ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
                    signal[i] = 1
                elif ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def ema_signal2(self,df, period):
        signal = []
        for i in range(0, len(df)):
            if df['EMA_' + str(period)].iloc[i] != np.nan:
                if df['EMA_' + str(period)].iloc[i] > df['4. close'].iloc[i]:
                    signal.append(-1)
                elif df['EMA_' + str(period)].iloc[i] < df['4. close'].iloc[i]:
                    signal.append(1)
                else:
                    signal.append(0)
            else:
                signal.append(np.nan)
        return signal
    def sma_signal(self, df, period):
        signal = []
        for i in range(0, len(df)):
            if df['SMA_' + str(period)].iloc[i] != np.nan:
                if df['SMA_' + str(period)].iloc[i] > df['4. close'].iloc[i]:
                    signal.append(-1)
                elif df['SMA_' + str(period)].iloc[i] < df['4. close'].iloc[i]:
                    signal.append(1)
                else:
                    signal.append(0)
            else:
                signal.append(np.nan)
        return signal

    def hma_signal(self, df, period=9):
        signal = np.zeros(len(df))
        for i in range(1, len(df)):
            if df['HMA_' + str(period)][i] is not None:
                if df['HMA_' + str(period)][i] > df['4. close'][i] and df['HMA_' + str(period)][i-1] <= df['4. close'][i-1]:
                    signal[i] = 1
                elif df['HMA_' + str(period)][i] < df['4. close'][i] and df['HMA_' + str(period)][i-1] >= df['4. close'][i-1]:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal
    
    def macd_signal(self, df):
        # return macds, signal
        # need to check
        label_macd = 'MACD_12_26_9'
        label_macds = 'MACDs_12_26_9'
        signal = [0]
        for i in range(1, len(df)):
            if df[label_macd][i] is not None and df[label_macds][i] is not None and \
                    df[label_macd][i-1] is not None and df[label_macds][i-1] is not None:
                if df[label_macd][i] > df[label_macds][i] and df[label_macd][i-1] <= df[label_macds][i-1]:
                    signal.append(1)
                elif df[label_macd][i] < df[label_macds][i] and df[label_macd][i-1] >= df[label_macds][i-1]:
                    signal.append(-1)
                else:
                    signal.append(0)
            else:
                signal.append(np.nan)
        return signal
        
    def rsi_signal(self, df):
        signal = np.zeros(len(df))
        label = 'RSI_14'
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] > 70:
                    signal[i] = -1
                elif df[label][i] < 30:
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def stochrsi_signal(self, df):
        label = 'STOCHRSIk_14_14_3_3'
        signal = np.zeros(len(df))
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] < 20:
                    signal[i] = 1
                elif df[label][i] > 80:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def willr_signal(self, df):
        signal = np.zeros(len(df))
        label = 'WILLR_14'
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] > -20 and df[label][i] < 0:
                    signal[i] = -1
                elif df[label][i] > -100 and df[label][i] < -80:
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def mom_signal(self, df):
        label = 'MOM_10'
        signal = np.zeros(len(df))
        for i in range(1, len(df)):
            if df[label][i] is not None and df[label][i-1] is not None:
                if (df[label][i] > 0 and df[label][i-1] <= 0) or \
                        (df[label][i] > 0 and df[label][i-1] > 0 and signal[i-1] == 1):
                    signal[i] = 1
                elif (df[label][i] < 0 and df[label][i-1] >= 0) or \
                    (df[label][i] < 0 and df[label][i-1] < 0 and signal[i-1] == -1):
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def eri_signal(self, df):
        signal = np.zeros(len(df))
        be = 'BEARP_13'
        bu = 'BULLP_13'
        for i in range(1, len(df)):
            if (df[bu][i] is not None and df[bu][i-1] is not None) or \
                (df[be][i] is not None and df[be][i-1] is not None):
                if (df[bu][i] > 0 and df[bu][i-1] < 0) or \
                        (df[bu][i] > 0 and df[bu][i-1] > 0 and signal[i-1] == 1):
                    signal[i] = 1
                elif (df[be][i] < 0 and df[be][i-1] > 0) or \
                        (df[be][i] < 0 and df[be][i-1] < 0 and signal[i-1] == -1):
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def cci_signal(self, df):
        signal = np.zeros(len(df))
        label = 'CCI_20_0.015'
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] > 100:
                    signal[i] = 1
                elif df[label][i] < -100:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def bbands_signal(self, df):
        signal = np.zeros(len(df))
        label = '_5_2.0'
        for i in range(40, len(df)):
            if pd.isna(df['BBL' + label][i]) or pd.isna(df['BBU' + label][i]) or pd.isna(df['BBB' + label][i]):
                signal[i] = np.nan
            elif df['4. close'][i] < df['BBL' + label][i] and df['4. close'][i-1] > df['BBL' + label][i-1]:
                signal[i] = 1
            elif df['4. close'][i] > df['BBU' + label][i] and df['4. close'][i-1] < df['BBU' + label][i-1]:
                signal[i] = -1
            else:
                signal[i] = 0
        return signal

    def uo_signal(self, df):
        signal = np.zeros(len(df))
        label = 'UO_7_14_28'
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] < 30:
                    signal[i] = 1
                elif df[label][i] > 70:
                    signal[i] = -1
            else:
                signal[i] = np.nan
        return signal

   

    def define_signal(self, path='./technical_signal/'):
        # if new_data:
        #     df = self.get_stock_dataframe(date_length=200)
        # else:
        #     if not file_exist(path, file_name=filename):
        #         df = self.get_stock_dataframe(date_length=200)
        #     else:
        #         df = read_csv_file(path, filename)

        df = self.get_stock_dataframe()

        for i in [10, 20, 30, 50, 100, 200]:
            sma_s = self.sma_signal(df, i)
            df['s_SMA_' + str(i)] = sma_s
            ema_s = self.ema_signal2(df, i)
            df['s_EMA_' + str(i)] = ema_s

        hma_s = self.hma_signal(df, 9)
        df['s_HMA'] = hma_s

        macd_s = self.macd_signal(df)
        df['s_MACD'] = macd_s

        rsi_s = self.rsi_signal(df)
        df['s_RSI'] = rsi_s

        stochrsi_s = self.stochrsi_signal(df)
        df['s_STOCHRSI'] = stochrsi_s

        willr_s = self.willr_signal(df)
        df['s_WILLR'] = willr_s

        mom_s = self.mom_signal(df)
        df['s_MOM'] = mom_s

        eri_s = self.eri_signal(df)
        df['s_ERI'] = eri_s

        cci_s = self.cci_signal(df)
        df['s_CCI'] = cci_s

        bbands_s = self.bbands_signal(df)
        df['s_BBANDS'] = bbands_s

        uo_s = self.uo_signal(df)
        df['s_UO'] = uo_s
        
        df['SELL'] = (df.iloc[:, 36:] == -1).sum(axis=1)
        df['NEU'] = (df.iloc[:, 36:] == 0).sum(axis=1)
        df['BUY'] = (df.iloc[:, 36:] == 1).sum(axis=1)
        
        new_columns = {
            '4. close': '4. close',
            '1. open': '1. open',
            '2. high': '2. high',
            '3. low': '3. low',
            '6. volume': '6. volumn'
        }
        df = df.rename(columns=new_columns)
        #df.columns = df.columns.str.replace('.', '')
        file_name = cf['alpha_vantage']['symbol'] + '_signal.csv'
        name = path + file_name
        #save to csv file
        df.to_csv(name)

        #save to json file
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] < '2023-05-02']
        df['date'] = df['date'].dt.strftime('%d/%m/%Y')
        df = df.fillna("")
        df = df.to_dict(orient='records')
        file_path = './APP_WEB/static/file/' + cf['alpha_vantage']['symbol'] + '_signal.json'
        # Save the data to a JSON file with the specified path
        with open(file_path, 'w') as file:
            file.write(json.dumps(df, indent=2))


        return df



if __name__ == "__main__":
    print(os.getcwd())
    signal = Signal()
    df = signal.define_signal(path='technical_signal/')
    
