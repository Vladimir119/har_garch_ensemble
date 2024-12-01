import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_hourly_data(name='BTC', start='2024-01-01', end='2024-10-01'):
    data = yf.download(f'{name}-USD', start=start, end=end, interval='1h')
    data['Returns'] = 100 * data['Adj Close'].pct_change().dropna()
    data['Log Returns'] = (np.log(data['Adj Close'])).diff()
    return data

def generate_HAR_data(data, w_len=7, m_len=30):
    """
    Функция для генерации нужных лагов для HAR модели из данных data.
    """
    new_data = data.copy()
    volatility = new_data['Log Returns'].resample('D').std()
    daily_returns = new_data['Log Returns'].resample('D').mean()

    rv_data = pd.DataFrame({
        'Datetime': pd.to_datetime(volatility.index),
        'rv': volatility.values,
        'daily returns': daily_returns.values
    })

    rv_data['rv w'] = rv_data['rv'].rolling(window=w_len).mean()
    rv_data['rv m'] = rv_data['rv'].rolling(window=m_len).mean()

    return rv_data

def generate_HARQ_data(data, w_len=7, m_len=30):
    """
    Функция для генерации нужных лагов для HARQ модели из данных data.
    """
    new_data = data.copy()
    volatility = new_data['Log Returns'].resample('D').std()
    rq = new_data['Log Returns'].resample('D').apply(lambda x: np.sum(x**4)) # hours / 3
    daily_returns = new_data['Log Returns'].resample('D').mean()

    rv_data = pd.DataFrame({
        'Datetime': pd.to_datetime(rq.index),
        'rv': volatility.values,
        'rq': rq.values,
        'daily returns': daily_returns.values
    })

    rv_data['rv w'] = rv_data['rv'].rolling(window=w_len).mean()
    rv_data['rv m'] = rv_data['rv'].rolling(window=m_len).mean()

    rv_data['rq w'] = rv_data['rq'].rolling(window=w_len).mean()
    rv_data['rq m'] = rv_data['rq'].rolling(window=m_len).mean()

    return rv_data

def generate_HARJ_data(data, w_len=7, m_len=30):
    """
    Функция для генерации нужных лагов для HARJ (с д jump) модели из данных data.
    """
    new_data = data.copy()
    volatility = new_data['Log Returns'].resample('D').std()
    daily_returns = new_data['Log Returns'].resample('D').mean()
    bpv = new_data['Log Returns'].resample('D').apply(lambda x: np.sum(np.abs(x[1:]) * np.abs(x[:-1])))
    jump = np.maximum(volatility.values - bpv.values, 0)

    rv_data = pd.DataFrame({
        'Datetime': pd.to_datetime(volatility.index),
        'rv': volatility.values,
        'daily returns': daily_returns.values,
        'bpv': bpv,
        'jump': jump
    })

    rv_data['rv w'] = rv_data['rv'].rolling(window=w_len).mean()
    rv_data['rv m'] = rv_data['rv'].rolling(window=m_len).mean()

    return rv_data

class HAR:
  def __init__(self, name='HAR'):
    self.name = name

  def __fit_har(self, data):
    X = pd.DataFrame({
      'RV': data['rv'][:-1],
      'RV_w': data['rv w'][:-1],
      'RV_m': data['rv m'][:-1]
    })
    self.X = X

  def __fit_harq(self, data):
    X = pd.DataFrame({
      'RV': data['rv'][:-1],
      'RV_w': data['rv w'][:-1],
      'RV_m': data['rv w'][:-1],
      'RQ': np.sqrt(data['rq'][:-1]) * data['rv'][:-1],
      'RQ_w': np.sqrt(data['rq w'][:-1]) * data['rv w'][:-1],
      'RQ_m': np.sqrt(data['rq m'][:-1]) * data['rv m'][:-1],
    })
    self.X = X

  def __fit_harj(self, data):
    X = pd.DataFrame({
      'RV': data['rv'][:-1],
      'RV_w': data['rv w'][:-1],
      'RV_m': data['rv w'][:-1],
      'J': data['jump'][:-1]
    })
    self.X = X

  def fit(self, data):
    if self.name == 'HAR':
        self.__fit_har(data)
    elif self.name == 'HARQ':
        self.__fit_harq(data)
    elif self.name == 'HAR-J':
        self.__fit_harj(data)

    X = self.X.reset_index(drop=True)
    X = sm.add_constant(X)
    y = data['rv'][1:]
    y = y.reset_index(drop=True)
    model = sm.OLS(y, X).fit()
    self.model = model

  def predict(self, data):

    last_row = 0
    if self.name == 'HAR':
      last_row = data[['rv', 'rv w', 'rv m']].iloc[-1]
    elif self.name == 'HARQ':
      last_row = data[['rv', 'rv w', 'rv m', 'rq', 'rq w', 'rq m']].iloc[-1]
    elif self.name == 'HAR-J':
      last_row = data[['rv', 'rv w', 'rv m', 'jump']].iloc[-1]

    last_row_with_const = [1] + last_row.tolist()
    last_row_with_const = np.array(last_row_with_const).reshape(1, -1)
    predictions = self.model.predict(last_row_with_const)

    return predictions

def get_rolling_har_predictions(data, model_name='HAR', window_size=100):
    vol_forecast = []

    for i in range(window_size, len(data)):
        window_data = data[i - window_size : i]
        model = HAR(name=model_name)
        model.fit(window_data)
        vol_forecast.append(model.predict(window_data))

    return np.array(vol_forecast).flatten()

def get_naive_predictions(data, window=100):
    return np.array(data['rv'][window:]).flatten()