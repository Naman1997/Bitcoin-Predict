# First step, import libraries.
import numpy as np
import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import glob
from sklearn import linear_model
from datetime import datetime, date
import pmdarima as pm
from scipy import stats
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from itertools import product
import seaborn as sns
import warnings
from sklearn import preprocessing
import pickle
# Import the dataset and encode the date

# RNN
coin = glob.glob('coinbase*.csv')
df = pd.read_csv(coin[0])
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()

#Linear Regression
# dados = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv',',').dropna()
# dados

def RNNpredictions(time =30):
    from tensorflow import keras
    regressor = keras.models.load_model('RNN.h5')

    # split data
    prediction_days = time
    df_train= Real_Price[:len(Real_Price)-prediction_days]
    df_test= Real_Price[len(Real_Price)-prediction_days:]

    # Data preprocess
    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0:len(training_set)-1]
    y_train = training_set[1:len(training_set)]
    X_train = np.reshape(X_train, (len(X_train), 1, 1))

    # Making the predictions
    test_set = df_test.values
    inputs = np.reshape(test_set, (len(test_set), 1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicted_BTC_price = regressor.predict(inputs)
    predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

    # Visualising the results
    # plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
    # ax = plt.gca()  
    # plt.plot(test_set, color = 'red', label = 'Real BTC Price')
    # plt.plot(predicted_BTC_price, color = 'blue', label = 'Predicted BTC Price')
    # plt.title('BTC Price Prediction', fontsize=40)
    # df_test = df_test.reset_index()
    # x=df_test.index
    # labels = df_test['date']
    # plt.xticks(x, labels, rotation = 'vertical')
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label1.set_fontsize(18)
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label1.set_fontsize(18)
    # plt.xlabel('Time', fontsize=40)
    # plt.ylabel('BTC Price(USD)', fontsize=40)
    # plt.legend(loc=2, prop={'size': 25})
    # plt.savefig("done.png")
    dd = predicted_BTC_price.reshape(-1)

    return np.around(dd,2)

def RNNtestdata(time = 30):

    # split data
    prediction_days = time
    df_train= Real_Price[:len(Real_Price)-prediction_days]
    df_test= Real_Price[len(Real_Price)-prediction_days:]

    # Data preprocess
    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0:len(training_set)-1]
    y_train = training_set[1:len(training_set)]
    X_train = np.reshape(X_train, (len(X_train), 1, 1))

    # Sending testing data
    test_set = df_test.values

    return np.around(test_set,2)
    
def LRpredictions(time =30):
  df = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",delimiter= ',')
  df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit = 's')
  df.set_index('Timestamp', inplace=True)
  df = df.resample('D').mean().dropna()
  df.reset_index(inplace = True)
  le = preprocessing.LabelEncoder()
  le.fit(df['Timestamp'])
  x = le.transform(df['Timestamp']).reshape(-1,1)
  y = df['Weighted_Price'].to_list()
  modelo=linear_model.LinearRegression()
  modelo.fit(x, y)
  return modelo.predict(x).ravel().tolist()[-time:]

def LRtest(time =30):
  df = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",delimiter= ',')
  df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit = 's')
  df.set_index('Timestamp', inplace=True)
  df = df.resample('D').mean().dropna()
  x = df['Weighted_Price'][-time:].to_list()
  return x

def SARIMAtest(time =30):
    df = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",delimiter= ',')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit = 's')
    df.set_index('Timestamp', inplace=True)
    df = df.resample('D').mean()
    df4 = df['Weighted_Price'][-time:]
    return df4.to_list()

def SARIMApred(time =30):
    df = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv",delimiter= ',')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit = 's')
    df.set_index('Timestamp', inplace=True)
    df = df.resample('D').mean()
    df1 = df.iloc[:-time]
    filename = 'Sarima.pickle'
    loaded = pickle.load(open(filename, 'rb'))
    start = len(df1.axes[0])
    end = start + time -1
    df3 = loaded.predict(start = start, end = end)
    return df3.to_list()



# print(RNNpredictions(30))
# print(RNNtestdata(30))
# print()
# print("Linear Regression")
# print(LRpredictions(30)[1])
# print("END")