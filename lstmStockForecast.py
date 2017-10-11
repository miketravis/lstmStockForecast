#Libraries
import quandl
from datetime import datetime
from pandas import Series, concat, DataFrame
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


def get_stock(ticker, start, end, time_range):
	#Retrive Stock Data Using Quandl
    stock = quandl.get('WIKI/{}'.format(ticker), start_date=start, end_date=end)
    #Keep Only Adjusted Closing Price And Adjusted Volume
    stock = stock.iloc[:,10:12]
    #Resample For Given Time Period
    stock = stock.resample('{}'.format(time_range)).mean()
    stock.dropna(inplace=True)
    return stock.values

def preprocess(data):
    #Transform Data and Calculate Percent Change
    data = np.log(data)
    percent_changes = np.empty(data.shape)
    for i in range(1, len(data)):
        percent_changes[i,:] = (data[i,:] - data[i-1,:])/data[i-1,:]
    return percent_changes

def series_to_supervised(data, n_in, n_out):
    #Convert The Data To A Supervised Learning Format
    cols = []
    df = DataFrame(data)
    for i in range(0, n_in+n_out):
        cols.append(df.shift(-i))
    data = concat(cols, axis=1)
    data.dropna(inplace=True)
    return data.values

def scale(train,test):
    #Scale The Data Using MinMax
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def fit_lstm(train, batch_size, nb_epoch, input_neurons, hidden_neurons):
    #Fits The LSTM To The Training Data
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(input_neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(LSTM(hidden_neurons))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error'])
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False)
    return model

def forecast_lstm(model, batch_size, X):
    #Forecasts The Average Price Of The Next Time Period
    X = X.reshape(1, len(X), 1)
    prediction = model.predict(X, batch_size=batch_size)
    return prediction[0,0]

def inverse_scale(predict, X, scaler):
    #Inverse Scales The Data
    new_row = [x for x in X] + [predict]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inv_scale = scaler.inverse_transform(array)
    return inv_scale[0,-1]

def inverse_preprocess(orig_data, predict, previous):
    #Inverse Percent Change And Transformation To Obtain The Actual Value
    inv_change = (predict + 1)*np.log(orig_data[-previous,0])
    return np.exp(inv_change)

#Stock Ticker
ticker = 'FMC'
#Timeframe of stock price data to r
start = datetime(2012,7,1)
end = datetime(2017,8,31)
#Time range ['M', 'W', 'D', 'A', 'Q']
time_range = 'M'
#Number of time instances to test on
test_inst = 12
#Number of past stock prices and volumes to use to predict future prices
num_prev = 4
#Number of future stock prices to predict
num_pred = 1
#Number of epochs
epochs = 1000
#Number of input neurons
input_neurons = 4
#Number of hidden neurons
hidden_neurons = 8
#Batch Size
batch_size = 1
#Number of trials to execute
trials = 5

print('Retrieving Stock Data...')
data = get_stock(ticker, start, end, time_range)

print('Preprocessing and Formatting Data...')
data_preprocessed = preprocess(data)
data_formatted = series_to_supervised(data_preprocessed, num_prev, num_pred)
#Drop Last Volume Row As We Are Only Predicting Stock Price
data_formatted = data_formatted[:,:-1]
train, test = data_formatted[0:-test_inst], data_formatted[-test_inst:]
scaler, train, test = scale(train, test)

print('Fitting and Testing Model...')
error_scores = []
for i in range(trials):
    model = fit_lstm(train, batch_size, epochs, input_neurons, hidden_neurons)
    predictions = []
    for j in range(len(test)):
        X = test[j,0:-1]
        pred = forecast_lstm(model, batch_size, X)
        pred_invScale = inverse_scale(pred, X, scaler)
        pred_actual = inverse_preprocess(data, pred_invScale,len(test)+1-j)
        predictions.append(pred_actual)
    rmse = sqrt(mean_squared_error(data[-test_inst:,0], predictions))
    print('RMSE {0}: {1}'.format(i+1, rmse))
    error_scores.append(rmse)

results = DataFrame()
results['rmse'] = error_scores
print(results.describe())

print('Forecasting Next Periods Average Price...')
forecast = forecast_lstm(model, batch_size, data_formatted[-1,num_pred:])
forecast_invScale = inverse_scale(forecast, data_formatted[-1,num_pred:], scaler)
forecast_actual = inverse_preprocess(data, forecast_invScale,len(test)+1-j)
predictions.append(forecast_actual)

print('Next Periods Forecasted Average Price: {}'.format(forecast_actual))

plt.plot(data[-test_inst:,0])
plt.plot(predictions, "o")
plt.show()