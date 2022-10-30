#%% LIBRARIES

import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import warnings

#%% GETTING DATA
#get the latest stock price data using Python: AAPL


today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=360)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('AAPL', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
# print(data.head())

#%% TRANSFORM DATA

data["Date"] = data.index
data = data[["Date", "Open", "High", 
             "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
# print(data.head())

data = data[["Date", "Close"]] #Only the closing value is needed
print(data.head())

#%% PLOTTING CLOSE PRICES

# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(15, 10))
# plt.plot(data["Date"], data["Close"])

#%% ARIMA (we have to figure out whether our data is stationary or seasonal)

result = seasonal_decompose(data["Close"], 
                            model='multiplicative', period = 30)
# fig = plt.figure()  
# fig = result.plot()  
# fig.set_size_inches(15, 10)

# pd.plotting.autocorrelation_plot(data["Close"])

# plot_pacf(data["Close"], lags = 100)

p, d, q = 5, 1, 6


model=sm.tsa.statespace.SARIMAX(data['Close'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
# print(model.summary())

#%% PREDICTIONS

predictions = model.predict(len(data), len(data)+10)
print(predictions)

#%%PLOTTING PREDICTIONS

data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")

