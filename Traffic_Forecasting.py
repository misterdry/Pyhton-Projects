#%% LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

#%% READ

data = pd.read_csv("Thecleverprogrammer.csv")
#print(data.head()) #Head means that prints the first (n) rows

data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y") #convert the Date column into Datetime data type
print(data.info()) #data information, The Date time column was an object initially, so I converted it into a Datetime column

#%% PLOTTING

# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(15, 10))
# plt.plot(data["Date"], data["Views"])
# plt.title("Daily Traffic of Thecleverprogrammer.com")
# plt.show()

#  Our website traffic data is seasonal because the traffic on the website increases
#  during the weekdays and decreases during the weekends. It is valuable to know if 
#  the dataset is seasonal or not while working on the problem of Time Series Forecasting

result = seasonal_decompose(data["Views"], 
                            model='multiplicative', 
                            extrapolate_trend='freq',
                            period = 30) #Period of the series. Must be used if x is not a pandas object or if the index of x does not have a frequency. Overrides default periodicity of x if x is a pandas object with a timeseries index.
# fig = plt.figure()  
# fig = result.plot()  
# fig.set_size_inches(15, 10)

#%% PREDICTION

# I will be using the Seasonal ARIMA (SARIMA) model to forecast traffic on the website. 
# Before using the SARIMA model, it is necessary to find the p, d, and q values. 
# As the data is not stationary, the value of d is 1. To find the values of p and q, 
# we can use the autocorrelation and partial autocorrelation plots:

# pd.plotting.autocorrelation_plot(data["Views"]) #In the above autocorrelation plot, the curve is moving down after the 5th line of the first boundary. That is how to decide the p-value.

# plot_pacf(data["Views"], lags = 100) #In the above partial autocorrelation plot, we can see that only two points are far away from all the points. That is how to decide the q value

p, d, q = 5, 1, 2
model=sm.tsa.statespace.SARIMAX(data['Views'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())

#%% RESULTS

predictions = model.predict(len(data), len(data)+50)
print(predictions)

data["Views"].plot(legend=True, label="Training Data", 
                   figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")

#Para tener una correcta visualización de los datos debemos ir plotteando las 
#gráficas por partes, para que no se sobrepongan.
