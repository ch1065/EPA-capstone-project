#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 08/06/22

from winreg import ExpandEnvironmentStrings
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

#  removing dew point, RH and pressure because all data is missing 
data = pd.read_csv('data/final_1.3.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)', 'Exceedance_bc', 'Exceedance_mc'], axis = 1)

# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Ozone (Parts per million)'])]
#sub = data.loc[(data['County Name'] == 'Philadelphia') & (data['Site Num'] == 48)]

#, parse_dates=[['Date Local', 'Time Local']]
#sub['Date Local_Time Local'] = pd.to_datetime(sub['Date Local_Time Local'])

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'].astype('str'))
sub = sub.set_index('date_time', drop = True)

dates = pd.DataFrame({'date_time' : pd.date_range(start = '2019-01-01', end = '2022-01-01', freq = 'H')}).set_index('date_time')

m_data = dates.merge(sub.loc[sub['Site Num'] == 9003], left_index = True, right_index = True)
sub = m_data.combine_first(sub.loc[(sub['Site Num'] == 1123)])

names = sub.columns.drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'])

# ++++++++++++++++++++++++++++++++ Long Short Term Memory (multivariate) ++++++++++++++++++++++++++++++++++++

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import seaborn as sns

print(sub.shape)


sys.exit()

# +++++++++++++++++++++++++++++++++++++++++ Holt Winters Exponential Smoothing ++++++++++++++++++++++++++++++++++++++++++

# plt.plot(sub['Ozone (Parts per million)'])
# plt.xticks(rotation = 45)
# plt.title('Ozone readings from Westport, CT')
# plt.subplots_adjust(bottom = 0.15)
# plt.show()

# removing nas

sub = sub.drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'], axis = 1).dropna().reset_index(drop = True)


# period of the time series (24 hours)
m = 24
alpha = 1/(2*m)

sub['HWES1'] = SimpleExpSmoothing(sub['Ozone (Parts per million)']).fit(smoothing_level = alpha, optimized = False, use_brute = True).fittedvalues

#sub[['Ozone (Parts per million)', 'HWES1']].plot(title = 'Holt Winters Single Exponential Smoothing')

sub['HWES2_ADD'] = ExponentialSmoothing(sub['Ozone (Parts per million)'], trend = 'add').fit().fittedvalues
sub['HWES2_MUL'] = ExponentialSmoothing(sub['Ozone (Parts per million)'], trend = 'mul').fit().fittedvalues

#sub[['Ozone (Parts per million)', 'HWES2_ADD']].plot(title = 'Holt Winters Doub;e Expential Smoothing: Additive and Multiplicative Trend')

#plt.plot(sub['Ozone (Parts per million)'])
#plt.plot(sub['HWES2_MUL'], alpha = 0.5)

sub['HWES3_ADD'] = ExponentialSmoothing(sub['Ozone (Parts per million)'], trend = 'add', seasonal = 'add', seasonal_periods = 24).fit().fittedvalues
sub['HWES3_MUL'] = ExponentialSmoothing(sub['Ozone (Parts per million)'], trend = 'mul', seasonal = 'mul', seasonal_periods = 24).fit().fittedvalues

#plt.plot(sub['Ozone (Parts per million)'])
#plt.plot(sub['HWES3_MUL'], alpha = 0.5)

#print(sub['Ozone (Parts per million)'].shape)
# size: (12878)

# test set is the last 1 weeks of data (168 hours)
#train = sub[:12710]
#test = sub[12710:]

# test set is the last 2 days of data (48 hours)
train = sub[:12830]
test = sub[12830:]

# test set is the last 1 day of data (24 hours)
#train = sub[:12854]
#test = sub[12854:]

fitted_model = ExponentialSmoothing(train['Ozone (Parts per million)'], trend = 'mul', seasonal = 'mul', seasonal_periods = 24).fit()
test_preds = fitted_model.forecast(48)

#train['Ozone (Parts per million)'].plot(legend = True, label = 'TRAIN')
test['Ozone (Parts per million)'].plot(legend = True, label = 'TEST')
test_preds.plot(legend = True, label = 'PREDS')

print(f'Root Mean Squared Error = {math.sqrt(mean_squared_error(test["Ozone (Parts per million)"], test_preds))}')

plt.show()

sys.exit()

data = pd.read_csv('data/air_passengers.csv', parse_dates=['Month'], index_col = ['Month'])

df = data
rolling_mean = df.rolling(window = 14).mean()
rolling_std = df.rolling(window = 14).std()
plt.plot(df, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')


result = adfuller(data['Passengers'])
print(result[1])

df_log = np.log(df)
df_log_stationary = df_log - df_log.rolling(window=12).mean()
plt.figure()
plt.plot(df_log_stationary)


df_log_stationary.dropna(inplace=True)
result = adfuller(df_log_stationary['Passengers'])
print('p-value: {}'.format(result[1]))


fig, ax = plt.subplots(2, figsize=(8,6))
ax[0] = plot_acf(data, ax=ax[0], lags=20)

# Calculate first-difference
df_log_stationary_diff = df_log_stationary - df_log_stationary.shift()
ax[1] = plot_pacf(df_log_stationary_diff.dropna(), ax=ax[1], lags=20)

auto = auto_arima(df_log_stationary_diff.dropna(), trace = True, suppress_warnings=True, start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P=0, seasonal = True,\
    d = None, D = 1, error_action='ignore', stepwise = True)
print(auto.summary())

plt.show()

sys.exit()
files = glob('data/raw_data/hourly*.csv')

# for f in files:
#     print(f)
#     data = pd.read_csv(f)
#     print(data.isnull().sum())
#     print('++++++++++++++++++++++++++++++++++++++++++')

data = pd.read_csv(files[0])

westport = data.loc[((data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003)) | ((data['County Name'] == 'Bronx') & (data['Site Num'] == 110)) | \
    ((data['County Name'] == 'District of Columbia') & (data['Site Num'] == 41) | ((data['County Name'] == 'Philadelphia') & (data['Site Num'] == 48)))]
#print(westport)


print(np.unique(westport['Site Num']))
print(np.unique(westport['County Name']))

#sys.exit()

lats = np.unique(westport['Latitude'])
lons = np.unique(westport['Longitude'])

#print(lats)

us = gpd.read_file('data/cb_2018_us_state_500k.shp')

us.plot()
plt.scatter(lons, lats, color = 'red')
plt.show()