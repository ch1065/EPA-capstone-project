#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 08/13/22



from pickletools import optimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima 
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import boxcox
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.preprocessing import normalize
import sys
import math
import warnings
warnings.filterwarnings('ignore')

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#  removing dew point, RH and pressure because all data is missing 
data = pd.read_csv('data/final_1.3.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)', 'Exceedance_bc', 'Exceedance_mc'], axis = 1)

# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Ozone (Parts per million)'] > 0)]
#sub = data.loc[(data['County Name'] == 'Philadelphia') & (data['Site Num'] == 48)]

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'].astype('str'))
sub = sub.set_index('date_time', drop = True)

dates = pd.DataFrame({'date_time' : pd.date_range(start = '2019-01-01', end = '2022-01-01', freq = 'H')}).set_index('date_time')


m_data = dates.merge(sub.loc[sub['Site Num'] == 9003], left_index = True, right_index = True)
sub = m_data.combine_first(sub.loc[(sub['Site Num'] == 1123)])

names = sub.columns.drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'])

# plt.plot(sub['Ozone (Parts per million)'])
# plt.xticks(rotation = 45)
# plt.title('Ozone readings from Westport, CT')
# plt.subplots_adjust(bottom = 0.15)
# plt.show()

# removing nas

sub = sub.drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'], axis = 1).dropna().reset_index(drop = True)


#sub['Ozone (Parts per million)'] = sub['Ozone (Parts per million)'] - sub['Ozone (Parts per million)'].rolling(window = 720).mean()

df_decomposed_add = seasonal_decompose(sub['Ozone (Parts per million)'], model='additive', period = 24) #4320
#df_decomposed_add = seasonal_decompose(sub['Ozone (Parts per million)'], period = 24)
#plt.rcParams.update({'figure.figsize': (8,8)}) #increase figure size
df_decomposed_add.plot()#.suptitle("Additive Model", fontsize=20) #plotting decomposed time-series

plt.show()

#df_og = sub.reset_index(drop = True).dropna()
df_og = sub

def ad_test(dataset):
    dftest = adfuller(dataset, autolag = 'AIC')
    print('1. ADF: ', dftest[0])
    print('2. p-value: ', dftest[1])
    print('3. Num of lags: ', dftest[2])
    print('4. Num of obs used for ADF reg & critical vals calc: ', dftest[3])
    print('5. Critical Values: ')
    for key, val, in dftest[4].items():
        print('\t', key, ': ', val)


# checking for stationality 
ad_test(df_og['Ozone (Parts per million)'])


#step_fit = auto_arima(df_og['Ozone (Parts per million)'], trace = True, suppress_warnings = True,\
#    error_action='ignore', stepwise = True, seasonal = True, m = 24)
#print(step_fit.summary())

# best so far: ARIMA(1,1,0)(2,0,1)[24] intercept   : AIC=-91876.930, Time=97.64 sec

#sys.exit()
# RUN THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#step_fit = auto_arima(df_og['Ozone (Parts per million)'], trace = True, suppress_warnings = True, seasonal = True, m = 24)#, \
#    start_p=1, start_q = 0, start_P = 2, start_Q = 1)

#print(step_fit.summary())
# run this in the morning 
# try (D = 1, d = 0), nothing to d an D

#sys.exit()

#df_og = df_og.sample(frac = 1).reset_index(drop = True)

print(df_og.shape)
train = df_og.iloc[:-48]
#x_train = train.drop('Ozone (Parts per million)')
#y_train['ozone'] = train['Ozone (Parts per million)']
test = df_og.iloc[-48:]
#x_test = test.drop('Ozone (Parts per million)')
y_test = test['Ozone (Parts per million)']

hwes = ExponentialSmoothing(train['Ozone (Parts per million)'], trend = 'mul', seasonal = 'mul', seasonal_periods = 24)
fitted = hwes.fit(optimized=True, use_brute=True)

preds_hwes = fitted.forecast(steps=48)
hwes_rmse = math.sqrt(mean_squared_error(y_test, preds_hwes))
print('HWES: ', hwes_rmse)


# (2, 0, 0)x(2, 1, 0, 24) 0.01811 --shuffled: 0.016075
# (1, 1, 0)x(2, 0, 1, 24) 0.016413 --shuffled: 0.018086

arima = ARIMA(train['Ozone (Parts per million)'], order = (1, 1, 0), seasonal_order=(2, 0, 1, 24))
arima_r = arima.fit()
preds_arima = arima_r.predict(start = len(train), end = (len(train) + len(test) - 1), type = 'levels')
arima_rmse = math.sqrt(mean_squared_error(test['Ozone (Parts per million)'], preds_arima))
print('sarimax: ', arima_rmse)


sys.exit()

df = df_og
rolling_mean = df.rolling(window = 14).mean()
rolling_std = df.rolling(window = 14).std()
plt.plot(df, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')


result = adfuller(df)
print(result[1])


plt.figure()
df_decomposed_add = seasonal_decompose(df_og, model='additive', period = 30)
plt.rcParams.update({'figure.figsize': (8,8)}) #increase figure size
df_decomposed_add.plot().suptitle("Additive Model", fontsize=20) #plotting decomposed time-series


df_log = np.log(df_og)
df_log_stationary = df_log - df_log.rolling(window=12).mean()

fig, ax = plt.subplots(2, figsize=(8,6))
ax[0] = plot_acf(df_og, ax=ax[0], lags=20)
df_log_stationary_diff = df_log_stationary - df_log_stationary.shift()
ax[1] = plot_acf(df_log_stationary_diff.dropna(), ax=ax[1], lags=20)

df_og = df_log_stationary

train = df_og.iloc[:len(df_og)-12] 
test = df_og.iloc[len(df_og)-12:]

model = SARIMAX(train).fit()
#print(model.summary())

plt.figure()
preds = model.predict(start = len(train), end = (len(train) + len(test) - 1)).rename('Preds')
plt.rcParams.update({'figure.figsize': (8,6)}) #increase figure size
# plot predictions and actual values 
preds.plot(legend = True) 
test.plot(legend = True) 
plt.show()


sys.exit()
 
x = sub.reset_index()['Ozone (Parts per million)']
#print(sub)


train = x[:(len(x) - 7)]
test = x[(len(x) - 7):]

ar = AutoReg(train, lags = 6).fit()

#print(ar.summary())

pred = ar.predict(start = len(train), end = (len(x) - 1), dynamic = False)


#print(len(train))
#print('+++++++++++++++++')
#print(test)

plt.plot(pred, label = 'pred')
plt.plot(test, color = 'red', label = 'Truth')
plt.legend()
#plt.show()

rmse = sqrt(mean_squared_error(test, pred))
r2 = r2_score(y_true=test, y_pred=pred)
adj_r2 = 1 - ((1 - r2) * (len(x) - 1) / (len(x) - 1 - 1))

print(rmse)
print(adj_r2)

sys.exit()
res = seasonal_decompose(sub['Ozone (Parts per million)'], model = 'additive', period = 30)
#fig, (ax1, ax2, ax3) = plt.subplot(3, 1, figsize = (15, 8))
res.trend.plot()
plt.title('trend')
plt.figure()
res.resid.plot()
plt.title('resid')
plt.figure()
res.seasonal.plot()
plt.title('seasonal')

plt.show()


