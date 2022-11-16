#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 07/05/22

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import math

#  removing dew point, RH and pressure because all data is missing 
data = pd.read_csv('data/final_1.1.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)'], axis = 1)
# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003) & (data['Ozone (Parts per million)'] >= 0)]

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'])
sub = sub.set_index('date_time', drop = True).drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num', 'Exceedance'], axis = 1).dropna()

bronx = data.loc[(data['County Name'] == 'Philadelphia') & (data['Site Num'] == 48) & (data['Ozone (Parts per million)'] >= 0)]

bronx['date_time'] = bronx['Date Local'] + ' ' + bronx['Time Local']
bronx['date_time'] = pd.to_datetime(bronx['date_time'].astype('str'))
bronx = bronx.set_index('date_time', drop = True).drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num', 'Exceedance'], axis = 1)

# print(bronx.shape)
# print(bronx.dropna().shape)
# print(bronx.isnull().sum())

merged = sub.merge(bronx, on = 'date_time')


# removing nas
sub = merged.dropna().sample(frac = 1).reset_index(drop = True)

# checking for colinearity
#print(sub[sub.columns[1:]].corr()['Outdoor Temperature (Degrees Fahrenheit)'][:])

y = sub['Ozone (Parts per million)_x']
x = sub.drop('Ozone (Parts per million)_x', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

lr = LinearRegression().fit(x_train, y_train)
lr_preds_t = lr.predict(x_train)
lr_preds = lr.predict(x_test)
lr_rmse = math.sqrt(mean_squared_error(y_test, lr_preds))
lr_rmse_t = math.sqrt(mean_squared_error(y_train, lr_preds_t))
print(lr_rmse, lr_rmse_t)

rf = RandomForestRegressor().fit(x_train, y_train)
rf_preds = rf.predict(x_test)
rf_preds_t = rf.predict(x_train)
rf_rmse = math.sqrt(mean_squared_error(y_test, rf_preds))
rf_rmse_t = math.sqrt(mean_squared_error(y_train, rf_preds_t))
print(rf_rmse, rf_rmse_t)

dt = DecisionTreeRegressor().fit(x_train, y_train)
dt_preds = dt.predict(x_test)
dt_preds_t = dt.predict(x_train)
dt_rmse = math.sqrt(mean_squared_error(y_test, dt_preds))
dt_rmse_t = math.sqrt(mean_squared_error(y_train, dt_preds_t))
print(dt_rmse, dt_rmse_t)

gb = GradientBoostingRegressor().fit(x_train, y_train)
gb_preds = gb.predict(x_test)
gb_preds_t = gb.predict(x_train)
gb_rmse = math.sqrt(mean_squared_error(y_test, gb_preds))
gb_rmse_t = math.sqrt(mean_squared_error(y_train, gb_preds_t))
print(gb_rmse, gb_rmse_t)





sys.exit()
arr = np.array(ozone).reshape(len(ozone))

x = 7
y = 3
window = x + y

dataset = np.empty((len(arr), window))

for i in range(len(arr)):
    dataset[i] = arr[i: i + window]

np.random.shuffle(dataset)



