#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 07/26/22

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
data = pd.read_csv('data/final_1.3.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)'], axis = 1)
# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003) & (data['Ozone (Parts per million)'] >= 0)]

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'].astype('str'))
sub = sub.set_index('date_time', drop = True)

# removing nas
sub = sub.drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num', 'Exceedance_mc', 'Exceedance_bc'], axis = 1).dropna().sample(frac = 1).reset_index(drop = True)

# checking for colinearity
#print(sub[sub.columns[1:]].corr()['Outdoor Temperature (Degrees Fahrenheit)'][:])

y = sub['Ozone (Parts per million)']
x = sub.drop('Ozone (Parts per million)', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

lr = LinearRegression().fit(x_train, y_train)
lr_preds_t = lr.predict(x_train)
lr_preds = lr.predict(x_test)
# lr_r2 = r2_score(y_test, lr_preds)
# lr_r2_t = r2_score(y_train, lr_preds_t)
lr_r2 = math.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2_t = math.sqrt(mean_squared_error(y_train, lr_preds_t))
print(lr_r2_t, lr_r2)

rf = RandomForestRegressor().fit(x_train, y_train)
rf_preds = rf.predict(x_test)
rf_preds_t = rf.predict(x_train)
# rf_r2 = r2_score(y_test, rf_preds)
# rf_r2_t = r2_score(y_train, rf_preds_t)
rf_r2 = math.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2_t = math.sqrt(mean_squared_error(y_train, rf_preds_t))
print(rf_r2_t, rf_r2)

dt = DecisionTreeRegressor().fit(x_train, y_train)
dt_preds = dt.predict(x_test)
dt_preds_t = dt.predict(x_train)
# dt_r2 = r2_score(y_test, dt_preds)
# dt_r2_t = r2_score(y_train, dt_preds_t)
dt_r2 = math.sqrt(mean_squared_error(y_test, dt_preds))
dt_r2_t = math.sqrt(mean_squared_error(y_train, dt_preds_t))
print(dt_r2_t, dt_r2)

gb = GradientBoostingRegressor().fit(x_train, y_train)
gb_preds = gb.predict(x_test)
gb_preds_t = gb.predict(x_train)
# gb_r2 = r2_score(y_test, gb_preds)
# gb_r2_t = r2_score(y_train, gb_preds_t)
gb_r2 = math.sqrt(mean_squared_error(y_test, gb_preds))
gb_r2_t = math.sqrt(mean_squared_error(y_train, gb_preds_t))
print(gb_r2_t, gb_r2)

# plt.plot(y_test.reset_index(drop = True), label = 'True')
# #plt.plot(lr_preds, label = 'Linear Regression')
# plt.plot(gb_preds, label = 'Grad Boosting')
# plt.legend()
# plt.show()

sys.exit()
arr = np.array(ozone).reshape(len(ozone))

x = 7
y = 3
window = x + y

dataset = np.empty((len(arr), window))

for i in range(len(arr)):
    dataset[i] = arr[i: i + window]

np.random.shuffle(dataset)



