#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 08/13/22

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

def preformance(x_train, y_train, x_test, y_test, mod):
    """
    x_train: the x parameters of the training set
    y_train: the y parameter (classification value) for the training set
    x_test: the x parameters of the test set
    y_test: the y parameter (classification value) for the test set
    mod: the modeling being run

    function that calculates the preformance (using RMSE) of the model on both training and test sets
    """
    mod.fit(x_train, y_train)
    pred_train = mod.predict(x_train)
    pred_test = mod.predict(x_test)

    rmse_train = math.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = math.sqrt(mean_squared_error(y_test, pred_test))

    print('RMSE (train): ', rmse_train)
    print('RMSE (test): ', rmse_test)

y = sub['Ozone (Parts per million)']
x = sub.drop('Ozone (Parts per million)', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print('+++ Linear Regression +++')
preformance(x_train, y_train, x_test, y_test, LinearRegression())

print('+++ Random Forests Regressor +++')
preformance(x_train, y_train, x_test, y_test, RandomForestRegressor())

print('+++ Decision Tree Regressor +++')
preformance(x_train, y_train, x_test, y_test, DecisionTreeRegressor())

print('+++ Gradient Boosting Regressor +++')
preformance(x_train, y_train, x_test, y_test, GradientBoostingRegressor())





