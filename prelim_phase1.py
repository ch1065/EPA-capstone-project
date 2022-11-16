#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 07/25/22

# Ozone (Parts per million)
# Nitrogen dioxide (NO2) (Parts per billion)
# Wind Direction - Resultant (Degrees Compass)
# Wind Speed - Resultant (Knots)
# Barometric pressure (Millibars)
# Dew Point (Degrees Fahrenheit)
# Relative Humidity  (Percent relative humidity)
# Outdoor Temperature (Degrees Fahrenheit)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import sys
import matplotlib.pyplot as plt

#  removing dew point, RH and pressure because all data is missing 
data = pd.read_csv('data/final.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)'], axis = 1)
# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003) & data['Ozone (Parts per million)'].notnull()]

#sub['Date/Time Local'] = sub['Date Local'] + sub['Time Local']
#print(sub['Date/Time Local'])

# filling in na's for now
#sub['Ozone (Parts per million)'] = sub['Ozone (Parts per million)'].fillna(sub['Ozone (Parts per million)'].mean())
#sub['Nitrogen dioxide (NO2) (Parts per billion)'] = sub['Nitrogen dioxide (NO2) (Parts per billion)'].fillna(sub['Nitrogen dioxide (NO2) (Parts per billion)'].mean())
#sub['Wind Direction - Resultant (Degrees Compass)'] = sub['Wind Direction - Resultant (Degrees Compass)'].fillna(sub['Wind Direction - Resultant (Degrees Compass)'].mean())
#sub['Wind Speed - Resultant (Knots)'] = sub['Wind Speed - Resultant (Knots)'].fillna(sub['Wind Speed - Resultant (Knots)'].mean())
#sub['Outdoor Temperature (Degrees Fahrenheit)'] = sub['Outdoor Temperature (Degrees Fahrenheit)'].fillna(sub['Outdoor Temperature (Degrees Fahrenheit)'].mean())

# removing nas
sub = sub.drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'], axis = 1).dropna()


x = sub.drop(['Ozone (Parts per million)'], axis = 1)
y = sub['Ozone (Parts per million)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

lm = LinearRegression().fit(x_train, y_train)

pred = lm.predict(x_test)
pred_train = lm.predict(x_train)

print(math.sqrt(mean_squared_error(y_test, pred)))
print(math.sqrt(mean_squared_error(y_train, pred_train)))



