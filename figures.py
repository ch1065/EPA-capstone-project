#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 08/13/22


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from imblearn.over_sampling import SMOTE
import sys

# +++++++++++++++++++++++++++ SMOTE balanced barplots ++++++++++++++++++++++++
data = pd.read_csv('data/final_1.3.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)'], axis = 1)
# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003) & data['Exceedance_bc'].notnull()]

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'].astype('str'))
sub = sub.set_index('date_time', drop = True)

# use for binary classification
sub = sub.drop(['Latitude', 'Longitude', 'Ozone (Parts per million)', 'Exceedance_mc',\
    'State Name', 'County Name', 'Site Num', 'Date Local', 'Time Local'], axis = 1).dropna()
x_i = sub.drop(['Exceedance_bc'], axis = 1)
y_i = sub['Exceedance_bc']

# multiclassification
#sub = sub.drop(['Latitude', 'Longitude', 'Ozone (Parts per million)', 'Exceedance_bc',\
#    'State Name', 'County Name', 'Site Num', 'Date Local', 'Time Local'], axis = 1).dropna()
#x_i = sub.drop(['Exceedance_mc'], axis = 1)
#y_i = sub['Exceedance_mc']

# SMOTE for multi class
#strat = {0:12028, 1:12028, 2:12028}
#sm = SMOTE(sampling_strategy = strat)
#x, y = sm.fit_resample(x_i, y_i)

# SMOTE for binary class
sm = SMOTE()
x, y = sm.fit_resample(x_i, y_i)

# for binary plot
#counts = np.unique(sub['Exceedance_bc'], return_counts=True)

# for multi class plot
counts = np.unique(y, return_counts = True)

plt.bar(np.arange(len(counts[0])), counts[1])
plt.ylabel('Frequency')
# for multiclass plot
#plt.title('Balanced AQI Classes using SMOTE')
#plt.xticks(np.arange(len(counts[0])), ['Good', 'Moderate', 'USG'])

# for binary plot
plt.title('Balanceed Exceedance vs. Non-Exceedance Events using SMOTE')
plt.xticks(np.arange(len(counts[0])), ['Non-Exceedance Event', 'Exceedance Event'])
plt.show()

sys.exit()

sys.exit()
# +++++++++++++++++++ bar plots to show imbalance +++++++++++++++++++++
data = pd.read_csv('data/final_1.3.csv')
data['date_time'] = data['Date Local'] + ' ' + data['Time Local']

sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Ozone (Parts per million)'] >= 0)]

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'].astype('str'))
sub = sub.set_index('date_time', drop = True)

dates = pd.DataFrame({'date_time' : pd.date_range(start = '2019-01-01', end = '2022-01-01', freq = 'H')}).set_index('date_time')

m_data = dates.merge(sub.loc[sub['Site Num'] == 9003], left_index = True, right_index = True)
sub = m_data.combine_first(sub.loc[(sub['Site Num'] == 1123)])

# for binary plot
#counts = np.unique(sub['Exceedance_bc'], return_counts=True)

# for multi class plot
counts = np.unique(sub['Exceedance_mc'], return_counts = True)

plt.bar(np.arange(len(counts[0])), counts[1])
plt.ylabel('Frequency')
# for multiclass plot
plt.title('Imbalance of AQI Classes')
plt.xticks(np.arange(len(counts[0])), ['Good', 'Moderate', 'USG'])

# for binary plot
#plt.title('Imbalance of Exceedance vs. Non-Exceedance Events')
#plt.xticks(np.arange(len(counts[0])), ['Non-Exceedance Event', 'Exceedance Event'])
plt.show()

sys.exit()
# +++++++++++++++++++++++++++++++ three years of data plot - filled in missing data ++++++++++++++++++++++
data = pd.read_csv('data/final_1.3.csv')
data['date_time'] = data['Date Local'] + ' ' + data['Time Local']

sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Ozone (Parts per million)'] >= 0)]

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'].astype('str'))
sub = sub.set_index('date_time', drop = True)

dates = pd.DataFrame({'date_time' : pd.date_range(start = '2019-01-01', end = '2022-01-01', freq = 'H')}).set_index('date_time')

m_data = dates.merge(sub.loc[sub['Site Num'] == 9003], left_index = True, right_index = True)
sub = m_data.combine_first(sub.loc[(sub['Site Num'] == 1123)])

plt.plot(sub['Ozone (Parts per million)'])
plt.xticks(rotation = 45)
plt.title('Ozone Concentration at the Westport, CT Site from 2019-2021\n(Filled in missing data will observations from Danbury, CT)')
plt.xlabel('Date/time')
plt.ylabel('Ozone Concentration')
plt.tight_layout()
plt.show()


sys.exit()
# ++++++++++++++++++++++++++++++++ 2 years of data plot - filled in missing data ++++++++++++++++++++
data = pd.read_csv('data/final_1.2.csv')
data['date_time'] = data['Date Local'] + ' ' + data['Time Local']

sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Ozone (Parts per million)'] >= 0)]

sub['date_time'] = sub['Date Local'] + ' ' + sub['Time Local']
sub['date_time'] = pd.to_datetime(sub['date_time'].astype('str'))
sub = sub.set_index('date_time', drop = True)

dates = pd.DataFrame({'date_time' : pd.date_range(start = '2019-01-01', end = '2022-01-01', freq = 'H')}).set_index('date_time')

m_data = dates.merge(sub.loc[sub['Site Num'] == 9003], left_index = True, right_index = True)
sub = m_data.combine_first(sub.loc[(sub['Site Num'] == 1123)])

plt.plot(sub['Ozone (Parts per million)'])
plt.xticks(rotation = 45)
plt.title('Ozone Concentration at the Westport, CT Site from 2020-2021\n(Filled in missing data will observations from Danbury, CT)')
plt.xlabel('Date/time')
plt.ylabel('Ozone Concentration')
plt.tight_layout()
plt.show()

sys.exit()
# ++++++++++++++++++++++++++++ plot of Westport ozone ++++++++++++++++++++++
data = pd.read_csv('data/final_1.2.csv')
data['date_time'] = data['Date Local'] + ' ' + data['Time Local']

data['date_time'] = pd.to_datetime(data['date_time'])
data = data.set_index(data['date_time']).sort_index()

sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003) & (data['Ozone (Parts per million)'] >= 0)]

plt.plot(sub['Ozone (Parts per million)'])
plt.xticks(rotation = 45)
plt.title('Ozone Concentration at the Westport, CT Site from 2020-2021')
plt.xlabel('Date/time')
plt.ylabel('Ozone Concentration')
plt.tight_layout()
plt.show()


sys.exit()
# ++++++++++++++++++++++++++++++ plot of site used +++++++++++++++++++++++

#sites = data.loc[(data['Date Local'] == '2020-01-01') & (data['Time Local'] == '01:00')]

#lats = [41.118333, 39.991389, 41.399167]
#lons = [-73.33667, -75.080833, -73.443056]

p1 = [-73.33667, 41.118333]
p2 = [-75.080833, 39.991389]
p3 = [-73.443056, 41.399167]

us = gpd.read_file('data/cb_2018_us_state_500k.shp')

us.plot()
#plt.scatter(lons, lats, color = 'red')
plt.scatter(p1[0], p1[1], label = 'Westport, CT', c = 'red')
plt.scatter(p2[0], p2[1], label = 'Philadelphia, PA', c = 'black')
plt.scatter(p3[0], p3[1], label = 'Danbury, CT', c = 'yellow')
plt.legend()
plt.show()

sys.exit()