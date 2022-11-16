#
# CareyAnne Howlett
# DS5500:Capstone
# Northeastern University
#
# Last updated: 07/02/22
#
# This file is for exploritory data analysis 

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
from collections import Counter
from datetime import datetime 

# column names
# Ozone (Parts per million)
# Nitrogen dioxide (NO2) (Parts per billion)
# Wind Direction - Resultant (Degrees Compass)
# Wind Speed - Resultant (Knots)
# Barometric pressure (Millibars)
# Dew Point (Degrees Fahrenheit)
# Relative Humidity  (Percent relative humidity)
# Outdoor Temperature (Degrees Fahrenheit)

data = pd.read_csv('data/final_1.1.csv')


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

# this is the westport, ct site
#ct = sites.loc[(sites['County Name'] == 'Fairfield') & (sites['Site Num'] == 9003)]

#print(np.unique(ct['Site Num']))

# print(ct['Date Local'] + ' ' + ct['Time Local'])
# sys.exit()
#plt.scatter(sites['Longitude'], sites['Latitude'], color = 'red')
#plt.show()

#latsu = np.unique(data['Latitude'])
#lonsu = np.unique(data['Longitude'])

#print(len(latsu), len(lonsu))

print('Shape of dataframe: ', data.shape)
#print('Number of sites: ', len(lats))
print('Missing values by column:\n', data.isnull().sum())


#print('+++++++++++++++++++++++++++++')

#datar = data.dropna(subset = ['Ozone (Parts per million)'])

#print('Shape of dataframe: ', datar.shape)
#print('Number of sites: ', len(lats))
#print('Missing values by column:\n', datar.isnull().sum())


#dt = data['Date Local'] + " " + data['Time Local']
#data['date/time'] = pd.to_datetime(dt)
#data['date/time'] = data['date/time'].apply(lambda x: datetime.strptime(x, '%-m/%-d/%Y %H:%M'))


sys.exit()
data = data.groupby('Site Num')

#print(data.isnull().sum())


#print(data.groups.keys())
keys = [41, 48, 110, 9003]

#print(data.get_group(keys[0]))

for i in range(0, 4):
    g = data.get_group(keys[i])

    o3 = g['Ozone (Parts per million)'].isnull() * 1
    no2 = g['Nitrogen dioxide (NO2) (Parts per billion)'].isnull() * 2
    windd = g['Wind Direction - Resultant (Degrees Compass)'].isnull() * 3
    winds = g['Wind Speed - Resultant (Knots)'].isnull() * 4
    p = g['Barometric pressure (Millibars)'].isnull() * 5
    #dp = data['Dew Point (Degrees Fahrenheit)'] * 6
    rh = g['Relative Humidity  (Percent relative humidity)'].isnull() * 7
    temp = g['Outdoor Temperature (Degrees Fahrenheit)'].isnull() * 8


    plt.scatter(g['date/time'], o3, label = 'o3', s = 0.1)
    plt.scatter(g['date/time'], no2, label = 'no2', s = 0.1)
    plt.scatter(g['date/time'], windd, label = 'wind d', s = 0.1)
    plt.scatter(g['date/time'], winds, label = 'wind s', s = 0.1)
    plt.scatter(g['date/time'], p, label = 'press', s = 0.1)
    plt.scatter(g['date/time'], rh, label = 'rel hum', s = 0.1)
    plt.scatter(g['date/time'], temp, label = 'temp', s = 0.1)
    plt.legend()
    plt.xticks(rotation = 45)
    plt.title('Is Null: {}'.format(keys[i]))
    plt.show()