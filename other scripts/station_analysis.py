#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 06/30/22

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

data = pd.read_csv('data/final_1.1.csv')
#d1 = pd.read_csv('data/raw_data/hourly_44201_2020.csv')
#d2 = pd.read_csv('data/raw_data/hourly_44201_2021.csv')
#data = pd.concat([d1, d2], ignore_index = True, axis = 0)

#westport = data.loc[((data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003)) | ((data['County Name'] == 'Bronx') & (data['Site Num'] == 110)) | \
#    ((data['County Name'] == 'District of Columbia') & (data['Site Num'] == 41) | ((data['County Name'] == 'Philadelphia') & (data['Site Num'] == 48)))]

#st = data.loc[((data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003)) | (data['County Name'] == 'Bronx')]

westport = data.loc[(data['County Name'] == 'Fairfield') & (data['State Name'] == 'Connecticut')]

#print(np.unique(westport['Site Num']))


#sys.exit()
dates = pd.DataFrame({'date/time' : pd.date_range(start = '2020-01-01', end = '2022-01-01', freq = 'H')})

dt = westport['Date Local'] + " " + westport['Time Local']
westport['date/time'] = pd.to_datetime(dt)

m_data = dates.merge(westport.loc[westport['Site Num'] == 9003], on = ['date/time'])
m_data = m_data.set_index('date/time').combine_first(westport.loc[(westport['Site Num'] == 1123)].set_index('date/time'))
#m_data['Ozone (Parts per million)'][m_data['Ozone (Parts per million)'].isnull()] = westport.loc[westport['Site Num'] == 1123]['Ozone (Parts per million)']
#m_data = dates.merge([westport.loc[westport['Site Num'] == 9003], westport['Site Num'] == 1123], on = ['date/time'])
print(m_data.shape)

print(m_data.isnull().sum())

# methods to try
#'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'
#plt.plot(m_data['date/time'], m_data.interpolate(method = 'linear', limit_direction = 'forward', inplace = True)['Ozone (Parts per million)'])
plt.plot(m_data['Ozone (Parts per million)'])
plt.xticks(rotation = 45)
#plt.plot(m_data['date/time'], m_data['Sample Measurement'].interpolate(mothod = 'linear'))
plt.title('Hourly Ozone Concentrations (Jan 1 2020 - Dec 31 2021)')
plt.ylabel('Parts per million')
plt.show()

#print(np.unique(m_data['Exceedance'].dropna(), return_counts = True))