#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
#
# Last Updated: 06/10/22

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import sys
from pyproj import Proj, transform 
from glob import glob
from csv import writer

def initial(og):
    sub_og = og.loc[og['Latitude'].isin([6, 11, 18, 25, 27, 29, 43, 44, 75, 76, 124, 133, 1010, 9003])]

    para_og = np.unique(sub_og['Parameter Name'])
    unit_og = np.unique(sub_og['Units of Measure'])

    sub_og.columns = sub_og.columns.str.replace('Sample Measurement', "%s (%s)"%(para_og[0], unit_og[0]))
    sub_og = sub_og.drop(columns = ['Units of Measure', 'Parameter Name', 'Parameter Code', 'Method Type',\
    'Method Name', 'Method Code', 'MDL', 'Uncertainty', 'Qualifier', 'Date of Last Change', 'Datum'])

    return sub_og

def add_new_col(og, tmp, para, unit):
    sub = tmp.loc[tmp['Latitude'].isin([6, 11, 18, 25, 27, 29, 43, 44, 75, 76, 124, 133, 1010, 9003])]

    sub.columns = sub.columns.str.replace('Sample Measurement', "%s (%s)"%(para, unit))

    sub2 = sub[['Date Local', 'Time Local', 'Site Num', '%s (%s)'%(para, unit), 'State Code', 'County Code',\
        'Latitude', 'Longitude', 'Date GMT', 'Time GMT', 'State Name', 'County Name']]

 
    merged = pd.merge(og, sub2, on = ['Date Local', 'Time Local', 'Site Num'])
    merged.to_csv('data/raw_data/test/test.csv', index = False)

    del merged, sub, og, sub2

files = glob('data/raw_data/test/hourly*.csv')
og = pd.read_csv(files[0])
og_20 = initial(og)

del og

og = pd.read_csv(files[1])

og_21 = initial(og)

cat = pd.concat([og_20, og_21], ignore_index = True, axis = 0)

cat.to_csv('data/raw_data/test/test.csv', index = False)

del cat, og_20, og_21, og

sys.exit()

for i in range(2, len(files)):
    og = pd.read_csv('data/raw_data/test/test.csv')
    data = pd.read_csv(files[i])

    para = np.unique(data['Parameter Name'])
    unit = np.unique(data['Units of Measure'])

    if len(para) > 1:
        for j in range(0, len(para)):
            if j > 0:
                og = pd.read_csv('data/raw_data/test/test.csv')
            data_g = data.groupby('Parameter Name')
            add_new_col(og, data_g.get_group(para[j]), para[j], unit[j])
    else:
        add_new_col(og, data, para[0], unit[0])

final = pd.read_csv('data/raw_data/test/test.csv')


