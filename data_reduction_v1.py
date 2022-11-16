#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University
# Last Update: 06/15/22
#
#   This script takes all the ozone, NO2, and meteorological datasets for the US during 2020 & 2021 and outputs a csv of the combined 
# subset of all the parameters in a tidy format 

import pandas as pd 
import numpy as np
from glob import glob
import os

def tidy(og):
    """

    input: (array) the dataframe of the original datafile 
    output: (array) reduced number of columns and tidy data

    """
    # selecting the stations of interest
    sub_og = og.loc[(og['Latitude'] < 45.592) & ((og['Latitude']) > 36.714) & (og['Longitude'] < -66.741) & (og['Longitude'] > -80.056)]
    # ulc 45.592, -80.056
    # lrc 36.714, -66.741

    # seeing how many variables are in the first file
    para_og = np.unique(sub_og['Parameter Name'])
    unit_og = np.unique(sub_og['Units of Measure'])

    # subsetting the dataset
    sub_og = sub_og[['State Name', 'County Name', 'Site Num', 'Latitude', 'Longitude', 'Date Local', \
        'Time Local', 'Sample Measurement', 'Parameter Name']]

    # for files that have more than 1 variable 
    if len(para_og) > 1:  
        # group the data by the parameter name
        data_g = sub_og.groupby('Parameter Name')
        # selecting the groups and columns we want to save
        new0 = data_g.get_group(para_og[0])[['State Name', 'County Name', 'Site Num', 'Latitude', \
            'Longitude', 'Date Local', 'Time Local', 'Sample Measurement']]
        new1 = data_g.get_group(para_og[1])[['State Name', 'County Name', 'Site Num', 'Latitude', \
            'Longitude', 'Date Local', 'Time Local', 'Sample Measurement']]
        # renaming the sample measurement column to the name of the sample and it's units
        new0.columns = new0.columns.str.replace('Sample Measurement', "%s (%s)"%(para_og[0], unit_og[0]))
        new1.columns = new1.columns.str.replace('Sample Measurement', "%s (%s)"%(para_og[1], unit_og[1]))

        # merging the two new columns
        sub_og = pd.merge(new0, new1, on = ['State Name', 'County Name', 'Site Num', 'Latitude', 'Longitude', 'Date Local', 'Time Local'])

    else:
        # if data doesn not hold more than 1 variable, the sample measurement column is renamed 
        sub_og.columns = sub_og.columns.str.replace('Sample Measurement', "%s (%s)"%(para_og[0], unit_og[0]))
        # dropping the parameter name column 
        sub_og = sub_og.drop(columns = ['Parameter Name'])
    # returning a tidy dataframe 
    return sub_og

# files that need to be combined
files = glob('data/raw_data/hourly*.csv')

# looping through files
for i in range(0, len(files), 2):
    # reading the first file (2020 of a givn parameter)
    og = pd.read_csv(files[i])
    # passing dataframe to the tidy function
    og_20 = tidy(og)
    # deleteing this from memory 
    del og
    # opening the second file (2021 of a given parameter)
    og = pd.read_csv(files[i + 1])
    # passing dataframe to the tidy function
    og_21 = tidy(og)
    # concatenating the 2020 and 2021 dataframes together
    cat = pd.concat([og_20, og_21], ignore_index = True, axis = 0)
    # writing the concatenated dataframe to a temporary csv file 
    cat.to_csv('data/raw_data/tmp_%s.csv'%(i), index = False)
    # delething these from memory 
    del cat, og_20, og_21, og

# getting the path of all the temporary files created
tmp = glob('data/raw_data/tmp*.csv')
f_len = len(tmp)
# writing the first file to a final csv file
f0 = pd.read_csv(tmp[0]).to_csv('data/final.csv', index = False)
os.remove(tmp[0])
# looping through the files 
for i in range(1, f_len):
    # open the final csv file so we can add to it
    final = pd.read_csv('data/final.csv')
    # open next temporary file 
    add = pd.read_csv(tmp[i])
    # preform an outer join on the final dataset and the dataset to add to the final dataset 
    m = pd.merge(final, add, on = ['State Name', 'County Name', 'Site Num', 'Latitude', \
        'Longitude', 'Date Local', 'Time Local'], how = 'outer')
    # writing the merged dataset to a csv 
    m.to_csv('data/final.csv',  index = False)
    os.remove(tmp[i])




