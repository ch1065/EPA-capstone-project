#
# Created by CareyAnne Howlett
# DS5500:Capstone
# Northeastern University 
# 
# Last Updated: 08/04/22

# Ozone (Parts per million)
# Nitrogen dioxide (NO2) (Parts per billion)
# Wind Direction - Resultant (Degrees Compass)
# Wind Speed - Resultant (Knots)
# Barometric pressure (Millibars)
# Dew Point (Degrees Fahrenheit)
# Relative Humidity  (Percent relative humidity)
# Outdoor Temperature (Degrees Fahrenheit)

# This file uses Westport & Philly data to predict the binary classification of the ozone threshold
#   Using SMOTE to deal with imbalanced data

# +++++++++++++++++++++ BINARY CLASSIFICATION WITH WESTPORT & PHILLY ++++++++++++++++++++++++++++++

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#  removing dew point, RH and pressure because all data is missing 
data = pd.read_csv('data/final_1.3.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)'], axis = 1)
# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003) & data['Exceedance_bc'].notnull()]

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

# ++++++++++++ ADDING IN PHILLY DATA ++++++++++++++++

sub = sub.drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'], axis = 1).dropna()

bronx = data.loc[(data['County Name'] == 'Philadelphia') & (data['Site Num'] == 48) & (data['Ozone (Parts per million)'] >= 0)]

bronx['date_time'] = bronx['Date Local'] + ' ' + bronx['Time Local']
bronx['date_time'] = pd.to_datetime(bronx['date_time'].astype('str'))
bronx = bronx.set_index('date_time', drop = True).drop(['County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'], axis = 1)

merged = sub.merge(bronx, on = 'date_time')

# removing nas
sub = merged.dropna().reset_index(drop = True)


# removing nas and features not needed -- multiclassification
sub = sub.drop(['Latitude_x', 'Longitude_x', 'Latitude_y', 'Longitude_y', 'Ozone (Parts per million)_x', 'Exceedance_mc_y', 'Exceedance_bc_x', 'Exceedance_bc_y'], axis = 1).dropna()

# use for binary classification
#sub = sub.drop(['Latitude_x', 'Longitude_x', 'Latitude_y', 'Longitude_y', 'Ozone (Parts per million)_x', 'Exceedance_mc_y', 'Exceedance_mc_x', 'Exceedance_bc_y'], axis = 1).dropna()

#plt.bar(['<= 50 ppb Events', '> 50 ppb Events'], [len(class_0['Exceedance']), len(class_1['Exceedance'])])
#plt.title('Distribution of Ozone Binary Classes')
#plt.ylabel('Frequency')
#plt.show()

def preformance(x_train, y_train, x_test, y_test, mod, set):
    """
    x_train: the x parameters of the training set
    y_train: the y parameter (classification value) for the training set
    x_test: the x parameters of the test set
    y_test: the y parameter (classification value) for the test set
    mod: the model being run
    set: to define the dataset being used--imbalanced vs balanced 

    function that calculates the preformance (using F1 score) of the model on both training and test sets 
    """
    print(set)
    mod.fit(x_train, y_train)
    preds_train = mod.predict(x_train)
    preds_test = mod.predict(x_test)

    # used for binary classification
    #print('F1 Score (train): ', f1_score(y_train, preds_train))
    #print('F1 Score (test): ', f1_score(y_test, preds_test))

    # used for multiclassification
    print('F1 Score (train): ', f1_score(y_train, preds_train, average = 'micro'))
    print('F1 Score (test): ', f1_score(y_test, preds_test, average = 'micro'))

# setting up dataset for train/test split
# imbalanced for binary classification
#x_i = sub.drop(['Exceedance_bc_x'], axis = 1)
#y_i = sub['Exceedance_bc_x']

# use for multiclassification 
x_i = sub.drop(['Exceedance_mc_x'], axis = 1)
y_i = sub['Exceedance_mc_x']

# ++++++++++++++++++++++++ USING SMOTE TO DEAL WITH IMBALANCED DATA ++++++++++++++++++

# use for binary classification
#sm = SMOTE(sampling_strategy=0.2) # used for binary classification

# use for multiclassification
strat = {0:12028, 1:12028, 2:12028}
sm = SMOTE(sampling_strategy = strat)
x, y = sm.fit_resample(x_i, y_i)

plt.bar(['<= 50 ppb Events', '> 50 ppb Events'], [len(x), len(y)])
plt.title('Distribution of Ozone Binary Classes After SMOTE')
plt.ylabel('Frequency')
#plt.show()

# train/test split of the imbalanced data
x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(x_i, y_i, test_size = 0.2)
# train/test split of the balanced data (balanced using SMOTE)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# running models
print('+++ Logistic Regression +++')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, LogisticRegression(), 'Imbalanced Set')
preformance(x_train, y_train, x_test, y_test, LogisticRegression(), 'Balanced Set')

print('+++ Decision Trees +++')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, DecisionTreeClassifier(), 'Imbalanced Set')
preformance(x_train, y_train, x_test, y_test, DecisionTreeClassifier(), 'Balanced Set')

print('+++ SVM +++')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, SVC(), 'Imbalanced Set')
preformance(x_train, y_train, x_test, y_test, SVC(), 'Balanced Set')

print('+++ KNN +++')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, KNeighborsClassifier(), 'Imbalanced Set')
preformance(x_train, y_train, x_test, y_test, KNeighborsClassifier(), 'Balanced Set')





sys.exit()

# ++++++++++++++++++++++ USING UNDER AND OVER SAMPLING ++++++++++++++++++++++++++++++++
# oversampling and undersampling to see which is better

# undersampling
x_u = test_under.drop(['Exceedance_x'], axis = 1)
y_u = test_under['Exceedance_x']

# oversampling
x_o = test_over.drop(['Exceedance_x'], axis = 1)
y_o = test_over['Exceedance_x']

x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(x_i, y_i, test_size = 0.2)
x_train_u, x_test_u, y_train_u, y_test_u = train_test_split(x_u, y_u, test_size = 0.2)
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(x_o, y_o, test_size = 0.2)


print('++++++++++++++ IMBALANCED ++++++++++++++++++++')
print('Logistic Regression')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, LogisticRegression())

print('Decision Tress')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, DecisionTreeClassifier())

print('SVM')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, SVC())

print('KNN')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, KNeighborsClassifier())

print('++++++++++++++ UNDERSAMPLING ++++++++++++++++++++')
print('Logistic Regression')
preformance(x_train_u, y_train_u, x_test_u, y_test_u, LogisticRegression())

print('Decision Tress')
preformance(x_train_u, y_train_u, x_test_u, y_test_u, DecisionTreeClassifier())

print('SVM')
preformance(x_train_u, y_train_u, x_test_u, y_test_u, SVC())

print('KNN')
preformance(x_train_u, y_train_u, x_test_u, y_test_u, KNeighborsClassifier())

print('++++++++++++++ OVERSAMPLING ++++++++++++++++++++')
print('Logistic Regression')
preformance(x_train_o, y_train_o, x_test_o, y_test_o, LogisticRegression())

print('Decision Tress')
preformance(x_train_o, y_train_o, x_test_o, y_test_o, DecisionTreeClassifier())

print('SVM')
preformance(x_train_o, y_train_o, x_test_o, y_test_o, SVC())

print('KNN')
preformance(x_train_o, y_train_o, x_test_o, y_test_o, KNeighborsClassifier())

sys.exit()

#lr_i = LogisticRegression().fit(x_train_i, y_train_i)
#preds_i = lr_i.predict(x_test_i)

#dtree = DecisionTreeClassifier().fit(x_train_i, y_train_i)
#preds_i = dtree.predict(x_test_i)

#svm_i = SVC().fit(x_train_i, y_train_i)
#preds_i = svm_i.predict(x_test_i)

knn_i = KNeighborsClassifier().fit(x_train_i, y_train_i)
preds_i = knn_i.predict(x_test_i)
preds_it = knn_i.predict(x_train_i)

print('F1 Score (imbalanced): ', f1_score(y_test_i, preds_i))
print('F1 Score (training-imbalanced): ', f1_score(y_train_i, preds_it))
print('F1 Score (undersampling):')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

#lr_u = LogisticRegression().fit(x_train_u, y_train_u)
#preds_u = lr_u.predict(x_test_u)

#dtree = DecisionTreeClassifier().fit(x_train_u, y_train_u)
#preds_u = dtree.predict(x_test_u)

#svm_u = SVC().fit(x_train_u, y_train_u)
#preds_u = svm_u.predict(x_test_u)

knn_u = KNeighborsClassifier().fit(x_train_u, y_train_u)
preds_u = knn_u.predict(x_test_u)
preds_ut = knn_u.predict(x_train_u)

print('F1 Score (undersampling): ', f1_score(y_test_u, preds_u))
#print('Confusion Matric (undersampling): ', confusion_matrix(y_test_u, preds_u))
print('F1 Score (training-undersampling): ', f1_score(y_train_u, preds_ut))
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


#lr_o = LogisticRegression().fit(x_train_o, y_train_o)
#preds_o = lr_o.predict(x_test_o)

#dtree = DecisionTreeClassifier().fit(x_train_o, y_train_o)
#preds_o = dtree.predict(x_test_o)

#svm_o = SVC().fit(x_train_o, y_train_o)
#preds_o = svm_o.predict(x_test_o)

knn_o = KNeighborsClassifier().fit(x_train_o, y_train_o)
preds_o = knn_o.predict(x_test_o)
preds_ot = knn_o.predict(x_train_o)

print('F1 Score (oversampling): ', f1_score(y_test_o, preds_o))
#print('Confusion Matrix (oversampling): ', confusion_matrix(y_test_o, preds_o))
print('F1 Score (training-oversampling): ', f1_score(y_train_o, preds_ot))