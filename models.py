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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import f1_score, confusion_matrix
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#  removing dew point, RH and pressure because all data is missing 
data = pd.read_csv('data/final_1.1.csv').drop(['Relative Humidity  (Percent relative humidity)', \
    'Dew Point (Degrees Fahrenheit)', 'Barometric pressure (Millibars)'], axis = 1)
# only looking at the westport site to begin with 
sub = data.loc[(data['County Name'] == 'Fairfield') & (data['Site Num'] == 9003) & data['Exceedance'].notnull()]

#sub['Date/Time Local'] = sub['Date Local'] + sub['Time Local']
#print(sub['Date/Time Local'])

# filling in na's for now
#sub['Ozone (Parts per million)'] = sub['Ozone (Parts per million)'].fillna(sub['Ozone (Parts per million)'].mean())
#sub['Nitrogen dioxide (NO2) (Parts per billion)'] = sub['Nitrogen dioxide (NO2) (Parts per billion)'].fillna(sub['Nitrogen dioxide (NO2) (Parts per billion)'].mean())
#sub['Wind Direction - Resultant (Degrees Compass)'] = sub['Wind Direction - Resultant (Degrees Compass)'].fillna(sub['Wind Direction - Resultant (Degrees Compass)'].mean())
#sub['Wind Speed - Resultant (Knots)'] = sub['Wind Speed - Resultant (Knots)'].fillna(sub['Wind Speed - Resultant (Knots)'].mean())
#sub['Outdoor Temperature (Degrees Fahrenheit)'] = sub['Outdoor Temperature (Degrees Fahrenheit)'].fillna(sub['Outdoor Temperature (Degrees Fahrenheit)'].mean())

# removing nas
sub = sub.drop(['Ozone (Parts per million)', 'County Name', 'State Name', 'Date Local', 'Time Local', 'Site Num'], axis = 1).dropna()

# dealing with imbalanced data
class_1 = sub[sub['Exceedance'] == 1]
class_0 = sub[sub['Exceedance'] == 0]

#plt.bar(['<= 50 ppb Events', '> 50 ppb Events'], [len(class_0['Exceedance']), len(class_1['Exceedance'])])
#plt.title('Distribution of Ozone Binary Classes')
#plt.ylabel('Frequency')
#plt.show()


class_0_under = class_0.sample(class_1.shape[0])
class_1_over = class_1.sample(class_0.shape[0], replace = True)

test_under = pd.concat([class_1, class_0_under], axis = 0)
test_over = pd.concat([class_1_over, class_0], axis = 0)

# setting up dataset for train/test split

# imbalanced
x_i = sub.drop(['Exceedance'], axis = 1)
y_i = sub['Exceedance']


def preformance(x_train, y_train, x_test, y_test, mod):
    model = mod.fit(x_train, y_train)
    preds_train = mod.predict(x_train)
    preds_test = mod.predict(x_test)

    print('F1 Score (train): ', f1_score(y_train, preds_train))
    print('F1 Score (test): ', f1_score(y_test, preds_test))

# ++++++++++++++++++++++++ USING SMOTE TO CREATE A BALANCED DATASET ++++++++++++++++++

sm = SMOTE(random_state = 42)
x, y = sm.fit_resample(x_i, y_i)

x_train_i, x_test_i, y_train_i, y_test_i = train_test_split(x, y, test_size = 0.2)

print('Logistic Regression')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, LogisticRegression())

print('Decision Tress')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, DecisionTreeClassifier())

print('SVM')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, SVC())

print('KNN')
preformance(x_train_i, y_train_i, x_test_i, y_test_i, KNeighborsClassifier())

sys.exit()

# ++++++++++++++++++++++ USING UNDER AND OVER SAMPLING ++++++++++++++++++++++++++++++++
# oversampling and undersampling to see which is better

# undersampling
x_u = test_under.drop(['Exceedance'], axis = 1)
y_u = test_under['Exceedance']

# oversampling
x_o = test_over.drop(['Exceedance'], axis = 1)
y_o = test_over['Exceedance']

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