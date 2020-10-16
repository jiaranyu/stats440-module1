#!/usr/bin/env python
# coding: utf-8

# In[582]:


#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
from collections import defaultdict
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
# In[45]:
from sklearn.linear_model import BayesianRidge, LinearRegression


def cal_time(x):
    return (x - pd.Timestamp('2020-01-01'))

def cal_average(agerange):
    
    if type(agerange) is str and '-' in agerange: 
        age = agerange.split('-')
        return (int(age[0])+int(age[1]))/2
    else:
        if type(agerange) == float and math.isnan(agerange):
            return 50
        return agerange

def label_helper(series):
    dic = {}
    counter = 0
    for s in series:
        if s in dic.keys():
            continue
        else:
            dic[s] = str(counter)
            counter += 1
    return dic

def convert_label(df, headers):
    for i in range(len(headers)):
        h = headers[i]
        df[h] = df[h].apply(label_helper(df[h]).get)

    
train = pd.read_csv('train_clean1.csv')
test = pd.read_csv('test_clean1.csv')

train['confirmed'] = pd.to_datetime(train['confirmed'], format='%d.%m.%Y')
train['confirmed'] = train['confirmed'].apply(cal_time).dt.days

test['confirmed'] = pd.to_datetime(test['confirmed'], format='%d.%m.%Y')
test['confirmed'] = test['confirmed'].apply(cal_time).dt.days
label_transform('sex')
label_transform('city')
label_transform('province')
label_transform('country')
label_transform('V1')
train['confirmed'] = train['confirmed'].apply(cal_average)
test['confirmed'] = test['confirmed'].apply(cal_average)


X_train, X_test, y_train, y_test = train_test_split(train.loc[:, (train.columns != 'duration') 
                                                              & (train.columns != 'Id') 
                                                            
 ], train['duration'], 
                                                    test_size=0.005, random_state=200)
test_train = test.loc[:,  (test.columns != 'Id')]

kernel = DotProduct() +1 * RBF(1.0)
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)
print('gpr score: '+ str(gpr.score(X_train, y_train)))
predict_result_gussian = gpr.predict(X_test, return_std=True)


regr = svm.SVR(kernel='linear').fit(X_train, y_train)
predict_result = regr.predict(X_test)
mse = (mean_squared_error(list(predict_result_gussian[0]), y_test))**(1/2)

predict_result_gussian = gpr.predict(test_train, return_std=True)
result_df = pd.DataFrame(columns=['Id', 'duration'])
result_df['Id'] = test['Id']+1
result_df['duration'] = predict_result_gussian[0]
result_df.to_csv('result_guassian.csv',index=False)


