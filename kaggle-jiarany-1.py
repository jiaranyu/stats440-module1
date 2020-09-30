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

def load_df():
    train = pd.read_csv('train2.txt',  sep=',')
    test = pd.read_csv('test2.txt', sep=',')
    return train, test

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

def get_sympts_list(df):
    raw_symptoms_list = df['symptoms'].value_counts().keys()
    dic = set()
    for smps in raw_symptoms_list:
        for w in smps.split('; '):
            dic.add(w.lower())
    return dic
        
def get_filtered_sympts(dic):
    filtered_dict = set()
    for i in dic:
        if i in syptsMap.keys():
            filtered_dict.add(syptsMap[i])
        else:
            filtered_dict.add(i)
    return filtered_dict

    
syptsMap = {'fever (38-39 ° c)': 'high fever', 
            'fever (38-39 ℃)': 'high fever',
            'fever (39.5 ℃)': 'high fever',
            'fever 38.3': 'high fever',
            'low fever (37.2 ° c)': 'fever',
            'toothache':'fever',
            'fever (37 ℃)': 'fever',
           'low fever 37.0 ℃':'fever',
            'low fever (37.4 ℃)':'fever',
           'fever 37.7℃':'fever',
            'diarrhoea':'diarrhea',
            'abdominal pain':'diarrhea',
            'diarrheoa':'diarrhea',
            'shortness breath': 'breath',
            'shortness of breath':'breath',
            'anhelation':'breath',
            'difficulty breathing':'breath',
            'systemic weakness':'weak',
            'poor physical condition':'weak',
            'anorexia':'weak',
            'feeling ill':'weak',
            'physical discomfort':'weak',
            'malaise':'weak',
            'general malaise':'weak',
            'discomfort':'weak',
            'fatigue':'weak',
            'weakness':'weak',
            'pharyngeal dryness':'throat',
            'pharyngeal discomfort':'throat',
             'pharynx':'throat',
            'sore throat':'throat',
            'dry throat':'throat',
            'throat discomfort':'throat',
            'myalgia':'muscle',
            'muscle ache': 'muscle',
            'muscular soreness': 'muscle',
            'muscular stiffness': 'muscle',
            'sore body':'muscle',
            'joint pain':'muscle',
            'soreness':'muscle',
            'chest distress':'chest',
            'pleuritic chest pain':'chest',
            'chest tightness':'chest',
            'chest pain':'chest',
            'pneumonitis':'pneumonia',
            'dry cough':'cough',
            'other symptoms':'cough',
            'coughing':'cough',
            'nausea':'vomiting',
             'expectoration':'sputum',
            'sneezing':'sneeze',
            
            'rhinorrhoea':'rhinorrhea',
            'nasal congestion':'nose',
            'runny nose':'nose',
            'dyspnea': 'severe pneumonia',
           }

def label_transform(col):
    le = preprocessing.LabelEncoder()
    le.fit(list(train[col].value_counts().keys()) + list(test[col].value_counts().keys()))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    
    



# In[451]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[583]:


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


# In[581]:


mse


# In[578]:


# predict_result = regr.predict(test_train)

# result_df = pd.DataFrame(columns=['Id', 'duration'])
# result_df['Id'] = test['Id']+1
# result_df['duration'] = predict_result
# result_df.to_csv('result_svm.csv',index=False)


# In[579]:


predict_result_gussian = gpr.predict(test_train, return_std=True)
result_df = pd.DataFrame(columns=['Id', 'duration'])
result_df['Id'] = test['Id']+1
result_df['duration'] = predict_result_gussian[0]
result_df.to_csv('result_guassian.csv',index=False)


# In[587]:





# In[588]:





# In[589]:





# In[ ]:




