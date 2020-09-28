#!/usr/bin/env python
# coding: utf-8

# In[191]:


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

train = pd.read_csv('train2.txt',  sep=',')
test = pd.read_csv('test2.txt', sep=',')


# In[192]:


def cal_average(agerange):
    
    if type(agerange) is str and '-' in agerange: 
        age = agerange.split('-')
        return (int(age[0])+int(age[1]))/2
    else:
        return agerange
    
def clean_age(df):
    df['age'].fillna(value= '1000',inplace=True)
    df['age'] = df['age'].apply(cal_average)
    temp_idx = df.loc[:,'age']=='1000'
    df.loc[temp_idx,'age'] = df.loc[temp_idx,'age'] = round(df.loc[~temp_idx,"age"].astype(float).mean(),0)
    


# In[193]:


clean_age(train)
clean_age(test)


# In[194]:


# def get_sympts_list(df):
#     raw_symptoms_list = df['symptoms'].value_counts().keys()
#     dic = set()
#     for smps in raw_symptoms_list:
#         for w in smps.split('; '):
#             dic.add(w.lower())
#     return dic
# all_symps = get_sympts_list(train).union(get_sympts_list(test))
# all_symps
train


# In[195]:


sympts = {}
high_fever = ['fever (38-39 ° c)', 'fever (38-39 ℃)','fever (39.5 ℃)','fever 38.3','high fever']
low_fever = ['low fever (37.2 ° c)','fever (37 ℃)','low fever 37.0 ℃','low fever (37.4 ℃)','fever 37.7℃','fever']
breath = ['severe dyspnea','dyspnea','difficulty breathing', 'anhelation', 'shortness of breath','shortness breath']
chest = ['chest tightness', 'chest distress','pleuritic chest pain','pleural effusion','chest pain']
throat = ['dry mouth','dry throat','throat discomfort', 'Sore throat','sore throat','acute pharyngitis','pharyngeal discomfort','Pharyngeal dryness','pharynx','pharyngeal discomfort','pharyngeal dryness']
diarrhea = ['abdominal pain', 'diarrheoa', 'diarrhea', 'diarrhea','diarrhoea']
cold = [ 'chills', 'cold', 'sneezing','sneeze']
nose = ['nasal congestion','runny nose','rhinorrhoea','rhinorrhea']
pneumonia = ['pneumonitis', 'pneumonia', 'severe pneumonia']
sputum = ['expectoration', 'sputum']
muscle = ['sore body','muscular soreness', 'muscle ache', 'myalgia','soreness', 'sore body','muscular stiffness','joint pain','soreness']
cough= ['cough','dry cough','coughing']
discomfort = ['other symptoms','malaise','discomfort' ,' malaise','feeling ill', 'anorexia','general malaise' ,'fatigue','physical discomfort']
weak = ['systemic weakness','poor physical condition', 'weakness', 'weak']
vomit = ['vomiting','nausea']
flu = ['flu-like symptoms','dizziness','headache','respiratory symptoms', 'toothache','conjunctivitis']


sympts['high_fever'] = high_fever
sympts['low_fever'] = low_fever
sympts['breath'] = breath
sympts['chest'] = chest
sympts['throat'] = throat
sympts['diarrhea'] = diarrhea
sympts['cold'] = cold
sympts['nose'] = nose
sympts['pneumonia'] = pneumonia
sympts['sputum'] = sputum
sympts['muscle'] = muscle
sympts['cough'] = cough
sympts['discomfort'] = discomfort
sympts['weak'] = weak
sympts['vomit'] = vomit
sympts['flu'] = flu


# In[196]:


# hello = set()
# for key in sympts.keys():
#     for i in sympts[key]:
#         hello.add(i)


# In[197]:


def clean_symp(df):
    df['symptoms'].fillna(value= 'no symptom',inplace=True)
    for key in sympts.keys():
        df[key] = 0
    df['no symptom'] = 0
    temp_idx = df.loc[:,'symptoms']=='no symptom'
    df.loc[temp_idx,'no symptom'] = 1
    
    for index, row in df.iterrows():
        symptoms = row['symptoms'].split('; ')
        for s in symptoms:
            for key in sympts.keys():
                if s in sympts[key]:
                    df.loc[index,key] = 1

    df = df.drop(columns=['symptoms'])
    return df

train = clean_symp(train).drop(columns=['outcome'])
test = clean_symp(test).drop(columns=['Id'])


# In[200]:


train.to_csv('train_clean1.csv')
test.to_csv('test_clean1.csv')


# In[199]:


test


# In[198]:


train


# In[ ]:


train


# In[ ]:




