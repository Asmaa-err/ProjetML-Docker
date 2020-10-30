import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import request
import json



#le dataset
train=pd.read_csv('heart_failure.csv')
train.rename(columns=lambda x: x.replace('DEATH_EVENT', 'Mort'), inplace=True)

train["age"] = train["age"]
bins = [39, 50, 65, 70, np.inf]
labels = ['39-50', '50-65', '65-70', '70-95']
train['AgeGroup'] = pd.cut(train["age"], bins, labels = labels)
train['AgeGroup']=train['AgeGroup'].map({'39-50':1, '50-65':2, '65-70':3, '70-95':4})




def prediction(param):

    param=np.array(param).reshape(1,-1) 
    
    #col_1=float(request.form.get(param1, False))
    #col_2=float(request.form.get(param2, False))
    #a=[]
    #cls=pickle.load(open("cls_heart_attack.pkl", "rb"))
    #return a"""

    cls=pickle.load(open("cls_heart_attack.pkl", "rb"))

    #return json.dumps({'user':cls.predict(np.array(a).reshape(1,-1))})
    return (cls.predict(param))


  
def entrainement():

    predictors = train.drop(['Mort'], axis=1)
    target = train["Mort"]

    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
    cls=RandomForestClassifier(max_depth=12,n_estimators=300).fit(x_train,y_train)
    
    #sauver cls
    filename = 'cls_heart_attack.pkl'
    pickle.dump(cls, open(filename, 'wb'))

    return(cls.score(x_val,y_val))




