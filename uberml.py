# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:44:39 2022

@author: Wakchaure Ishwari Dnyaneshwar
"""

import numpy as np
import pandas as pd

df=pd.read_csv("C:/Users/Asus/Desktop/uber.csv")
df.head()
df.columns
df=df.drop(['Unnamed: 0', 'key'],axis=1)
df.head()
df.dtypes

df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])
df.dtypes

df.isna().sum()
df=df.dropna(axis=0)

df.shape

df=df.assign(hour=df.pickup_datetime.dt.hour,
             month=df.pickup_datetime.dt.month,
             year=df.pickup_datetime.dt.year,
             dayofweek=df.pickup_datetime.dt.dayofweek,
             day=df.pickup_datetime.dt.day
             )

df.head()

df=df.drop(['pickup_datetime'],axis=1)
df.head()
df.columns

x=df.drop(['fare_amount'],axis=1)
y=df['fare_amount']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#linearRegression
from sklearn.linear_model import LinearRegression

model= LinearRegression()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score


#accuracy_score(y_test,y_pred)
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))

#RandomForest

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(n_estimators=100)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score


#accuracy_score(y_test,y_pred)
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))
