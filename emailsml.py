# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:30:51 2022

@author: Wakchaure Ishwari Dnyaneshwar
"""

import numpy as np
import pandas as pd

df=pd.read_csv("C:/Users/Asus/Desktop/emails.csv")
df.head()
df.columns
df.isna().sum()
df=df.drop(['Email No.'],axis=1)

x=df.drop(['Prediction'],axis=1)
y=df['Prediction']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

x_train=sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Knn
from sklearn.neighbors import KNeighborsClassifier

model= KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error

mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
acc= accuracy_score(y_test, y_pred)
rmse=np.sqrt(mse)
print(mse,mae,rmse,acc)


#svm

from sklearn.svm import SVC

svmm=SVC()
svmm.fit(x_train,y_train)
y_preds=svmm.predict(x_test)

from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error

mse=mean_squared_error(y_test, y_preds)
mae=mean_absolute_error(y_test, y_preds)
acc= accuracy_score(y_test, y_preds)
rmse=np.sqrt(mse)
print(mse,mae,rmse,acc)

