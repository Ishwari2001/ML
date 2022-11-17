# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:05:13 2022

@author: Wakchaure Ishwari Dnyaneshwar
"""

import numpy as np
import pandas as pd

df=pd.read_csv("C:/Users/Asus/Desktop/diabetes.csv")
df.head()
df.isna().sum()

x=df.drop(['Outcome'],axis=1)
y=df['Outcome']
x.shape

y.shape
from sklearn.metrics import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
accuracy_score(y_test,y_pred)
acc=(tp+tn)/(tp+tn+fn+fp)
print(round(acc,2)*100,"%")
err=(fp+fn)/(tp+tn+fn+fp)
err

precision=tp/(tp+fp)
precision

recc=tp/(tp+fn)
recc
round(precision,2)
round(recc,2)
