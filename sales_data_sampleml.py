# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 13:23:28 2022

@author: Wakchaure Ishwari Dnyaneshwar
"""

import numpy as np
import pandas as pd

df=pd.read_csv("sales_data_sample.csv",encoding='latin1')
df.head()
df.info()

df.isna().sum()
df=df.dropna()
df.head()

X=df.iloc[:,[1,4]].values

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#elbow method
list=[]
for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=42)
    model.fit(X)
    list.append(model.inertia_)
list

plt.plot(range(1,11),list)
plt.title("Elbow Method graph")
plt.ylabel("WCSS list")
plt.xlabel("Number of clusters(k)")


#KMeans

model=KMeans(n_clusters=3,random_state=42)

y_pred=model.fit_predict(X)


plt.scatter(X[y_pred==0,0],X[y_pred==0,1],c="blue",label="C1")
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],c="red",label="C2")
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],c="green",label="C3")

plt.title("K-Means Clustering")
plt.ylabel("Quantity Ordered")
plt.xlabel("sales")
plt.legend()
plt.show()

