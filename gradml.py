# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:53:21 2022

@author: Wakchaure Ishwari Dnyaneshwar
"""

import numpy as np

def grad(start,lr,gradient,iteration,tol=0.01):
    steps=[start]
    X=start
    for i in range(iteration):
        
        diff=lr*gradient(X)
        if(np.abs(diff)<tol):
            break
        X=X-diff
        steps.append(X)
    return(steps,X,lr)

def gradient_f(X):
    return(X+3)**2

def gradient(X):
    return(X**2+6*X+9)

history,lr,result=grad(2, 0.001, gradient, 200)
steps=history
print(steps)
print(result)
print(len(steps))