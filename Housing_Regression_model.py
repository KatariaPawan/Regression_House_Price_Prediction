# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:02:16 2022

@author: Pawan Kataria
"""
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# importing the housing dataset
house = pd.read_csv(r"C:\Users\Pawan Kataria\OneDrive\ML_Projects\Regression\housing.csv")
#checking the any null values inside the dataset
house.isnull().sum()
# separating the dataset into independent and dependent variables
x = house.iloc[:, 0:13].values
y = house.iloc[:, -1].values
# making all the data of x variable into standard form
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
# spliting the dataset into traing and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 0)
# appling the linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
#finding the model accuracy
y_pred = reg.predict(X_test)
from sklearn.metrics import r2_score
model_accuracy = r2_score(y_test,y_pred)
model_accuracy
