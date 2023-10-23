# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 23:15:21 2023

@author: allan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression

import joblib


data = pd.read_csv('parkinsons.data')

data.drop('name',axis=1,inplace=True)

data.head()

data.info()

data = shuffle(data,random_state=42)

data['status'].value_counts(normalize=True) 

X=data.drop('status',axis=1)
y=data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

X_train_lr = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns) #return as df
X_test_lr = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns) #return as df

clf=RandomForestClassifier(random_state=42,n_estimators=500,max_features=5)
clf.fit(X_train,y_train)
y_pred_class = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:,1]

joblib.dump(clf, "parkinsons_model.joblib")

