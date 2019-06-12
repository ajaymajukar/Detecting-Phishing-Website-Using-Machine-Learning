#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:05:25 2019

@author: vkdarshan
"""

import numpy as np
import feature_extraction
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC
from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr

data = np.loadtxt("dataset.csv", delimiter = ",")

#Seperating features and labels
X = data[: , :-1]
y = data[: , -1]

#Seperating training features, testing features, training labels & testing labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = rfc()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score*100)