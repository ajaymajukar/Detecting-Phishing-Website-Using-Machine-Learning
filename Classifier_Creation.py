
import os
import pandas as pd
import numpy as np
import knn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle



save_model_folder = "model/"


directory = os.path.dirname(save_model_folder)
if not os.path.exists(directory):
	os.makedirs(directory)


input_file = "dataset.csv"
 #Importing dataset
data = np.loadtxt("dataset.csv", delimiter = ",")
data2 = np.loadtxt("new_dataset.csv", delimiter = ",")

#Seperating features and labels

X = data[: , :-1]
y = data[: , -1]

#Seperating training features, testing features, training labels & testing labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


h = .02  # step size in the mesh

names = [
		"RBF SVM",
        "Decision Tree",
		"Random Forest"
		]

classifiers = [
           SVC(kernel='rbf', C=1),
           DecisionTreeClassifier(max_depth=5),
           RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
          ]




# iterate over classifiers
for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    pickle.dump(clf, open(save_model_folder+name+".sav", 'wb'))

    print(name," -- ",100*score,"%")
