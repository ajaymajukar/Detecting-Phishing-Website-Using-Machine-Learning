import knn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = np.loadtxt("new_dataset.csv", delimiter=',')

X = data[:,:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

clf = knn.KNN(3)

y_pred = clf.predict(X_test,X_train,y_train)

score = accuracy_score(y_test,y_pred)

print(score)
