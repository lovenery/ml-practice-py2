from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import cPickle as pickle
import requests, json

import pandas as pd
import os.path as path
# filename = path.join(path.dirname(__file__), './iris.data')
# data = pd.read_csv(filename, header=None)
# print data

iris = datasets.load_iris()
# print iris.DESCR
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
rfc = RandomForestClassifier(n_estimators=100, n_jobs=2)
rfc.fit(X_train, y_train)

# print "Accuracy = %0.2f" % accuracy_score(y_test, rfc.predict(X_test))
print classification_report(y_test, rfc.predict(X_test))

# Model serialization / marshalling
output_file = path.join(path.dirname(__file__), './iris_rfc.pkl')
pickle.dump(rfc, open(output_file, "wb"))
my_random_forest = pickle.load(open(output_file, "rb"))
# print classification_report(y_test, my_random_forest.predict(X_test))