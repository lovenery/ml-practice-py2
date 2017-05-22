# https://www.kaggle.com/c/titanic/data
# https://www.youtube.com/watch?v=yLsKZTWyEDg

### 1. Data Preparation ###
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(path.join(path.dirname(__file__), './train.csv'))
# print(df)
# print(df.columns)

# we only use three column
X = pd.DataFrame()
X['sex'] = df['Sex']
X['age'] = df['Age']
X['survived'] = df['Survived']
# print(X)

# delete lost row (NaN), but the best way maybe 'mean imputation'
X = X.dropna(axis=0, ) # drop index (0)
# X = X.reset_index(drop=True) # no needs # drop=True: del index column
# print(X)

# Dependent variable y, delete from X
y = X['survived']
X = X.drop(['survived'], axis=1)
# print(y)

# convert string to normalized number!
# print(pd.get_dummies(X.sex))
X['sex'] = pd.get_dummies(X.sex)['female'] # so, male: 0; female: 1;
# print(X)

# ref: http://www.cnblogs.com/chaosimple/p/4153167.html
# scale our features, as with linear regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Fit to data, then transform it.
# print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_test)


### 2. Model Creation ###
'''
At this point I have a test and train set defined. I will use train to train my model and test to see how accurate the model is.
There's one problem with that though. Lets say my model is right 70% of the time. Is that good? Maybe?
I'm going to build a simple 'base rate' model to compare my logistic model to, so we can see if our logistic model is useful or not.
Then, I'll build my logistic model.
'''

## Base Rate Model
# For my base rate model, predict that everyone dies.
def my_base_rate_model(X):
    y = np.zeros(X.shape[0])
    return y

from sklearn.metrics import accuracy_score # accuracy_score(true_value, predict)
y_base_rate = my_base_rate_model(X_test) # all zero
print("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))

## Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1.0) # l1 or l2
model.fit(X_train, y_train)
print("Logistic accuracy is %2.2f" % accuracy_score(y_test, model.predict(X_test)))

# Use ofther metric, AUC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report # Better way compare model than accuracy
print("---Base Model---")
base_roc_auc = roc_auc_score(y_test, my_base_rate_model(X_test))
print("Base Rate AUC = %2.2f" % base_roc_auc)
print( classification_report(y_test, my_base_rate_model(X_test)) )

print("\n\n---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print("Logistic AUC = %2.2f" % logit_roc_auc)
print( classification_report(y_test, model.predict(X_test)) )

# Plot
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower right')
plt.show()
