# http://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(path.join(path.dirname(__file__), './contrive.csv'))
# print(df)

Y = df['Y'].as_matrix()
X1 = df['X1'].as_matrix()  # ndarray # df['X1'].tolist() # list
X2 = df['X2'].as_matrix()  # ndarray # df['X2'].tolist() # list
# blue = plt.scatter(X1[:5], X2[:5])  # [:5] : 0 ~ 4
# red = plt.scatter(X1[5:], X2[5:], color='red') # [5:] : 5 ~ end
# plt.legend((blue, red), ('Series1', 'Series2'), scatterpoints=1, loc='lower left')
# plt.show()

# output = b0 + b1*x1 + b2*x2
# p(class=0) = 1 / (1 + e^(-output))
B0 = 0.0
B1 = 0.0
B2 = 0.0
alpha = 0.3

for epoch in range(10):
    for j in range(len(Y)):
        yHat = 1 / (1 + np.exp(-(B0 + B1 * X1[j] + B2 * X2[j])))
        # print(yHat)
        error = Y[j] - yHat
        B0 = B0 + alpha * error * yHat * (1 - yHat) * 1.0
        B1 = B1 + alpha * error * yHat * (1 - yHat) * X1[j]
        B2 = B2 + alpha * error * yHat * (1 - yHat) * X2[j]
        # print(B0)
        # print(B1)
        # print(B2)
        # print('---------')

print("B0: %f" % B0)
print("B1: %f" % B1)
print("B2: %f" % B2)

output = B0 + B1 * X1 + B2 * X2
probabilities = 1 / (1 + np.exp(-output))
print('probabilities:', probabilities)

def convert(n):
    return 0 if n < 0.5 else 1
prediction = map(convert, probabilities)
print('prediction: ', prediction)

print('accuracy = %d%%' % (sum(prediction == Y) / len(Y) * 100))
