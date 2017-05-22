# http://machinelearningmastery.com/linear-regression-tutorial-using-gradient-descent-for-machine-learning/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.array([1, 2, 4, 3, 5])
y = np.array([1, 3, 3, 2, 5])
# plt.scatter(x, y)
# plt.grid()
# plt.show()

# y = B0 + B1 * x
print("B0, B1")
iters = 20
B0 = 0.0
B1 = 0.0
alpha = 0.01
eps = 1e-4
errorSet = []
for i in range(0, iters):
    predict = B0 + B1 * x
    error = predict - y # also called loss
    B0 = B0 - alpha * np.sum(error)
    B1 = B1 - alpha * np.sum(error * x)
    print(B0, B1)
    e = -alpha * np.sum(error * x)
    errorSet.append(e)
    # see what happen
    # yHat = B0 + B1 * x
    # plt.plot(x, yHat, color='red')
    if (e < eps):
        print("Stop at %d iter" % i)
        break

# error close to zero
# print(errorSet)
# plt.plot(errorSet)
# plt.show()

# ans
yHat = B0 + B1 * x
plt.plot(x, yHat, color='red')
plt.scatter(x, yHat, color='red')
plt.scatter(x, y)
plt.grid()
plt.title('x and y')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
