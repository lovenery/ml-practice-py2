# https://www.youtube.com/watch?v=JsX0D92q1EI
# https://anaconda.org/benawad/gradient-descent/notebook

import os.path as path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

x_points = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
y_points = [1, 2, 3, 1, 4, 5, 6, 4, 7, 10, 15, 9]
# plt.plot(x_points, y_points, 'bo')
# plt.show()

# y = mx + b
m = 0
b = 0
def y(x):
    return m * x + b
def plot_line(y, data_points):
    x_values = [i for i in range(int(min(data_points)) - 1, int(max(data_points)) + 2)]
    y_values = [y(x) for x in x_values]
    plt.plot(x_values, y_values, 'r')
# plot_line(y, x_points)
# plt.show()

def summation(y, x_points, y_points):
    total1 = 0
    total2 = 0
    for i in range(1, len(x_points)):
        total1 += y(x_points[i]) - y_points[i]
        total2 += (y(x_points[i]) - y_points[i]) * x_points[i]
    return total1 / len(x_points), total2 / len(x_points)


## Start Here ##
# learn rate (alpha)
"""
learn = .001  # .001, .01, .1, 1 ...
for i in range(2000):
    s1, s2 = summation(y, x_points, y_points)
    m = m - learn * s2
    b = b - learn * s1
    # print(learn * s1)
    if(-learn * s2 < 1e-4):
        print("Stop at %d iter" % i)
        break
    # plot_line(y, x_points)

print("Slope: %2.3f, Constant %2.3f" % (m, b))
plot_line(y, x_points)
plt.plot(x_points, y_points, 'bo')
plt.show()
"""

# """
## Another Example: Chirps ##
df = pd.read_csv(path.join(path.dirname(__file__), './chirps.csv'))
# print(df.head())
x_points = df['chirps'].tolist()
y_points = df['temp'].tolist()
# plt.plot(x_points, y_points, 'bo')
# plt.show()
learn = .001 # more smaller, more slower learn
for i in range(1000):
    s1, s2 = summation(y, x_points, y_points)
    m = m - learn * s2
    b = b - learn * s1
    if(-learn * s2 < 1e-4):
        print("Stop at %d iter" % i)
        break

plt.plot(x_points, y_points, 'bo')
plot_line(y, x_points)
plt.show()
# """
