import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5)
# print(x)
y = 1 / ( 1 + np.exp(-x) )
# print(y)

plt.plot(x, y)
plt.grid(axis='y')
plt.title('Logistic Function')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()