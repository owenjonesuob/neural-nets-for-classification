import numpy as np
from network import *
from layers import *
import utils
#from netbp import netbp

import matplotlib.pyplot as plt


# "Islands" data
n = 200
X = np.random.uniform(0, 1, (n, 2))

y = np.zeros(n)

for k in range(n):
    if np.linalg.norm(X[k, :] - np.array([0.5, 0.5])) < 0.4:
        if X[k, 1] > 0.5:
            y[k] = 0
        else:
            y[k] = 1
    else:
        if X[k, 0] > 0.5:
            y[k] = 2
        else:
            y[k] = 3

y = y.astype(int)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


model = Network(layers = [
    Input(2),
    Dense(2, 20, "sigmoid"),
    Dense(20, 20, "sigmoid"),
    Dense(20, 4, "sigmoid")
])

success = model.train(X, y, epochs=1000, batch_size=32, learning_rate=10, penalty=0)
utils.plot_cost_curves(model, val_curve=False)
utils.plot_boundaries(model, X, y, subdivs=150)




# "Dodgy" overfitting data
n = 300
X = np.random.uniform(0, 1, (n, 2))

y = np.zeros(n)

for k in range(n):
    if X[k, 0] > X[k, 1]:
        y[k] = (np.random.binomial(1, 0.05, 1))
    else:
        y[k] = (np.random.binomial(1, 0.95, 1))

y = y.astype(int)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


X_train, y_train, X_val, y_val = utils.make_sets(X, y, [0.8, 0.2])


model = Network(layers = [
    Input(2),
    Dense(2, 50, "sigmoid"),
    Dense(50, 20, "sigmoid"),
    Dense(20, 2, "sigmoid")
])


success = model.train(X, y, epochs=10000, batch_size=128, learning_rate=10, penalty=0)
utils.plot_cost_curves(model, val_curve=False)
utils.plot_boundaries(model, X, y, subdivs=150)

model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=10000, batch_size=128, learning_rate=10, penalty=0)
utils.plot_cost_curves(model, val_curve=False)
utils.plot_boundaries(model, X, y, subdivs=150)

model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=10000, batch_size=128, learning_rate=10, penalty=0.1)
utils.plot_cost_curves(model, val_curve=False)
utils.plot_boundaries(model, X, y, subdivs=150)
