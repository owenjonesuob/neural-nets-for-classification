import numpy as np
from network import *
from layers import *
import utils

import matplotlib.pyplot as plt
plt.style.use(["seaborn-paper", "seaborn-white"])
plt.rcParams["image.cmap"] = "viridis"

from sklearn.datasets import make_blobs, make_moons

# Moons!
np.random.seed(101)

X, y = make_moons(10000, noise=0.1)
X = utils.scale_minmax(X)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Logistic regression

model = Network(layers = [
    Input(2),
    Dense(2, 2, "sigmoid")
])

success = model.train(X, y, epochs=400, batch_size=512, learning_rate=10, penalty=0)
utils.plot_cost_curves(model, val_curve=False)
utils.plot_boundaries(model, X, y, subdivs=150)

# Smallish neural network, with same hyperparams

model = Network(layers = [
    Input(2),
    Dense(2, 10, "sigmoid"),
    Dense(10, 4, "sigmoid"),
    Dense(4, 2, "sigmoid")
])

success = model.train(X, y, epochs=400, batch_size=512, learning_rate=10, penalty=0)
utils.plot_cost_curves(model, val_curve=False)
utils.plot_boundaries(model, X, y, subdivs=150)

# Split the dataset to show off validation curves and early stopping

X_train, y_train, X_val, y_val = utils.make_sets(X, y, [0.75, 0.25])

model = Network(layers = [
    Input(2),
    Dense(2, 20, "sigmoid"),
    Dense(20, 4, "sigmoid"),
    Dense(4, 2, "softmax")
])

success = model.train(X_train, y_train, X_val, y_val, epochs=2000, batch_size=256, learning_rate=10, penalty=0.01, early_stopping=150)
utils.plot_cost_curves(model)
utils.plot_boundaries(model, X, y, subdivs=150)


# Blobs!
np.random.seed(101)

X, y = make_blobs(5000, 2, 4, center_box=(10, 30))
X_train, y_train, X_val, y_val = utils.make_sets(X, y, [0.75, 0.25])

X_train = utils.scale_minmax(X_train)
X_val = utils.scale_minmax(X_val)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()

model2 = Network(layers = [
    Input(2),
    Dense(2, 30, "sigmoid"),
    Dense(30, 10, "sigmoid"),
    Dense(10, 4, "softmax")
])

success = model2.train(X_train, y_train, X_val, y_val, epochs=500, batch_size=128, learning_rate=1, penalty=0.1, early_stopping=100)
utils.plot_cost_curves(model2)
utils.plot_boundaries(model2, X_train, y_train, subdivs=100)
utils.plot_boundaries(model2, X_val, y_val, subdivs=100)
