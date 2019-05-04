import numpy as np
from network import *
from layers import *
import utils

import matplotlib.pyplot as plt
plt.style.use(["seaborn-paper", "seaborn-white"])
plt.rcParams["image.cmap"] = "viridis"


# "Dodgy" data

np.random.seed(101)

n = 250
X = np.random.uniform(0, 1, (n, 2))

y = np.zeros(n)

for k in range(n):
    if X[k, 0] > X[k, 1]:
        y[k] = (np.random.binomial(1, 0.05, 1))
    else:
        y[k] = (np.random.binomial(1, 0.95, 1))

y = y.astype(int)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("tricky_data.png")
plt.show()


X_train, y_train, X_val, y_val = utils.make_sets(X, y, [0.75, 0.25])
X_train = utils.scale_minmax(X_train)
X_val = utils.scale_minmax(X_val)


model = Network(layers = [
    Input(2),
    Dense(2, 100, "sigmoid"),
    Dense(100, 20, "sigmoid"),
    Dense(20, 2, "softmax")
])

# First with no early stopping or regularisation
success = model.train(X_train, y_train, X_val, y_val, epochs=10000, batch_size=128, learning_rate=10, penalty=0)
utils.plot_cost_curves(model, file="tricky_over_curves.png")
utils.plot_boundaries(model, X_train, y_train, file="tricky_over_train.png")
utils.plot_boundaries(model, X_val, y_val, file="tricky_over_val.png")

# Now with early stopping...
model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=10000, batch_size=128, learning_rate=10, penalty=0, early_stopping=200)
utils.plot_cost_curves(model, file="tricky_early_curves.png")
utils.plot_boundaries(model, X_train, y_train, file="tricky_early_train.png")
utils.plot_boundaries(model, X_val, y_val, file="tricky_early_val.png")



# "Islands" data

np.random.seed(101)

n = 500
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

X_train, y_train, X_val, y_val = utils.make_sets(X, y, [0.75, 0.25])
X_train = utils.scale_minmax(X_train)
X_val = utils.scale_minmax(X_val)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()


model = Network(layers = [
    Input(2),
    Dense(2, 20, "sigmoid"),
    Dense(20, 20, "sigmoid"),
    Dense(20, 4, "softmax")
])

success = model.train(X_train, y_train, X_val, y_val, epochs=5000, batch_size=128, learning_rate=10, penalty=0.05, early_stopping=80)
utils.plot_boundaries(model, X_val, y_val, file="islands_early_val.png")

model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=5000, batch_size=128, learning_rate=10, penalty=0.05, early_stopping=800)
utils.plot_cost_curves(model, file="islands_curves.png")
utils.plot_boundaries(model, X_val, y_val, file="islands_late_val.png")
