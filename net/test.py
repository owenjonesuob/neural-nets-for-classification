import numpy as np
from network import *
from layers import *
import utils
#from netbp import netbp

from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt


X, y = make_moons(10000, noise=0.1)
X = utils.scale_minmax(X)
print(X.max(), X.min())
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


model = Network(layers = [
    Input(2),
    Dense(2, 10, "sigmoid"),
    Dense(10, 4, "sigmoid"),
    Dense(4, 2, "sigmoid")
])

success = model.train(X, y, epochs=400, batch_size=512, learning_rate=10, penalty=0)
utils.plot_cost_curves(model, val_curve=False)
utils.plot_boundaries(model, X, y, subdivs=150)



X_train, y_train, X_val, y_val = utils.make_sets(X, y, [0.8, 0.2])

model = Network(layers = [
    Input(2),
    Dense(2, 10, "sigmoid"),
    Dense(10, 4, "sigmoid"),
    Dense(4, 2, "sigmoid")
])

success = model.train(X_train, y_train, X_val, y_val, epochs=500, batch_size=256, learning_rate=10, penalty=0)
utils.plot_cost_curves(model)
utils.plot_boundaries(model, X, y, subdivs=150)




#X, y = make_blobs(10000, 2, 4)

X, y = make_blobs(10000, 2, 4)#, center_box=(10, 30))
X = utils.scale_minmax(X)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

model2 = Network(layers = [
    Input(2),
    Dense(2, 50, "sigmoid"),
    Dense(50, 20, "sigmoid"),
    Dense(20, 4, "softmax")
])


success = model2.train(X, y, epochs=100, batch_size=128, learning_rate=1, penalty=0.1)
utils.plot_cost_curves(model2, val_curve=False)
utils.plot_boundaries(model2, X, y, subdivs=100)