import numpy as np
from network import *
from layers import *
#from netbp import netbp

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


#X = np.random.uniform(0, 1, size=(10000, 20))
#y = np.random.randint(4, size=10000)

X, y = make_blobs(1000, 20, 4)


model = Network(layers = [
    Input(20),
    Dense(20, 10, "sigmoid"),
    Dense(10, 8, "sigmoid"),
    Dense(8, 4, "sigmoid")
])


print("Network show:")
model.show()

print("Naive preds:", model.predict(X))

print("Initial cost:", model.get_cost(X, y))

weights = model.get_all_weights()
print("Weights:", weights.shape, weights[:10])

grads = model.get_all_grads(X, y)
print("Grads:", grads.shape, grads[:10])

print("Weight shapes:", [l.weights.shape for l in model.layers[1:]])
print("Grad shapes:  ", [l.grads.shape for l in model.layers[1:]])

# Explodes when rate is too high?
success = model.train(X, y, max_iters=1000, learning_rate=0.1, penalty=1, report_level=100)
print("Train success:", success)


preds = model.predict(X)
print(np.mean(preds == y))