import numpy as np
from network import *
from layers import *
import utils
#from netbp import netbp

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


#X = np.random.uniform(0, 1, size=(10000, 20))
#y = np.random.randint(4, size=10000)

X, y = make_blobs(10000, 2, 4)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

model = Network(layers = [
    Input(2),
    Dense(2, 50, "sigmoid"),
    Dense(50, 20, "sigmoid"),
    Dense(20, 4, "sigmoid")
])


print("Network show:")
model.show()



print("Naive preds:", model.predict(X)[:10], "...")

print("Initial cost:", model.get_cost(X, y))

weights = model.get_all_weights()
print("Weights:", weights.shape, weights[:10])

grads = model.get_all_grads(X, y)
print("Grads:", grads.shape, grads[:10])

print("Weight shapes:", [l.weights.shape for l in model.layers[1:]])
print("Grad shapes:  ", [l.grads.shape for l in model.layers[1:]])


success = model.train(X, y, epochs=100, batch_size=100, learning_rate=1, penalty=0.1)
plt.plot(model.cost_history)
utils.plot_boundary(model, X, y, subdivs=150)