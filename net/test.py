import numpy as np
from network import *
from layers import *
#from netbp import netbp


X = np.random.uniform(0, 1, size=(100, 20))
y = np.random.randint(4, size=100)


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


print(model.layers[-1].grads.shape)

print([l.error.shape for l in model.layers[1:]])
print([l.grads.shape for l in model.layers[1:]])