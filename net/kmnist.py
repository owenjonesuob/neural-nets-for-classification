import numpy as np
from network import *
from layers import *
import utils

# https://www.kaggle.com/anokas/kuzushiji

kmnist_imgs = np.load("../data/kmnist-train-imgs.npz")["arr_0"]
print("kmnist shape:", kmnist_imgs.shape)
kmnist_imgs = utils.scale_minmax(kmnist_imgs.reshape(-1, 784))[0:10000, ]
print("kmnist reshaped:", kmnist_imgs.shape)

kmnist_labs = np.load("../data/kmnist-train-labels.npz")["arr_0"][0:10000, ]
print("labs shape:", kmnist_labs.shape)

X_train, y_train, X_val, y_val, X_test, y_test = make_sets(kmnist_imgs, kmnist_labs, [0.6, 0.2, 0.2])


model = Network(layers = [
    Input(784),
    Dense(784, 100, "sigmoid"),
    Dense(100, 50, "sigmoid"),
    Dense(50, 10, "softmax")
])

model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=500, batch_size=128, learning_rate=10, penalty=0.1, early_stopping=10)
utils.plot_cost_curves(model)

preds = model.predict(X_val)
print("Confusion matrix:\n", utils.confusion_matrix(preds, y_val))
#utils.plot_boundaries(model, X, y, subdivs=150)

import matplotlib.pyplot as plt
plt.matshow(X_train[0, :].reshape(28, 28), cmap="Greys_r")
plt.show()

plt.subplots(5, 5)

for k in range(25):
    plt.subplot(5, 5, k+1)
    plt.imshow(model.layers[1].weights[k, 1:].reshape(28, 28), cmap="Greys_r", origin="upper", interpolation="nearest")

plt.show()


# Examples of: correct in first 5, incorrect in 6th col
plt.subplots(10, 6)
for row in range(10):
    correct_idxs = [i for i, x in enumerate(preds == y_val) if preds[i] == row and x]
    incorrect_idxs = [i for i, x in enumerate(preds == y_val) if preds[i] == row and not x]
    for col in range(5):
        plt.subplot(10, 6, 6*row + col + 1)
        plt.imshow(X_val[correct_idxs[col], :].reshape(28, 28), cmap="Greys_r", origin="upper", interpolation="nearest")
        plt.axis("off")
    plt.subplot(10, 6, 6*row + 5 + 1)
    plt.imshow(X_val[incorrect_idxs[0], :].reshape(28, 28), cmap="Greys_r", origin="upper", interpolation="nearest")
    plt.axis("off")

plt.show()


