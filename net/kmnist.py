import numpy as np
from network import *
from layers import *
import utils

# https://www.kaggle.com/anokas/kuzushiji

kmnist_imgs = np.load("../data/kmnist-train-imgs.npz")["arr_0"]
print("kmnist shape:", kmnist_imgs.shape)
kmnist_imgs = kmnist_imgs.reshape(-1, 784)#[0:1000, ]

kmnist_labs = np.load("../data/kmnist-train-labels.npz")["arr_0"]#[0:1000, ]
print("labs shape:", kmnist_labs.shape)

X_train, y_train, X_val, y_val, X_test, y_test = make_sets(kmnist_imgs, kmnist_labs, [0.6, 0.2, 0.2])


model = Network(layers = [
    Input(784),
    Dense(784, 100, "sigmoid"),
    Dense(100, 40, "sigmoid"),
    Dense(40, 10, "softmax")
])

model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=512, learning_rate=10, penalty=0.1, early_stopping=10)
utils.plot_cost_curves(model)

preds = model.predict(X_val)
print("Confusion matrix:\n", utils.confusion_matrix(preds, y_val))
#utils.plot_boundaries(model, X, y, subdivs=150)




