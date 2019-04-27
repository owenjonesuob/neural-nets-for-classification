import numpy as np
from network import *
from layers import *
import utils

# https://www.kaggle.com/anokas/kuzushiji

idxs = np.random.choice(60000, 10000, replace=False)

kmnist_imgs = np.load("../data/kmnist-train-imgs.npz")["arr_0"]
kmnist_imgs = kmnist_imgs.reshape(-1, 784)[idxs, :]

kmnist_labs = np.load("../data/kmnist-train-labels.npz")["arr_0"][idxs]


X_train, y_train, X_val, y_val = make_sets(kmnist_imgs, kmnist_labs, [0.9, 0.1])
X_train = utils.scale_minmax(X_train)
X_val = utils.scale_minmax(X_val)


model = Network(layers = [
    Input(784),
    Dense(784, 200, "sigmoid"),
    Dense(200, 30, "sigmoid"),
    Dense(30, 10, "softmax")
])

model.load_weights("val_weights.npy")
success = model.train(X_train, y_train, X_val, y_val, epochs=1000, batch_size=256, learning_rate=10, penalty=0.12, early_stopping=30)


model.save_weights("final_weights.npy")


# Final score
kmnist_test_imgs = np.load("../data/kmnist-test-imgs.npz")["arr_0"]
kmnist_test_imgs = kmnist_test_imgs.reshape(-1, 784)

kmnist_test_labs = np.load("../data/kmnist-test-labels.npz")["arr_0"]

print("Test accuracy:", model.get_accuracy(kmnist_test_imgs, kmnist_test_labs))

