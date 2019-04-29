import numpy as np
from network import *
from layers import *
import utils

# https://www.kaggle.com/anokas/kuzushiji

idxs = np.random.choice(60000, 10000, replace=False)

kmnist_imgs = np.load("../data/kmnist-train-imgs.npz")["arr_0"]
kmnist_imgs = kmnist_imgs.reshape(-1, 784)[idxs, :]

kmnist_labs = np.load("../data/kmnist-train-labels.npz")["arr_0"][idxs]


labels_dict = dict([(0, u"\u304A"), (1, u"\u304D"), (2, u"\u3059"), (3, u"\u3064"),
                    (4, u"\u306A"), (5, u"\u306F"), (6, u"\u307E"), (7, u"\u3084"),
                    (8, u"\u308C"), (9, u"\u3093")])



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
success = model.train(X_train, y_train, X_val, y_val, epochs=1000, batch_size=256, learning_rate=10, penalty=0.12, early_stopping=10)


model.save_weights("final_weights.npy")


# Final score
kmnist_test_imgs = np.load("../data/kmnist-test-imgs.npz")["arr_0"]
kmnist_test_imgs = utils.scale_minmax(kmnist_test_imgs.reshape(-1, 784))

kmnist_test_labs = np.load("../data/kmnist-test-labels.npz")["arr_0"]

X_test, y_test = utils.make_sets(kmnist_test_imgs, kmnist_test_labs, [1.0])

print("Test accuracy:", model.get_accuracy(kmnist_test_imgs, kmnist_test_labs))
preds = model.predict(kmnist_test_imgs)
utils.confusion_matrix(preds, kmnist_test_labs)

