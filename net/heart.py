# https://www.kaggle.com/ronitf/heart-disease-uci

import numpy as np
from network import *
from layers import *
import utils

heart = np.genfromtxt("../data/heart.csv", delimiter=",")
heart = np.delete(heart, np.isnan(heart[:, -1]), axis=0)
print("heart shape:", heart.shape)

X_train, y_train, X_val, y_val, X_test, y_test = make_sets(heart[:, :-1], heart[:, -1].astype(int), [0.6, 0.2, 0.2])
print(np.unique(heart[:, -1].astype(int)))


model = Network(layers = [
    Input(13),
    Dense(13, 20, "sigmoid"),
    Dense(20, 2, "softmax")
])


model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=100000, batch_size=64, learning_rate=10, penalty=0.1, early_stopping=2000)
utils.plot_cost_curves(model)

preds = model.predict(X_val)
print("Confusion matrix:\n", utils.confusion_matrix(preds, y_val))
#utils.plot_boundaries(model, X, y, subdivs=150)




