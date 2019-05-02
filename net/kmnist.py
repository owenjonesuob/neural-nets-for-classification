import numpy as np
from network import *
from layers import *
import utils

import matplotlib.pyplot as plt
plt.style.use(["seaborn-paper", "seaborn-white"])
plt.rcParams["image.cmap"] = "Greys"

np.random.seed(101)

# https://www.kaggle.com/anokas/kuzushiji


# Take a random subset of the data if necessary (e.g. if there is too much to fit into memory)
idxs = np.random.choice(60000, 10000, replace=False)

kmnist_imgs = np.load("../data/kmnist-train-imgs.npz")["arr_0"]
kmnist_imgs = kmnist_imgs.reshape(-1, 784)[idxs, :]

kmnist_labs = np.load("../data/kmnist-train-labels.npz")["arr_0"][idxs]

labels_dict = {
    0: u"\u304A", 1: u"\u304D", 2: u"\u3059", 3: u"\u3064", 4: u"\u306A",
    5: u"\u306F", 6: u"\u307E", 7: u"\u3084", 8: u"\u308C", 9: u"\u3093"
}


# Visualise some examples from the dataset
n_examples = 4

plt.subplots(n_examples+1, 10)
for col in range(10):

    # Modern typeset character in first row
    plt.subplot(n_examples+1, 10, col+1)
    plt.text(0, 0, labels_dict.get(col), fontname="Yu Mincho", fontsize=32)
    plt.axis("off")

    # Random matching examples from dataset
    idxs = [j for j, x in enumerate(kmnist_labs) if x == col]
    to_plot = np.random.choice(idxs, 4)
    for row in range(n_examples):
        plt.subplot(5, 10, 10*(row+1) + col + 1)
        plt.imshow(kmnist_imgs[to_plot[row], :].reshape(28, 28), origin="upper")
        plt.axis("off")

plt.show()


# Split the data into three sets
X_train, y_train, X_val, y_val, X_test, y_test = make_sets(kmnist_imgs, kmnist_labs, [0.6, 0.2, 0.2])
X_train = utils.scale_minmax(X_train)
X_val = utils.scale_minmax(X_val)
X_test = utils.scale_minmax(X_test)


# Create and train our network
model = Network(layers = [
    Input(784),
    Dense(784, 200, "sigmoid"),
    Dense(200, 30, "sigmoid"),
    Dense(30, 10, "softmax")
])

model.reset()
success = model.train(X_train, y_train, X_val, y_val, epochs=1000, batch_size=256, learning_rate=10, penalty=0.12, early_stopping=30)
model.save_weights("val_weights.npy")


# Evaluate on test set; adjust hyperparameters to achieve best performance on validation set
utils.plot_cost_curves(model)

preds = model.predict(X_val)
print("Confusion matrix:\n", utils.confusion_matrix(preds, y_val))


# Visualising some of the weights in the first layer
plt.subplots(10, 10)

for k in range(100):
    plt.subplot(10, 10, k+1)
    plt.imshow(model.layers[1].weights[k, 1:].reshape(28, 28), origin="upper")
    plt.axis("off")

plt.show()


# Final score (how well we can expect our model to perform on unseen data)
test_preds = model.predict(X_test)
print("Test accuracy:", model.get_accuracy(X_test, y_test))
