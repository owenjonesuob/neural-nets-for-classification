import numpy as np
from network import *
from layers import *
import utils

import matplotlib.pyplot as plt
plt.style.use(["seaborn-paper", "seaborn-white"])
plt.rcParams["image.cmap"] = "Greys"

np.random.seed(101)

# https://www.kaggle.com/anokas/kuzushiji


kmnist_test_imgs = np.load("../data/kmnist-test-imgs.npz")["arr_0"]
kmnist_test_imgs = utils.scale_minmax(kmnist_test_imgs.reshape(-1, 784))

kmnist_test_labs = np.load("../data/kmnist-test-labels.npz")["arr_0"]

labels_dict = {
    0: u"\u304A", 1: u"\u304D", 2: u"\u3059", 3: u"\u3064", 4: u"\u306A",
    5: u"\u306F", 6: u"\u307E", 7: u"\u3084", 8: u"\u308C", 9: u"\u3093"
}


model = Network(layers = [
    Input(784),
    Dense(784, 200, "sigmoid"),
    Dense(200, 30, "sigmoid"),
    Dense(30, 10, "softmax")
])

model.load_weights("val_weights.npy")


# Final score
test_preds = model.predict(kmnist_test_imgs)
print("Test accuracy:", model.get_accuracy(kmnist_test_imgs, kmnist_test_labs))


# "Confusion plot"
n_classes = 10
fig, axs = plt.subplots(n_classes, n_classes)
fig.subplots_adjust(top=0.82, left=0.15)

# For each row (predicted class)...
for row in range(n_classes):

    # Add axis labels
    p_pos = axs[row][0].get_position()
    fig.text(p_pos.x0 - 0.07, p_pos.y0 + 0.01, labels_dict.get(row), fontname="Yu Mincho", fontsize=18)
    
    a_pos = axs[0][row].get_position()
    fig.text(a_pos.x0 + 0.01, a_pos.y1 + 0.05, labels_dict.get(row), fontname="Yu Mincho", fontsize=18)


    # Get indexes of images classified with that label
    p_idxs = [i for i, x in enumerate(model.predict(kmnist_test_imgs)) if x == row]

    # Then for each column (actual class)...
    for col in range(n_classes):

        # Get indexes of images with that ACTUAL label
        a_idxs = [j for j, x in enumerate(kmnist_test_labs) if x == col]

        # Plot an image with that ACTUAL label
        plt.subplot(n_classes, n_classes, row*n_classes + col + 1)
        candidates = np.intersect1d(p_idxs, a_idxs)

        if len(candidates) == 0:
            plt.axis("off")
        else:
            to_plot = np.random.choice(candidates, 1)
            plt.imshow(kmnist_test_imgs[to_plot, :].reshape(28, 28), origin="upper")
            plt.axis("off")

# Add figure labels
fig.text(0.05, 0.5, "Predicted class", ha="center", va="center", fontsize=12, rotation="vertical")
fig.text(0.5, 0.95, "Actual class", ha="center", va="center", fontsize=12)

plt.show()
