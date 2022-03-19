import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(3,3))
    ax = plt.subplot(111)
    plt.xticks([]), plt.yticks([])
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1],
                    c = plt.cm.bwr(d[i] / 1.),
                    marker = 'o',
                    s = 12)
        savename = title + '.png'
        plt.savefig(savename, dpi=300)