import numpy as np
import matplotlib.pyplot as plt


def make_sets(data, labels, props, shuffle=True):
    
    if not np.isclose(sum(props), 1):
        raise ValueError("proportions must sum to 1")
    
    m = data.shape[0]
    idx = np.random.permutation(m) if shuffle else range(m)
    
    sets = []
    cursor = 0
    
    for p in props:
        dset = data[idx, :][cursor:(cursor+int(p*m)), :]
        sets.append(dset)
        lset = labels[idx][cursor:(cursor+int(p*m))]
        sets.append(lset)
        
        cursor = int(p*m)
    
    return sets



def scale_minmax(data):
    scaled = np.amax(data, axis=0) - data
    scaled = scaled / (np.amax(data, axis=0) - np.amin(data, axis=0))
    return scaled



def plot_boundaries(model, data, labels, subdivs=200, alpha=0.2):
    
    if data.shape[1] != 2:
        raise ValueError("Can only visualise 2D data")

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    
    xmin, xmax = ax.get_xlim()
    xstep = (xmax-xmin)/subdivs
    ymin, ymax = ax.get_ylim()
    ystep = (ymax-ymin)/subdivs
    
    grid = np.mgrid[xmin:xmax:xstep, ymin:ymax:ystep].reshape(2, -1).T

    ax.contourf(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep),
               model.predict(grid).reshape(-1, subdivs).T,
               alpha = alpha)
    
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return None



def plot_cost_curves(model, val_curve=True):

    plt.plot(model.train_cost_history)
    if val_curve:
        plt.plot(model.val_cost_history)
        
    plt.show()

    return None
