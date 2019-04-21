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



def cross_validate(model, data, labels, folds=10, val_prop=0.2, val_data=None, val_labels=None, epochs=100, batch_size=128, learning_rate=0.1, penalty=0, early_stopping=100, verbose=False):

    dummy = model

    print("Performing {}-fold cross-validation...".format(folds))

    # Split data in preparation for cross-validation...
    data_sets = make_sets(data, labels, [1/folds]*folds)

    accs = np.zeros(folds)

    for fold in range(folds):

        # Isolate validation set
        X_val, y_val = data_sets[2*fold], data_sets[2*fold + 1]
        # Training set is ever-so-slightly trickier!
        X_train = np.vstack([data_set for j, data_set in enumerate(data_sets[::2]) if j != fold])
        y_train = np.concatenate([data_set for j, data_set in enumerate(data_sets[1::2]) if j != fold])

        dummy.reset()

        success = dummy.train(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, penalty, early_stopping, verbose)

        print("Fold: {0} | Best epoch: {1} | Validation accuracy: {2}%".format(
            fold+1,
            dummy.best_epoch[0],
            dummy.best_epoch[2]*100
        ))

        accs[fold] = dummy.best_epoch[2]
    
    print("Average validation accuracy of {}%".format(round(accs.mean()*100, 2)))
    print("")

    return accs.mean()