import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5000)

def netbp():

    x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
    x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(2, 10)
    
    W2 = 0.5*np.random.randn(2, 2)
    W3 = 0.5*np.random.randn(3, 2)
    W4 = 0.5*np.random.randn(2, 3)
    b2 = 0.5*np.random.randn(2, 1)
    b3 = 0.5*np.random.randn(3, 1)
    b4 = 0.5*np.random.randn(2, 1)


    eta = 0.05
    Niter = 1000
    savecost = np.zeros(Niter)

    for counter in range(Niter):
        k = np.random.randint(10)
        x = np.array([x1[k], x2[k]]).reshape(2, 1)

        a2 = activate(x, W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        delta4 = a4*(1-a4) * (a4 - y[:, k].reshape(2, 1))
        delta3 = a3*(1-a3) * (W4.T @ delta4)
        delta2 = a2*(1-a2) * (W3.T @ delta3)

        W2 -= eta * (delta2 @ x.T)
        W3 -= eta * (delta3 @ a2.T)
        W4 -= eta * (delta4 @ a3.T)

        b2 -= eta * delta2
        b3 -= eta * delta3
        b4 -= eta * delta4
        
        newcost = cost(W2, W3, W4, b2, b3, b4, x1, x2, y)
        savecost[counter] = newcost
    
    plt.plot(savecost)
    plt.show()




def cost(W2, W3, W4, b2, b3, b4, x1, x2, y):
    costvec = np.zeros(10)
    for k in range(10):
        x = np.array([x1[k], x2[k]]).reshape(2, 1)
        a2 = activate(x, W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        costvec[k] = np.sqrt(((y[:, k] - a4)**2).sum())
    return (costvec**2).sum()



def activate(x, W, b):
    return 1 / (1 + np.exp(-((W @ x) + b)))

