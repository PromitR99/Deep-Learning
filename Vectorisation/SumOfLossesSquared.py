# GRADED FUNCTION: L1
import numpy as np

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    loss = abs(y - yhat)
    loss_squared = loss**2
    loss_squared_sum = loss_squared.sum()
    print(loss_squared_sum)

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
L1(yhat, y)