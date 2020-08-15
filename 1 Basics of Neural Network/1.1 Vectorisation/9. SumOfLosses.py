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
    loss_sum = loss.sum()
    print(loss_sum)

yhat = np.array([0.98, 0.07, 0.69, 0.19])
y = np.array([1, 0, 1, 0])
L1(yhat, y)