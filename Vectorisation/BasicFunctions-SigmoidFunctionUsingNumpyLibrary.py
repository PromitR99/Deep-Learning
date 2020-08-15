import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1 + np.exp(-x))
    ### END CODE HERE ###
    
    print(s)

#try the following input
x = np.array([1, 2, 3])
sigmoid(x)