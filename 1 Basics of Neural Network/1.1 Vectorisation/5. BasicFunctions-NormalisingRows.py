import numpy as np

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)

    print(x_norm)
    
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###

    return x

#try the following input
x = np.array([[0, 3, 4],
              [1, 6, 4]])
r = normalizeRows(x)
print(r)