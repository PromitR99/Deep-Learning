def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J) presented in Figure 1 (J(theta) = theta * x)
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    
    ### START CODE HERE ### (approx. 1 line)
    J = theta*x
    ### END CODE HERE ###
    
    return J

x, theta = 2, 4
J = forward_propagation(x, theta)
print ("J = " + str(J))

#Output: J = 8

'''
Exercise:
Now, implement the backward propagation step (derivative computation) of Figure 1.
That is, compute the derivative of J(θ)=θx with respect to θ.
To save you from doing the calculus, you should get dtheta = (∂J/∂θ) = x.
'''

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta (see Figure 1).
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    
    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """
    
    ### START CODE HERE ### (approx. 1 line)
    dtheta = x
    ### END CODE HERE ###
    
    return dtheta

x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print ("dtheta = " + str(dtheta))

#Output: dtheta = 2
