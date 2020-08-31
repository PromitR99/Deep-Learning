def gradient_check(x, theta, epsilon = 1e-7):
    """
    Implement the backward propagation presented in Figure 1.
    
    Arguments:
    x -- a real-valued input
    theta -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    ### START CODE HERE ### (approx. 5 lines)
    thetaplus = theta + epsilon                    # Step 1
    thetaminus = theta - epsilon                   # Step 2
    J_plus = forward_propagation(x, thetaplus)     # Step 3
    J_minus = forward_propagation(x, thetaminus)   # Step 4
    gradapprox = (J_plus - J_minus)/(2*epsilon)    # Step 5
    ### END CODE HERE ###
    
    # Check if gradapprox is close enough to the output of backward_propagation()
    ### START CODE HERE ### (approx. 1 line)
    grad = backward_propagation(x, theta)
    ### END CODE HERE ###
    
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)                    # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator/denominator                               # Step 3'
    ### END CODE HERE ###
    
    if difference < 1e-7:
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")
    
    return difference

x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))

'''
Output:
The gradient is correct!
difference = 2.91933588329e-10


Congrats, the difference is smaller than the 10^−7 threshold.
So you can have high confidence that you've correctly computed the gradient in backward_propagation().

Now, in the more general case, your cost function J has more than a single 1D input.
When you are training a neural network, θ actually consists of multiple matrices W^[l] and biases b^[l]!
It is important to know how to do a gradient check with higher-dimensional inputs.
Let's do it!
'''
