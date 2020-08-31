def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)                # Step 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon           # Step 2
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))         # Step 3
        ### END CODE HERE ###
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)               # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon         # Step 2        
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))       # Step 3
        ### END CODE HERE ###
        
        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
        ### END CODE HERE ###
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)                                 # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)               # Step 2'
    difference = numerator/denominator                                            # Step 3'
    ### END CODE HERE ###

    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference

X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)

#There is a mistake in the backward propagation! difference = 0.285093156654

'''
It seems that there were errors in the backward_propagation_n code we gave you! Good that you've implemented the gradient check.
Go back to backward_propagation and try to find/correct the errors (Hint: check dW2 and db1).
Rerun the gradient check when you think you've fixed it.
Remember you'll need to re-execute the cell defining backward_propagation_n() if you modify the code.

Can you get gradient check to declare your derivative computation correct?
Even though this part of the assignment isn't graded, we strongly urge you to try to find the bug and re-run gradient check until you're convinced backprop is now correctly implemented.

Note:-
- Gradient Checking is slow! Approximating the gradient with (∂J/∂θ)≈(J(θ+ε)−J(θ−ε))/2ε is computationally costly.
For this reason, we don't run gradient checking at every iteration during training.
Just a few times to check if the gradient is correct.
-Gradient Checking, at least as we've presented it, doesn't work with dropout.
You would usually run the gradient check algorithm without dropout to make sure your backprop is correct, then add dropout.
Congrats, you can be confident that your deep learning model for fraud detection is working correctly!
You can even use this to convince your CEO. :)

What you should remember from this notebook:

- Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
- Gradient checking is slow, so we don't run it in every iteration of training.
You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.
'''
