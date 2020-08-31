'''
1.1 - Linear function
Lets start this programming exercise by computing the following equation:
Y = WX + b, where W and X are random matrices and b is a random vector.

Exercise:
Compute WX+b where W, X, and b are drawn from a random normal distribution.
W is of shape (4, 3), X is (3,1) and b is (4,1).
As an example, here is how you would define a constant X that has shape (3,1):
X = tf.constant(np.random.randn(3,1), name = "X")

You might find the following functions helpful:
- tf.matmul(..., ...) to do a matrix multiplication
- tf.add(..., ...) to do an addition
- np.random.randn(...) to initialize randomly
'''

def linear_function():
    """
    Implements a linear function: 
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns: 
    result -- runs the session for Y = WX + b 
    """
    
    np.random.seed(1)
    
    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W, X), b)
    ### END CODE HERE ### 
    
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    
    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y)
    ### END CODE HERE ### 
    
    # close the session 
    sess.close()

    return result

print( "result = \n" + str(linear_function()))

'''
Output:
result = 
[[-2.15657382]
 [ 2.95891446]
 [-1.08926781]
 [-0.84538042]]


1.2 - Computing the sigmoid
Great! You just implemented a linear function.
Tensorflow offers a variety of commonly used neural network functions like tf.sigmoid and tf.softmax.
For this exercise lets compute the sigmoid function of an input.

You will do this exercise using a placeholder variable x.
When running the session, you should use the feed dictionary to pass in the input z.
In this exercise, you will have to (i) create a placeholder x, (ii) define the operations needed to compute the sigmoid using tf.sigmoid, and then (iii) run the session.

Exercise:
Implement the sigmoid function below. You should use the following:

- tf.placeholder(tf.float32, name = "...")
- tf.sigmoid(...)
- sess.run(..., feed_dict = {x: z})

Note that there are two typical ways to create and use sessions in tensorflow:

Method 1:

sess = tf.Session()
# Run the variables initialization (if needed), run the operations
result = sess.run(..., feed_dict = {...})
sess.close() # Close the session

Method 2:

with tf.Session() as sess: 
    # run the variables initialization (if needed), run the operations
    result = sess.run(..., feed_dict = {...})
    # This takes care of closing the session for you :)
'''

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32, name = "x")

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above. 
    # You should use a feed_dict to pass z's value to x. 
    with tf.Session() as sess:
        # Run session and call the output "result"
        result = sess.run(sigmoid, feed_dict = {x: z})

    ### END CODE HERE ###
    
    return result

print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))

'''
Output:
sigmoid(0) = 0.5
sigmoid(12) = 0.999994


To summarize, you how know how to:
1. Create placeholders
2. Specify the computation graph corresponding to operations you want to compute
3. Create the session
4. Run the session, using a feed dictionary if necessary to specify placeholder variables' values.
'''
