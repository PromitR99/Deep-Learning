'''
TensorFlow Tutorial
Welcome to this week's programming assignment. Until now, you've always used numpy to build neural networks.
Now we will step you through a deep learning framework that will allow you to build neural networks more easily.
Machine learning frameworks like TensorFlow, PaddlePaddle, Torch, Caffe, Keras, and many others can speed up your machine learning development significantly.
All of these frameworks also have a lot of documentation, which you should feel free to read. In this assignment, you will learn to do the following in TensorFlow:
-Initialize variables
-Start your own session
-Train algorithms
-Implement a Neural Network
Programing frameworks can not only shorten your coding time, but sometimes also perform optimizations that speed up your code.


1 - Exploring the Tensorflow Library

To start, you will import the library:
'''

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)

'''
Now that you have imported the library, we will walk you through its different applications.
You will start with an example, where we compute for you the loss of one training example.
loss = L(ŷ, y) = (ŷ^(i)−y^(i))^2
'''

y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39

loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
                                                 # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss

#Output: 9

'''
Writing and running programs in TensorFlow has the following steps:
1. Create Tensors (variables) that are not yet executed/evaluated.
2. Write operations between those Tensors.
3. Initialize your Tensors.
4. Create a Session.
5. Run the Session. This will run the operations you'd written above.
Therefore, when we created a variable for the loss, we simply defined the loss as a function of other quantities, but did not evaluate its value.
To evaluate it, we had to run init=tf.global_variables_initializer(). That initialized the loss variable, and in the last line we were finally able to evaluate the value of loss and print its value.

Now let us look at an easy example. Run the cell below:
'''

a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)

#Tensor("Mul:0", shape=(), dtype=int32)

'''
As expected, you will not see 20! You got a tensor saying that the result is a tensor that does not have the shape attribute, and is of type "int32".
All you did was put in the 'computation graph', but you have not run this computation yet.
In order to actually multiply the two numbers, you will have to create a session and run it.
'''

sess = tf.Session()
print(sess.run(c))

#20

'''
Great! To summarize, remember to initialize your variables, create a session and run the operations inside the session.

Next, you'll also have to know about placeholders.
A placeholder is an object whose value you can specify only later.
To specify values for a placeholder, you can pass in values by using a "feed dictionary" (feed_dict variable).
Below, we created a placeholder for x.
This allows us to pass in a number later when we run the session.
'''

x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()

#6

'''
When you first defined x you did not have to specify a value for it.
A placeholder is simply a variable that you will assign data to only later, when running the session.
We say that you feed data to these placeholders when running the session.

Here's what's happening:
When you specify the operations needed for a computation, you are telling TensorFlow how to construct a computation graph.
The computation graph can have some placeholders whose values you will specify only later.
Finally, when you run the session, you are telling TensorFlow to execute the computation graph.
'''
