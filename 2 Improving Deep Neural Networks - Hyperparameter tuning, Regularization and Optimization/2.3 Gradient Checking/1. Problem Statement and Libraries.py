'''
Gradient Checking
Welcome to this week's third programming assignment!
You will be implementing gradient checking to make sure that your backpropagation implementation is correct.
By completing this assignment you will:

- Implement gradient checking from scratch.

- Understand how to use the difference formula to check your backpropagation implementation.

- Recognize that your backpropagation algorithm should give you similar results as the ones you got by computing the difference formula.

- Learn how to identify which parameter's gradient was computed incorrectly.


Gradient Checking

Welcome to the final assignment for this week!
In this assignment you will learn to implement and use gradient checking.

You are part of a team working to make mobile payments available globally, and are asked to build a deep learning model to detect fraud--whenever someone makes a payment, you want to see if the payment might be fraudulent, such as if the user's account has been taken over by a hacker.

But backpropagation is quite challenging to implement, and sometimes has bugs.
Because this is a mission-critical application, your company's CEO wants to be really certain that your implementation of backpropagation is correct.
Your CEO says, "Give me a proof that your backpropagation is actually working!"
To give this reassurance, you are going to use "gradient checking".

Let's do it!
'''

# Packages
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
