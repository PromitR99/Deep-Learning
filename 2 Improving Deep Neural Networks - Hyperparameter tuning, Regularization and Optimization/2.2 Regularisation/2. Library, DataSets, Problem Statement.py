'''
Regularization
Welcome to the second assignment of this week. Deep Learning models have so much flexibility and capacity that overfitting can be a serious problem, if the training dataset is not big enough.
Sure it does well on the training set, but the learned network doesn't generalize to new examples that it has never seen!

You will learn to: Use regularization in your deep learning models.

Let's first import the packages you are going to use.
'''

# import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

'''
Problem Statement: You have just been hired as an AI expert by the French Football Corporation.
They would like you to recommend positions where France's goal keeper should kick the ball so that the French team's players can then hit it with their head.

They give you the following 2D dataset from France's past 10 games.
'''

train_X, train_Y, test_X, test_Y = load_2D_dataset()

'''
Each dot corresponds to a position on the football field where a football player has hit the ball with his/her head after the French goal keeper has shot the ball from the left side of the football field.
- If the dot is blue, it means the French player managed to hit the ball with his/her head
- If the dot is red, it means the other team's player hit the ball with their head
Your goal: Use a deep learning model to find the positions on the field where the goalkeeper should kick the ball.

Analysis of the dataset: This dataset is a little noisy, but it looks like a diagonal line separating the upper left half (blue) from the lower right half (red) would work well.

You will first try a non-regularized model.
Then you'll learn how to regularize it and decide which model you will choose to solve the French Football Corporation's problem.
'''
