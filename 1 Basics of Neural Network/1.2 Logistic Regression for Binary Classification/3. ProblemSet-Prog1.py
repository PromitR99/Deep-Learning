#numpy is the fundamental package for scientific computing with Python.
import numpy as np
#h5py is a common package to interact with a dataset that is stored on an H5 file.
import matplotlib.pyplot as plt
#matplotlib is a famous library to plot graphs in Python.
import h5py
#PIL and scipy are used here to test your model with your own picture at the end.
import scipy
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset

#%matplotlib inline

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 20
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")