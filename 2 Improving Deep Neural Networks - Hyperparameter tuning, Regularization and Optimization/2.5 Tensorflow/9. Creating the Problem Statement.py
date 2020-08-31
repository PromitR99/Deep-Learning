# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

#Change the index below and run the cell to visualize some examples in the dataset.

# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

#y = 5

'''
As usual you flatten the image dataset, then normalize it by dividing by 255.
On top of that, you will convert each label to a one-hot vector as shown in Figure 1.
Run the cell below to do so.
'''

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

'''
Output:
number of training examples = 1080
number of test examples = 120
X_train shape: (12288, 1080)
Y_train shape: (6, 1080)
X_test shape: (12288, 120)
Y_test shape: (6, 120)


Note that 12288 comes from 64×64×3.
Each image is square, 64 by 64 pixels, and 3 is for the RGB colors.
Please make sure all these shapes make sense to you before continuing.

Your goal is to build an algorithm capable of recognizing a sign with high accuracy.
To do so, you are going to build a tensorflow model that is almost the same as one you have previously built in numpy for cat recognition (but now using a softmax output).
It is a great occasion to compare your numpy implementation to the tensorflow one.

The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX.
The SIGMOID output layer has been converted to a SOFTMAX.
A SOFTMAX layer generalizes SIGMOID to when there are more than two classes.
'''
