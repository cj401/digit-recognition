"""
Softmax regression is implemented with notation followed as in the link given below.
http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
"""
import numpy as np
from unpickle import loadMNIST
import itertools

def one_hot(y, K):
	"""
	y is array of single scalar values selected from {0...9}
	K is the number of classes, so that if classes are increased the, function still works
	The functions returns a one hot encoded representation,
	in which each row is non-zero(one) given by values in y
	"""
	Y = np.zeros((y.shape[0], K))
	Y[np.arange(y.shape[0]), y] = 1.0

	return Y

def hypothesis(theta, X):
	"""
	theta stores the K weight vectors as rows
	X contains the data points in rows.
	The hypothesis for each data point is returned as a vector
	containing probabilites for each of the K class. 
	The hypothesis matrix will contain score in rows for each data point for k classes.
	"""
	hypothesis = np.dot(X, theta.T)
	# For numeric stability
	hypothesis -= np.amax(hypothesis, axis=1)[:, None]
	hypothesis = np.exp(hypothesis)
	# Caluclating the divisor
	totalSum = np.sum(hypothesis, axis=1)
	# hypothesis after division stores the individual class probabilites
	hypothesis = np.divide(hypothesis.T, totalSum).T

	return hypothesis

def cost(theta, X, Y):
	"""
	theta stores the K weight vectors as rows.
	X contains the data points in rows.
	Y is one_hot encoded version of the target values
	The function returns the cost function as given in the above link
	"""
	currentHypo = hypothesis(theta, X)

	# Having only relevant entries
	indicatorHypo = currentHypo*Y

	# Only having to compute log of non-zero terms
	rows, cols = np.nonzero(indicatorHypo)

	cost = -np.sum(np.log(indicatorHypo[rows, cols]))

	return cost

def gradient(theta, X, Y):
	"""
	theta stores the K weight vectors as rows.
	X contains the data points in rows.
	Y is one_hot encoded version of the target values
	The function returns the gradient w.r.t theta matrix
	"""
	first = np.dot(Y.T, X)
	second = np.dot(hypothesis(theta, X).T, X)

	delta = first - second

	return -delta

def fit(X, Y):
	"""
	X contains the data points in rows.
	Y is one_hot encoded version of the target values
	The function iterates until the cost functions does not change by a given threshold
	Function returns the optimum weights as a (k-1)*(28*28) matrix
	"""
	noOfIter = 200 # Total number of iteration
	alpha = 0.00001 # Learning rate

	n = X.shape[1]
	K = Y.shape[1]
	theta = np.zeros((K, n))
	for i in range(0, noOfIter):
		theta -= alpha*gradient(theta, X, Y)

	return theta

def predict(theta, X):
	"""
	theta stores the K weight vectors as rows.
	X contains the data points in rows.
	The function returns the prediction using the softmax classifier as an array
	"""
	hypo = hypothesis(theta, X)

	return np.argmax(hypo, axis=1)

def accuracy(prediction, y):
	"""
	prediction stores the scalar values picked from {0...9}
	y stores the scalar values picked from {0...9}
	The function returns the ratio of correct prediction by total data points
	"""
	diff = prediction - y

	correct = np.sum(np.where(diff == 0, 1.0, 0.0))

	return correct/y.shape[0]


train_set, valid_set, test_set = loadMNIST()

# Extract the data and labels from training set
x_train = np.concatenate((train_set[0], valid_set[0]), axis=0)
y_train = np.concatenate((train_set[1], valid_set[1]))

Y_train = one_hot(y_train, 10) # One hot encode the target variable

theta = fit(x_train, Y_train) # Returns the learned parameters as a matrix 

prediction = predict(theta, test_set[0]) # finding the predictions based on the learned weights

print accuracy(prediction, test_set[1]) # Calculating the accuracy
