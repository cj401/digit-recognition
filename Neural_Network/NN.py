import numpy as np
import sys
sys.path.insert(0,"..")
from unpickle import loadMNIST

class NN:
	"""
	The class NN has the member variables defined in __init__
	The main function provided are initialize, fit, predict and accuracy
	There are helper functions for each of them, in which feedforward
	and backprop forms the backbone of the whole class.
	"""
	def __init__(self, train_set, valid_set, neurons=[30, 60], eta=3.0, batch_size=10, iterations=50, K=10):
		"""
		layers describe how many hidden layers will be there in the network.
		neurons is a list containing the number of neurons that have to be there in hidden layers.
		eta is the learning rate for the matrices and biases in backprop algo
		batch_size describes the batch size for stochastic update.
		iterations is the number of iterations feedforward and backprop will be called.
		K is the number of classes, in this case it is 10 but the code will work
		for any arbitrary number of classes.
		Initializes the network parameters, and keeps the data as memeber variable.
		Network parameters can be given as arguments otherwise default
		parameters will be used.
		"""

		# Extract the data and labels from training set 
		self.K = K

		self.x_train = train_set[0]
		y_train = train_set[1]
		self.Y_train = self.one_hot(y_train) # One hot encode the target variable

		self.layers = len(neurons)
		self.eta = eta 
		self.batch_size = batch_size
		self.iter = iterations

		s = self.x_train.shape[1]
		neurons.insert(0, s)
		neurons.append(K)

		self.neurons = neurons

		# Extract the data and labels from validation set 
		self.x_valid = valid_set[0]
		self.y_valid = valid_set[1]


	def one_hot(self, y):
		"""
		y is array of single scalar values selected from {0...9}
		K is the number of classes, so that if classes are increased the, function still works
		The functions returns a one hot encoded representation,
		in which each row is non-zero(one) given by values in y
		"""
		Y = np.zeros((y.shape[0], self.K))
		Y[np.arange(y.shape[0]), y] = 1.0

		return Y

	def sigmoid(self, x):
		return 1/(1.0 + np.exp(-x))

	def sigmoid_derv(self, x):
		y = self.sigmoid(x)
		return y * (1 - y)

	def relu(self, x):
		return np.where(x>=0, x, 0)

	def relu_derv(self, x):
		return np.where(x>=0, 1.0, 0)

	def g(self, x):
		return self.sigmoid(x)

	def dg(self, x):
		return self.sigmoid_derv(x)

	def init_network(self):
		"""
		matrix list stores the weight matrix associated with each of the layers
		bias list stores the bias term associated with each of the layer
		a_list stores the activations of each layer during the feedforward step
		z_list stores the vectors before activations during the feedforward step

		"""
		self.matrix_list = list()
		self.bias_list = list()
		self.a_list = list()
		self.z_list = list()

		np.random.seed(0)

		# Initialiazation of matrix and bias associated with each layer
		self.matrix_list = [np.random.randn(j, l)  for j, l in zip(self.neurons[1:], self.neurons[:-1])]
		self.bias_list = [np.random.randn(j, 1) for j in self.neurons[1:]]

	def feedforward(self, dataset, trainORVal):
		"""
		dataset contains the data points in rows.
		trainORVal is a boolean value to distinguish between training and validation,
		True implies training and False implies validation
		The function returns the output from the final layer, which are class scores
		"""
		# Emptying the list
		self.a_list = list()
		self.z_list = list()

		# Feeding the data depending on trainORVal
		if trainORVal:
			# Creating a batch of data
			data = dataset.T
		else:
			data = self.x_valid.T

		self.a_list.append(data)

		output = data
		# Matrix multiplicatioin followed by non-linearity
		for w, b in zip(self.matrix_list, self.bias_list):
			# The output is the input for the next iteration
			z = np.dot(w, output) + b
			output = self.g(z)
			self.a_list.append(output)
			self.z_list.append(z)
		# return only the output of final layer, which are class scores
		return output

	def backprop(self, j,l):
		"""
		j and l describes the starting and ending point to slice the data(hacky way of implementing batch_size)
		The function calculates the gradient of all matrices and biases, and updates them at the end 
		"""
		# dw will store the gradient w.r.t matrix of each layer after each feedforward step
		# db will store the gradient w.r.t bias of each the layer after each feedforward step
		# Emptying the previous gradients  
		dw = list()
		db = list()

		# Implementaion of the backprop algorithm, it works after 2 sleepless nights
		# Best description I have found is given in the link below 
		# https://sudeepraja.github.io/Neural/
		da = self.a_list[-1] - self.Y_train[j:l][:].T
		da_dz = self.dg(self.z_list[-1])
		delta = da * da_dz
		dw.append(np.dot(delta, self.a_list[-2].T))
		db.append(np.sum(delta, axis=1, keepdims=True))

		for i in range(1, self.layers + 1):
			da = np.dot(self.matrix_list[-i].T, delta)
			da_dz = self.dg(self.z_list[-i - 1])
			delta = da * da_dz
			dw.append(np.dot(delta, self.a_list[-i - 2].T))
			db.append(np.sum(delta, axis=1, keepdims=True))

		dw.reverse()
		db.reverse()

		# Update of the matrices and bias terms, with eta as the learning rate
		self.matrix_list = [(original - (self.eta/self.batch_size)*grad) for original, grad in zip(self.matrix_list, dw)]
		self.bias_list = [(original - (self.eta/self.batch_size)*grad) for original, grad in zip(self.bias_list, db)]

	def fit(self):
		"""
		The function which trains by first creating the mini-batches, and doing first the forward pass,
		then the backward pass to update the weights. 
		"""
		for i in range(0, self.iter):
			print i
			self.validate()
			for j in range(0, self.x_train.shape[0] - self.batch_size, self.batch_size):
				self.feedforward(self.x_train[j:j+self.batch_size-1,:], True) # True, because we are doing training right now
				self.backprop(j, j+self.batch_size-1)

	def validate(self):
		"""
		Function to tune the parameters using validation dataset.
		It does a forward pass with the learned weights,
		finds the class from class scores.
		"""
		# Feedforward step to get the class scores
		classScores = self.feedforward(None, False)
		# Picking the class with maximum score
		prediction = np.argmax(classScores, axis=0)

		difference = prediction - self.y_valid

		correct = np.sum(np.where(difference == 0, 1.0, 0.0))

		print "The accuracy on validation set is: ", correct/self.y_valid.shape[0]

	def store(self):
		"""
		It stores the weights learned from training in a .npz format, 
		so that NN_test can read the weights and make prediction.
		It makes sure that we are not using test data during training.
		Right now the saving method is very primeval. Further work can be done.
		"""
		np.savez("weights.npz", matrix1=self.matrix_list[0],matrix2=self.matrix_list[1],matrix3=self.matrix_list[2],\
			bias1=self.bias_list[0],bias2=self.bias_list[1],bias3=self.bias_list[2])

	def predict(self, x_test):
		"""
		Function to predict the classes given the data points.
		One feedforward step is done, using the learned weights.
		"""
		# FeedForward step to get the class scores
		classScores = self.feedforward(x_test, True)
		# Picking the class with maximum score
		prediction = np.argmax(classScores, axis=0)

		return prediction

	def accuracy(self, prediction, y_test):
		"""
		Prints the ratio of same entries to total entries
		"""
		difference = prediction - y_test

		correct = np.sum(np.where(difference == 0, 1.0, 0.0))

		print "The accuracy on test set is: ", correct/y_test.shape[0]

if __name__ == "__main__":
	# Unpacking the dataset of train, validation and testing data
	train_set, valid_set, test_set = loadMNIST('../data/mnist.pkl.gz')
	hidden_neurons = [30, 80, 50]
	A = NN(train_set, valid_set, neurons=hidden_neurons)
	A.init_network()
	A.fit()
	print "Neurons are", hidden_neurons[1] 
	prediction_test = A.predict(test_set[0])
	A.accuracy(prediction_test, test_set[1])