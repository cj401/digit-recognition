import cPickle, gzip, numpy

def loadMNIST():
	# Load the dataset
	f = gzip.open('./data/mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	return train_set, valid_set, test_set