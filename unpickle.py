import cPickle, gzip, numpy

def loadMNIST(file_path):
	# Load the dataset
	f = gzip.open(file_path, 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()
	return train_set, valid_set, test_set