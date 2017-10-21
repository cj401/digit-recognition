import pandas as pd

def load_sonar():
	df = pd.read_csv("sonar.csv", header=None)
	dataset = df.values

	train = dataset[:150, :]
	test = dataset[150:, :]

	X_train = train[:, :60].astype(float)
	Y_train = train[:, 60]
	X_test = test[:, :60].astype(float)
	Y_test = test[:, 60]

	print Y_train

	return X_train, Y_train, X_test, Y_test