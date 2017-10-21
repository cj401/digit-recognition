import numpy
# fix random seed for reproducibility
seed = 9
numpy.random.seed(seed)
# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
# load pima indians dataset
dataset = numpy.loadtxt("data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
x = dataset[:,0:8]
Y = dataset[:,8]

scaler = StandardScaler()
X = scaler.fit_transform(x)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=12)

# # # create model
# model = Sequential()
# # # model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
# # # model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
# model.add(Dense(1, input_dim=8, init='zero', activation='sigmoid'))
# # # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # # Fit the model
# history = model.fit(X_train, Y_train, epochs=60, batch_size=1, verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test)
# print "\n"
# # list all data in history
# print "The loss value on train data at the end %f" % history.history["loss"][-1]
# print "The accuracy on train data is %f" % history.history["acc"][-1]
# print "The loss value on test data at the end %f" % score[0]
# print "The accuracy on test data is %f" % score[1]

# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# # plt.savefig("./plots/logistic/mean_squared_acc.png")
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# # plt.savefig("./plots/logistic/mean_squared_loss.png")
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# define 10-fold cross validation test harness
def frange(start, stop, step):
	i = start
	while i < stop:
		yield i
		i += step

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
i = 0
print "Accuracy over the iteration: ",
for train, test in kfold.split(X, Y):
	# create model
	model = Sequential()
	model.add(Dense(2, input_dim=8, init='uniform', activation='sigmoid',W_regularizer=l2(.00	01)))#, W_regularizer=l2(lamb)
	model.add(Dense(1, activation='sigmoid', init="uniform"))
	# model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	history = model.fit(X[train], Y[train], epochs=53, batch_size=20, verbose=0, validation_data=(X[test], Y[test]))
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print "%.2f%% ," % (scores[1]*100),
	cvscores.append(scores[1] * 100)
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig("./plots/logistic/2_neuron_hidden_layer" + str(i) + ".png")
	i = i+1
	# plt.show(block=False)

# plt.show()
print " "
print "Mean accuracy is ",
print "%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores))