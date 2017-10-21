# fix random seed for reproducibility
import numpy as np
seed = 9
np.random.seed(seed)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
import matplotlib.pyplot as plt

nb_classes = 10
nb_epoch = 60
batch_size = 64


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Make the value floats in [0;1] instead of int in [0;255]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices (ie one-hot vectors)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Create the model
model = Sequential()
# Adding zero padding to make image 32*32
model.add(ZeroPadding2D((2,2), input_shape=(28, 28, 1))) 
# Add convolution layer with 6 5*5 filters and activation being ReLU
# Output would be of size 28*28*6
model.add(Conv2D(6, (5, 5), activation='relu', padding='valid'))
# Add a max pool layer with stride 2 and region to be 2*2
# Output would be of size 14*14*6
model.add(MaxPooling2D((2,2), strides=(2, 2), padding='valid'))	
# Add convolution layer with 16 5*5 filters and activation being ReLU
# Output would be of size 10*10*16
model.add(Conv2D(16, (5, 5), activation='relu', padding='valid'))
# Add a max pool layer with stride 2 and region to be 2*2
# Output would be of size 5*5*16
model.add(MaxPooling2D((2,2), strides=(2, 2), padding='valid'))
# Flatten the ouput so that fully connected layer can be used
# Output would be of size 400
model.add(Flatten())
# Add a fully connected layer with the activation being ReLU
# Output would be of size 120
model.add(Dense(120, kernel_initializer="uniform", activation='relu'))
# Add a fully connected layer with the activation being ReLU
# Output would be of size 84
model.add(Dense(84, kernel_initializer="uniform", activation='relu'))
# The final layer is a softmax layer
# Output is of size 10 for each class
model.add(Dense(nb_classes, kernel_initializer="uniform", activation='softmax'))

# Compile the model with loss being categorical cross-entropy with optimizer set to adam
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with epoch being set to nb_epoch
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model accuracy on test set
score = model.evaluate(X_test, Y_test)

print "The loss on test set is %f" % score[0]
print "The accuracy on test set is %f" % score[1]