from keras.datasets import mnist
from utils import load_sonar
from keras.utils import np_utils as util
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 9
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_test[1:2,:]
input_dim = 28*28
no_classes = 10
output_dim = 10
epoch = 40
batch_size = 128

X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

Y_train = util.to_categorical(y_train, no_classes)
Y_test = util.to_categorical(y_test, no_classes)


model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))


model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_split=0.33, batch_size=32, nb_epoch=epoch, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=1)
# print type(history)
print history.history["acc"]
print history.history["loss"]
print history.history["val_acc"]
print history.history["val_loss"]
print "\nTest score is ",score[0]
print "Accuracy on Test data is ",score[1]

# json_string = model.to_json() # as json 
# open('mnist_Logistic_model.json', 'w').write(json_string) 
# yaml_string = model.to_yaml() #as yaml 
# open('mnist_Logistic_model.yaml', 'w').write(yaml_string) 

# save the weights in h5 format 
# model.save_weights('mnist_Logistic_wts_2.h5') 

# uncomment the code below (and modify accordingly) to read a saved model and weights 
# model = model_from_json(open('my_model_architecture.json').read())# if json 
# model = model_from_yaml(open('my_model_architecture.yaml').read())# if yaml 

# model.load_weights('my_model_weights.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Softmax Regression MSE accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('softmax_reg_mse_acc.png', bbox_inches='tight')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Softmax Regression MSE loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('softmax_reg_mse_loss.png', bbox_inches='tight')
plt.show()
