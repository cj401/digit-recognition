# Prediction on PIMA-indian-diabetes dataset using a simple logistic regression model
## Dataset preparation
* The dataset considered of 9 values in each row
* Target label were extracted from last column
* Features were normalized to have mean 0 and unit variance
* Data was split into train and test with test having .33 percentage of total

## Input/Output format
* Input consist of a 7 dimensional vector with all the values being continous
* Target variable is a real valued scalar which is 1 if the person has diabetes and 0 if not

## Initial model
* Dense layer is used with output dimension being set to 1, and activation being sigmoid
* loss was taken as binary_crossentropy, optimizer being rmsprop, metric being accuracy
* Whilst training epoch were set to 60 and batch size to 10

### Results
* The loss value on train data at the end is 0.4587
* The accuracy on train data is 0.7762
* The loss value on test data at the end is 0.7834
* The accuracy on test data is 0.5132

### Plots
* Major concern in the plots was that test accuracy seemed to be better than training
* It is due to the fact that at each epoch keras average out the trainig accuracy  over all batches
* Test accuracy is calculated after training is done over one epoch
* Accuracy plot is saved as bin_cross_entropy_acc.png
* Loss plot is saved as bin_cross_entropy_loss.png

## Model with MSE loss
* The loss function was changed to mean_squared_error
* Other parameters were as before

### Results
* The loss value on train data at the end is 0.1507
* The accuracy on train data is 0.7782
* The loss value on test data at the end is 0.7874
* The accuracy on test data is 0.1604
* There are 34% labels which are 1, so it indicates that are model is better than just ouptputting all 0's

### Plots
* Accuracy plot is saved as mean_squared_acc.png
* Loss plot is saved as mean_squared_loss.png

## Model with weight decay

* There was no improvement adding weight decay

## Using K fold Cross Validation
* Due to small size of dataset the above result are not reliable
* The data was split into 10 segments and at each iteration one was choosen as test set
* All the above techniques are going to performed again but without detailed analysis, only the result will be stated
* The optimizer is changes to adam for fast convergence and batch size is also increased to 20

### Result for simple logistic model
Accuracy over the iteration: 84.42%, 77.92%, 76.62%, 77.92%, 71.43%, 72.73%, 79.22%, 71.43%, 80.26%, 80.26%
Mean accuracy is 77.22% (+/- 4.04%)

### Result for MSE loss
Accuracy over the iteration: 85.71% , 77.92% , 76.62% , 76.62% , 70.13% , 72.73% , 76.62% , 71.43% , 77.63% , 80.26%
Mean accuracy is 76.57% (+/- 4.28%)

### Result with regularization
No improvement, the accuracy went down

### Result with 1 hidden layer
1 neuron in hidden layer
	Accuracy over the iteration: 75.32% , 74.03% , 72.73% , 74.03% , 67.53% , 64.94% , 74.03% , 68.83% , 75.00% , 77.63% 
	Mean accuracy is  72.41% (+/- 3.78%)
2 neuron in hidden layer
	Accuracy over the iteration: 84.42% , 76.62% , 77.92% , 76.62% , 71.43% , 72.73% , 80.52% , 72.73% , 81.58% , 80.26%  
	Mean accuracy is  77.48% (+/- 4.07%)
3 neuron in hidden layer
	Accuracy over the iteration:  83.12% , 76.62% , 76.62% , 74.03% , 72.73% , 71.43% , 77.92% , 72.73% , 76.32% , 80.26% ,  
	Mean accuracy is  76.18% (+/- 3.45%)\
4 neuron in hidden layer
	Accuracy over the iteration:  85.71% , 75.32% , 76.62% , 76.62% , 71.43% , 71.43% , 76.62% , 72.73% , 76.32% , 80.26% ,  
	Mean accuracy is  76.31% (+/- 4.07%)
5 neuron in hidden layer
	Accuracy over the iteration:  85.71% , 75.32% , 76.62% , 76.62% , 70.13% , 72.73% , 76.62% , 72.73% , 76.32% , 80.26% ,  
	Mean accuracy is  76.31% (+/- 4.11%)


# Making a logistic regression model to predict the outcomes in mnist datset which can be loaded by keras automatically.

## Input/Output format
* Input is 786(28*28) dimensional vector
* Input vector values belonging to {0, 1, 2,..255} are scaled back by dividing with 255 
* The labels are converted from real belonging to {1,2,..9} to a 10 dimensional vector using to_categorical()
* Labels are basically one-hot encoded  

## Initial model

* Dense layer is used for the input, output dimension being equal to the number of classes i.e. 10
* Softmax is used as a activation function.(Generalization of logistic regression)
* loss funstion is categorical cross entropy
* Optimizer is sgd(Stochastic Gradient Descent) with default parameters
* epoch is set to 20
* batch_size is set to 128

## Results

* The loss value on train data at the end is 0.3537
* The accuracy on train data is 0.9027
* The loss value on test data at the end is 0.3357
* The accuracy on test data is 0.9096

# 2nd model

* The loss function is changed to mean_squared_error
* To get a repectable accuracy on the test set the epochs were increased to 40

## Results

* The loss value on train data at the end is 0.0297
* The accuracy on train data is 0.8431
* The loss value on test data at the end is 0.02844
* The accuracy on test data is 0.8537

## LeNet Architecture

### Pre-Processing
* Minimal pre-processing is done
* The images are padded with zero to make them 32*32
* The gray scale value are scaled between [0,1]
* The labels are one-hot encoded

### Architecture

Conv(28*28*6) => ReLU(28*28*6) => MaxPool(14*14*6) => Conv(10*10*16) => ReLU(10*10*16) => MaxPool(5*5*16) => Flatten(400) => Dense(120) => Dense(84) => Softmax(10)

* Convolution layer which takes 32*32 pixel images
* Stride of Conv layer is 1, with filter size being 5*5
* Activation is taken to be ReLU
* MaxPooling layer is used with pool size 2*2  and stride 2
* Dense Layer also has activation set to ReLU

### Hyper-Parameters
* batch size is set to 64
* epochs are set to 60

### Result
* The loss on training set is 4.6411e-04
* The accuracy on trainig set is 0.9999
* The loss on test set is 0.059603
* The accuracy on test set is 0.9915

## LeNet with Dropout
Dropout layer was introduced after both fully connected layer with = 0.5
epoch = 80
First fully connected layer has 240 neurons

### Result
* The loss on training set is 0.0120
* The accuracy on trainig set is 0.9970
* The loss on test set is 0.047266
* The accuracy on test set is 0.9938

## LeNet inspired variant with smaller Architecture
* Images are not padded with zeros  other than that pre-processing step is same

### Architecture

Conv(24*24*6) => sigmoid(24*24*6) => MaxPool(12*12*6) => Flatten(864) => Dense(100) => Softmax(10) 

* Convolution layer which takes 28*28 pixel images
* Stride of Conv layer is 1, with filter size being 5*5
* Number of filter are 6
* Activation is taken to be ReLU
* MaxPooling layer is used with pool size 2*2  and stride 2
* Dense Layer also has activation set to sigmoid

### Results
* The loss on training set is 9.1778e-04
* The accuracy on trainig set is 1.0
* The loss on test set is 0.047136
* The accuracy on test set is 0.988400

## Another LeNet inspired Architecture

### Architecture

Conv(24*24*20) => ReLU(24*24*20) => MaxPool(12*12*20) => Conv(8*8*40) => ReLU(8*8*40) => MaxPool(4*4*40) => Flatten(640) => Dense(100) => Softmax(10) 

* Convolution layer which takes 28*28 pixel images
* Stride of Conv layer is 1, with filter size being 5*5
* Number of filter are 6
* Activation is taken to be ReLU 
* MaxPooling layer is used with pool size 2*2  and stride 2
* Dense Layer also has activation set to sigmoid

### Results
* The loss on training set is 2.6943e-04
* The accuracy on trainig set is 1.0000   
* The loss on test set is 0.050776
* The accuracy on test set is 0.993100