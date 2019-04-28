# Sources
'''
    Tutorial:       'https://elitedatascience.com/keras-tutorial-deep-learning-in-python'
    Dropout method: https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning-And-why-is-it-claimed-to-be-an-effective-trick-to-improve-your-network
    Entropy Ex:     https://bricaud.github.io/personal-blog/entropy-in-decision-trees/
                    https://www.math.unipd.it/~aiolli/corsi/0708/IR/Lez12.pdf

'''

# LOAD LIBRARIES_________________________________________________________________
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

# Set Random Seed
np.random.seed(123)


# Import Sequantial Model Type
'Simple linear stack of neural network layers.  Good for feed-forward CNN'
from keras.models import Sequential

# Import Core Layers From Keras
from keras.layers import Dense, Dropout, Activation, Flatten

'''
    Dense:          implements the operation: output = activation(dot(input, kernel) + bias)
    Dropout:        consists in randomly setting a fraction rate of input units to 0.  
                    helps prevent overfitting. 
    Activation:     element-wise-activation function pass as the activation argument.
    Flatten:        flattens the inputs. 
    kernel:         weights matrix created by the layer.

    ref:            Dropout - prevent overfitting
                    http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf 
    flattening exp  https://www.quora.com/What-is-the-meaning-of-flattening-step-in-a-convolutional-neural-network'''


# Import CNN Layers From Keras
from keras.layers import Convolution2D, MaxPooling2D

# Import utilities 
from keras.utils import np_utils




# IMPORT & INSPECTING IMAGES______________________________________________________

# Import Images From MNIST
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Shape X_train => {}'.format(X_train.shape))      # 60k samples, images of 28X28 pixels


# Plot first sample image of X_train
plt.imshow(X_train[0])
#plt.show()




# PREPROCESSING INPUT DATA FOR KERAS_____________________________________________
'''Need to transform data set from 
        (n, width, height) to 
        (n, depth, width, height)
   This is due to the fact that an image that is black and white vs color will have a diff
   dimension.  Color has a depth of 3. This dimension needs to be explicitly declared in keras.
'''

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)  # the 1 refers to the dimension
X_test =  X_test.reshape(X_test.shape[0], 1, 28, 28)    
print('X_train new shape => {}'.format(X_train.shape))



# Final Step - Convert data to data type float32 andnormalize to range [0,1]
'https://stackoverflow.com/questions/43440821/the-real-difference-between-float32-and-float64'
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255                         # This converts them to values that range betw 0-1?
X_test /= 255




# PROCESS CLASS LABELS FOR KERAS__________________________________________________

# Convert 1-dimensional class array to 10-dimensional class matrices
print('Y_train shape {}'.format(y_train.shape))
Y_train = np_utils.to_categorical(y_train, 10)
Y_test =  np_utils.to_categorical(y_test, 10)
print('Y_train shape after conversion {}'.format(Y_train.shape))



# DEFINE MODEL ARCHITECTURE______________________________________________________

# Declare sequential model
model = Sequential()
# Declare input layers
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
'First 3 parameters correspond to the (depth, width, height)'
print('Model output shape {}'.format(model.output_shape))

# Add More Layers
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
'https://computersciencewiki.org/index.php/Max-pooling_/_Pooling'

# Finalize Model - Add a Fully Connected Layer and Output Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
''' Dense:      First parameter is the output size.
    Output:     Size is 10 as that is the number of categories that we have
    Softmax:    https://en.wikipedia.org/wiki/Softmax_function
    
    *note:      sigmoid function is used for the two-class logistic regression. 
                softmax fucntion is used for multiclass logistic regression. 
    '''

# Compile Model
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', 
              metrics=['accuracy'])


# Fit Model on Training Data
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
'epoch:     number of times the NN is trained on the data and the weights updated'



# Evaluate Model on Test
score = model.evaluate(X_test, Y_test, verbose=0)
print('Score => {}'.format(score))






















