# SOURCE
'''
    Tutorial:       https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn
 vizual rep NN:  https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

'''

# Load Libraries
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# Setwd
target_wd = '/home/ccirelli2/Desktop/Datasets/sentiment labelled sentences/sentiment labelled sentences'
os.chdir(target_wd)


# Load data
yelp_data = pd.read_csv('yelp_labelled.txt', names = ['sentence', 'label'], sep='\t')
amazon_data = pd.read_csv('amazon_cells_labelled.txt', names = ['sentence', 'label'], sep='\t')
imdb_data_data = pd.read_csv('imdb_labelled.txt', names = ['sentence', 'label'], sep='\t')


# BAG OF WORDS (BOW) MODEL______________________________________________________________

# Vectorize Data
'Notes:  tokenizes and then enumerates the text'
ex_sentence = ['Today is a good day to code', 'Yesterday was also a good day to code']
vectorizer = CountVectorizer(min_df=0, lowercase=False)
v_fit = vectorizer.fit(ex_sentence)
v_vocab = vectorizer.vocabulary_
'similar approach'
test = list(enumerate(ex_sentence[0].split(' ')))

# Transform To Array Binary values
v_transf = vectorizer.transform(ex_sentence).toarray()
#print(v_transf)



# DATA PREP____________________________________________________________________________
'''
Activation Function:   Customary to use ReLU for the hidden layers, a sigmoid or softmax 
                        for the output layers (depending on whether or target is binary or multi

'''
# Separate X / Y
from sklearn.model_selection import train_test_split

y = yelp_data['label'].values
sentences = yelp_data['sentence'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
                    sentences, y, test_size=0.25, random_state=1000)


# Vectorize
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

x_train = vectorizer.transform(sentences_train)
x_test  = vectorizer.transform(sentences_test)



# MODEL_____________________________________________________________________
from keras.models import Sequential
from keras import layers

# Convert X Values Into an Array
input_dim = x_train.shape[1]            # Number of features
input_dim

# Instantiate Model
model = Sequential()

# Add Layers
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Specify Optomizer
model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
# Prints Summary of NN Structure
model.summary()

# Set Number of Epochs
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    verbose=False, 
                    validation_data=(x_test, y_test),
                    batch_size=10)

# Measure Accuracy of Model
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print('Training Accuracy: {}'.format(round(accuracy,4)))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print('Test Accuracy: {}'.format(round(accuracy, 2)))





# Plot 
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
plt.show()













