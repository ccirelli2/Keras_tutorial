# SOURCE
'https://realpython.com/python-keras-text-classification/#convolutional-neural-networks-cnn'

# Load Libraries
import pandas as pd
import os

# Setwd
target_wd = '/home/ccirelli2/Desktop/Datasets/sentiment labelled sentences/sentiment labelled sentences'
os.chdir(target_wd)


# Load data
yelp_data = pd.read_csv('yelp_labelled.txt', names = ['sentence', 'label'], sep='\t')


print(head(yelp_data))



