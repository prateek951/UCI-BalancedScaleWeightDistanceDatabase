import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function to import the dataset
def import_dataset():
    dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/balance-scale/balance-scale.data',sep=',',header=None)
    # Print the shape for the dataset
    print(dataset.shape)
    # Print the length for the dataset
    print(len(dataset))
    print(dataset.head())
    return dataset

# Function to split the dataset
def split_dataset(dataset):
    X = dataset.values[:,1:5]
    Y = dataset.values[:,0]

    # Perform the split 
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3,random_state=100)

    return X,Y,X_train,X_test,Y_train,Y_test

