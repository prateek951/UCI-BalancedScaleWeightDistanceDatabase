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
