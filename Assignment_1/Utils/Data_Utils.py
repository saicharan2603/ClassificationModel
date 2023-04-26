'''
Utility function to get the data from the CIFAR-10 dataset
The data is stored in the Assignment_1/Data folder

The function unplicles the data and returns it as a dictionary
'''
from Utils.unpickle import unpickle
from PreProcessor.Augment import get_Augmented_Data
import numpy as np

path0 = "Data/data_batch_"
path1 = "Assignment_1/Data/data_batch_"
path2 = "Data\\data_batch_"
path3 = "Assignment_1\\Data\\data_batch_"

def get_train_data(set=1):
    # by default it returns the first set of data
    
    if set <= 0 or set > 5:
        raise ValueError("Invalid set number")
    try:
        try:
            return unpickle(path0 + str(set))
        except FileNotFoundError:
            return unpickle(path1 + str(set))
    except FileNotFoundError:
        try:
            return unpickle(path2 + str(set))
        except FileNotFoundError:
            return unpickle(path3 + str(set))

def get_train_data_all():
    data = get_train_data(1)
    for i in range(2, 6):
        raw_data = get_train_data(i)
        data[b'data'] = np.vstack((data[b'data'], raw_data[b'data']))
        data[b'labels'] += raw_data[b'labels']

    return data    

def get_train_data_aug():
    return get_Augmented_Data(get_train_data_all())

def get_test_data():
    try:
        try:
            return unpickle("Data/test_batch")
        except FileNotFoundError:
            return unpickle("Assignment_1/Data/test_batch")
    except FileNotFoundError:
        try:
            return unpickle("Data\\test_batch")
        except FileNotFoundError:
            return unpickle("Assignment_1\\Data\\test_batch")


def get_labels():
    try:
        try:
            return unpickle("Data/batches.meta")
        except FileNotFoundError:
            return unpickle("Assignment_1/Data/batches.meta")
    except FileNotFoundError:
        try:
            return unpickle("Data\\batches.meta")
        except FileNotFoundError:
            return unpickle("Assignment_1\\Data\\batches.meta")