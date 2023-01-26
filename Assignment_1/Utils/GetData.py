'''
Utility function to get the data from the CIFAR-10 dataset
The data is stored in the Assignment_1/Data folder

The function unplicles the data and returns it as a dictionary
'''

# importing sys
import sys

# add the path of the Assignment_1 folder to the sys.path
sys.path.append('Assignment_1')

from PreProcessor.unpickle import unpickle

def get_train_data(set = 1):
    # by default it returns the first set of data
    if set == 1:
        return unpickle("Assignment_1\\Data\\data_batch_1")
    elif set == 2:
        return unpickle("Assignment_1\\Data\\data_batch_2")
    elif set == 3:
        return unpickle("Assignment_1\\Data\\data_batch_3")
    elif set == 4:
        return unpickle("Assignment_1\\Data\\data_batch_4")
    elif set == 5:
        return unpickle("Assignment_1\\Data\\data_batch_5")
    else:
        raise ValueError("Invalid set number")

def get_test_data():
    return unpickle("Assignment_1\\Data\\test_batch")

def get_labels():
    return unpickle("Assignment_1\\Data\\batches.meta")