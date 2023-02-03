
# importing sys
import sys
from Assignment_1.Utils import Data_Utils

# add the path of the Assignment_1 folder to the sys.path
sys.path.append('Assignment_1')


from PreProcessor import ImagePreProcessor
from Utils.unpickle import unpickle
from Utils import Info


import numpy as np
from matplotlib import pyplot as plt

def test_data(data):
    assert data is not None, "Data is None"
    assert data[b'data'] is not None, "Data[b'data'] is None"
    assert data[b'labels'] is not None, "Data[b'labels'] is None"
    assert len(data[b'data']) > 0, "Data[b'data'] is empty"
    assert len(data[b'labels']) > 0, "Data[b'labels'] is empty"

    assert len(data[b'data']) == len(data[b'labels']), "Data[b'data'] and Data[b'labels'] are not the same length"

    assert len(data[b'data'][0]) == 3072, "Data[b'data'][0] is not 3072"
    assert data[b'data'][0][0].dtype == np.uint8, "Data[b'data'][0][0] is not np.uint8"
    assert type(data[b'data'][0]) == np.ndarray, "Data[b'data'][0] is not np.ndarray"
    assert data[b'data'][0].shape == (3072,), "Data[b'data'][0] is not (3072,)"
    assert data[b'data'].shape == (10000, 3072) , "Data[b'data'] shape is not (10000, 3072)"

    assert len(data[b'labels']) == 10000, "Data[b'labels'][0] is not 10000"
    assert type(data[b'labels'][0]) == int, "Data[b'labels'][0] is not int"

def test_data_all(data):
    assert data is not None, "Data is None"
    assert data[b'data'] is not None, "Data[b'data'] is None"
    assert data[b'labels'] is not None, "Data[b'labels'] is None"
    assert len(data[b'data']) > 0, "Data[b'data'] is empty"
    assert len(data[b'labels']) > 0, "Data[b'labels'] is empty"

    assert len(data[b'data']) == len(data[b'labels']), "Data[b'data'] and Data[b'labels'] are not the same length"

    assert len(data[b'data'][0]) == 3072, "Data[b'data'][0] is not 3072"
    assert data[b'data'][0][0].dtype == np.uint8, "Data[b'data'][0][0] is not np.uint8"
    assert type(data[b'data'][0]) == np.ndarray, "Data[b'data'][0] is not np.ndarray"
    assert data[b'data'][0].shape == (3072,), "Data[b'data'][0] is not (3072,)"
    assert data[b'data'].shape == (50000, 3072) , "Data[b'data'] shape is not (10000, 3072)"

    assert len(data[b'labels']) ==50000, "Data[b'labels'][0] is not 50000"
    assert type(data[b'labels'][0]) == int, "Data[b'labels'][0] is not int"

def test_labels(labels):
    assert labels is not None, "Labels is None"
    assert labels[b'label_names'] is not None, "Labels[b'label_names'] is None"
    assert len(labels[b'label_names']) == 10, "Labels[b'label_names'] is not 10"
    assert type(labels[b'label_names'][0]) == bytes, "Labels[b'label_names'][0] is not of type bytes"

if __name__ == "__main__":
    labels = Data_Utils.get_labels()
    dict = Data_Utils.get_train_data(1)
    dict_all = Data_Utils.get_train_data_all()

    Info.print_dict_info(dict)
    Info.print_labels_info(labels)
    Info.print_dict_info(dict_all)

    test_data(dict)
    test_labels(labels)
    test_data_all(dict_all)
