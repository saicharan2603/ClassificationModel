# Testing the Feature Vector Generator 
# importing sys
import sys
from Assignment_1.Utils import Data_Utils

# add the path of the Assignment_1 folder to the sys.path
sys.path.append('Assignment_1')

from PreProcessor.FeatureVectorGenerator import generate_feature_vector, resize
from Tests import PickleTest
from Utils import Info
import numpy as np

def test_feature_vector_data(original_data: np.ndarray, feature_vector_data):
    assert feature_vector_data is not None, "Feature Vector Data is None"
    assert original_data is not None, "Original Data is None"
    assert type(feature_vector_data) == np.ndarray, "Feature Vector Data is not np.ndarray"

    assert feature_vector_data.shape[0] == original_data.shape[0], "Feature Vector Data and Original Data are not the same length"
    assert feature_vector_data.shape[1] == 512, "Feature Vector Data is not 512"
    assert feature_vector_data.dtype == np.float32, "Feature Vector Data is not np.float32"
    #assert np.max(feature_vector_data) <= 1 , "Feature Vector Data is greater than 1"

def test_one_hot_encoding(labels: list, one_hot_encoded_labels: np.ndarray):
    assert labels is not None, "Labels is None"

    assert one_hot_encoded_labels is not None, "One Hot Encoded Labels is None"

    assert one_hot_encoded_labels.shape[0] == len(labels), "One Hot Encoded Labels and Labels are not the same length"
    assert one_hot_encoded_labels.shape[1] == max(labels)+1, f"One Hot Encoded Labels is not {max(labels)+1}"
    assert one_hot_encoded_labels.dtype == np.float32, "One Hot Encoded Labels is not np.float32"

def test_resize(image: np.ndarray, from_shape = (3, 32, 32), to_shape = (3, 224, 224)):
    assert image is not None, "Image is None"
    assert type(image) == np.ndarray, "Image is not np.ndarray"

    assert image.dtype == np.uint8, "Image is not np.uint8"

    resized_image = resize(image, from_shape, to_shape)

    assert resized_image is not None, "Resized Image is None"
    assert type(resized_image) == np.ndarray, "Resized Image is not np.ndarray"

    assert resized_image.shape == to_shape, "Resized Image shape is not (3, 224, 224)"
    assert resized_image.dtype == np.float32, "Resized Image is not np.float32"

if __name__ == "__main__":
    # get the test data
    test_data = Data_Utils.get_test_data()

    # test the test data
    PickleTest.test_data(test_data)

    # selecting the subset of the test data
    test_data = test_data[b'data'][:1024]

    # test the resize function
    test_resize(test_data[0])

    # get the feature vector data
    feature_vector_data = generate_feature_vector(test_data)

    # test the feature vector data
    test_feature_vector_data(test_data, feature_vector_data)

    # print the feature vector data
    print(feature_vector_data)

