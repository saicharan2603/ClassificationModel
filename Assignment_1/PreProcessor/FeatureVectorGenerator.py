# Purpose: Generate feature vector for each image vector
# Path: Assignment_1\PreProcessor\GenerateFeatureVector.py


from PIL import Image
from PreProcessor.feature_extractor import BBResNet18
import numpy as np 

def resize(image, from_shape = (3, 32, 32), to_shape = (3, 224, 224)):
    # resize the image from given shape to given shape
    reshaped_image = image.reshape(from_shape)
    resized_image = np.zeros(to_shape).astype(np.float32)
    for i in range(3):
        # accessing each channel of the image
        img = Image.fromarray(reshaped_image[i], 'L')
        # resizing the image by each channel
        img = img.resize(to_shape[1:], Image.ANTIALIAS)
        # converting the image to numpy array
        resized_image[i] = np.asarray(img).astype(np.float32) / 255

    return resized_image

def generate_feature_vector(data, from_shape = (3, 32, 32), to_shape = (3, 224, 224)):

    resized_data = np.zeros((data.shape[0], to_shape[0], to_shape[1], to_shape[2])).astype(np.float32)
    for i in range(len(data)):
        # resize the image
        resized_data[i] = resize(data[i], from_shape, to_shape)

    # Testing the shape of the resized data
    assert resized_data.shape == (data.shape[0], to_shape[0], to_shape[1], to_shape[2]), f"Resized Data shape is not ({data.shape[0]}, 3, 224, 224)"
    assert np.max(resized_data) <= 1.0, f'Max of resized data = {np.max(resized_data)}' 

    # generate feature vector
    feature_extractor = BBResNet18()
    feature_vector = feature_extractor.feature_extraction(resized_data)
        
    return feature_vector.astype(np.float32)

def one_hot_encoding(labels):
    # one hot encoding of the labels
    one_hot_encoded_labels = np.zeros((len(labels), max(labels)+1))
    one_hot_encoded_labels[np.arange(len(labels)), labels] = 1

    return one_hot_encoded_labels.astype(np.float32)