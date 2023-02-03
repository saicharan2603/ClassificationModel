'''
Function to return the train and test data
after generating the feature vector and augumenting
'''
from Utils import Data_Utils
from PreProcessor.Augment import get_Augmented_Data
from PreProcessor import FeatureVectorGenerator
import numpy as np

class Data:
    def __init__(self) -> None:
        # importing the raw data by unpickling the data
        raw_data = Data_Utils.get_train_data_all()
        raw_data_aug = Data_Utils.get_train_data_aug()

        # importing the labels
        self.raw_labels = Data_Utils.get_labels()

        # importing the test data
        raw_test_data = Data_Utils.get_test_data()

        # declaring the batch size
        batch_size = 1024
    
        # initializing x vectors
        self.x_train = np.zeros((raw_data[b'data'].shape[0], 512)).astype(np.float32)

        # Generating the feature vector
        for i in range(0, raw_data[b'data'].shape[0], batch_size):
            self.x_train[i:i+batch_size] = FeatureVectorGenerator.generate_feature_vector(raw_data[b'data'][i:i+batch_size])

        self.x_test = np.zeros((raw_test_data[b'data'].shape[0], 512)).astype(np.float32)

        # feature vector for test data
        for i in range(0, raw_test_data[b'data'].shape[0], batch_size):
            self.x_test[i:i+batch_size] = FeatureVectorGenerator.generate_feature_vector(raw_test_data[b'data'][i:i+batch_size])

        self.x_train_aug = np.zeros((raw_data_aug[b'data'].shape[0], 512)).astype(np.float32)

        # feature vector for augmented data
        for i in range(0, raw_data_aug[b'data'].shape[0], batch_size):
            self.x_train_aug[i:i+batch_size] = FeatureVectorGenerator.generate_feature_vector(raw_data_aug[b'data'][i:i+batch_size])
        
        # one hot encoding the data to get y matrix
        self.y_train = FeatureVectorGenerator.one_hot_encoding(raw_data[b'labels'])
        self.y_train_aug = FeatureVectorGenerator.one_hot_encoding(raw_data_aug[b'labels'])

        self.y_test = FeatureVectorGenerator.one_hot_encoding(raw_test_data[b'labels'])

        # raw data without onehot encoding
        self.y_train_raw = raw_data[b'labels']
        self.y_train_raw_aug = raw_data_aug[b'labels']

        self.y_test_raw = raw_test_data[b'labels']

        # normalizing the train and test data
        minimum = np.min(self.x_train)
        scaler = np.max(self.x_train) - np.min(self.x_train)
        self.x_train = (self.x_train - minimum) / scaler
        self.x_test = (self.x_test - minimum) / scaler

        # normalizing the augmented train and test data
        minimum = np.min(self.x_train_aug)
        scaler = np.max(self.x_train_aug) - np.min(self.x_train_aug)
        self.x_train_aug = (self.x_train_aug - minimum) / scaler
        self.x_test_aug = (self.x_test - minimum) / scaler # scaled using a scaler for augmented train data
