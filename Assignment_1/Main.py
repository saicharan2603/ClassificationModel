import numpy as np
from Utils import GetData
from Tests import PickleTest, FeatureVectorGeneratorTest
from PreProcessor import FeatureVectorGenerator
from Model.mlp_model import FCLayer, ActivationLayer, Network
from Model.activation_error_functions import relu, relu_prime, softmax, softmax_prime, cross_entropy, cross_entropy_prime

# importing the raw data by unpickling the data
raw_data = GetData.get_train_data(1)
raw_labels = GetData.get_labels()

raw_test_data = GetData.get_test_data()

# Testing the input data
PickleTest.test_data(raw_data)
PickleTest.test_labels(raw_labels)

x_train = np.zeros((raw_data[b'data'].shape[0], 512))
x_test = np.zeros((raw_data[b'data'].shape[0], 512))

# declaring the batch size
batch_size = 1024

for i in range(0, raw_data[b'data'].shape[0], batch_size):
    x_train[i:i+batch_size] = FeatureVectorGenerator.generate_feature_vector(raw_data[b'data'][i:i+batch_size])
    x_test[i:i+batch_size] = FeatureVectorGenerator.generate_feature_vector(raw_test_data[b'data'][i:i+batch_size])

# one hot encoding the data to get y matrix
y_train = FeatureVectorGenerator.one_hot_encoding(raw_data[b'labels'])
y_test = FeatureVectorGenerator.one_hot_encoding(raw_test_data[b'labels'])

# Testing the feature vector and one hot encoded labels
FeatureVectorGeneratorTest.test_feature_vector_data(raw_data[b'data'], x_train)
FeatureVectorGeneratorTest.test_feature_vector_data(raw_test_data[b'data'], x_test)
FeatureVectorGeneratorTest.test_one_hot_encoding(raw_data[b'labels'], y_train)
FeatureVectorGeneratorTest.test_one_hot_encoding(raw_test_data[b'labels'], y_test)

print('Training the Data')
# network
net = Network()
net.add(FCLayer(512, 64))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(64, 64))
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(64, 10))
net.add(ActivationLayer(softmax, softmax_prime))

# train
net.use(cross_entropy, cross_entropy_prime)
net.fit(x_train, y_train, epochs=10, learning_rate=0.1)

print('Training is Complete')

print('Predicting the Data')
y_hat = net.predict(x_test)

from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, net.predict(x_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_hat))