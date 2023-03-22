from Utils.GetData import Data
from Model import Train_ML_Models as ML
from Model import Train_MLP_Models as MLP

# importing data
data = Data()

# train the MLP models with these hyper parameters
# trains on both original and augmented data
MLP.train_mlp_models(data, epochs=100, lr=0.01, patience=5)

# train the ML Models using the 
ML.train_all_models(data)

# train using the augmented data
ML.train_all_models_aug(data)