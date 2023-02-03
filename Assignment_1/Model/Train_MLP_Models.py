from Model.MLP_Model import NeuralNeworks
from Utils.GetData import Data
import pickle
import numpy as np

# defining the file paths
mlp_filepath = 'TrainedModels/mlp_model'
mlp_aug_filepath = 'TrainedModels/mlp_model_aug'

# function to train the given model with given data
def train_and_save_model(model, x_train, y_train, save_path, epochs, lr, patience,):
    err = model.fit(x_train, y_train, epochs, lr, patience)
    pickle.dump(model, open(save_path, 'wb'))
    return err


def train_mlp_models(Data :Data, epochs, lr, patience, retrain = False):
    if retrain:
        # to give the ability to retrain by opening the previous saved data
        mlp_model = pickle.load(open(mlp_filepath, 'rb'))
        mlp_model_aug = pickle.load(open(mlp_aug_filepath, 'rb'))
        err = pickle.load(open('TrainedModels/etc/error_vs_epoch', 'rb'))
        err_aug = pickle.load(open('TrainedModels/etc/error_vs_epoch_aug', 'rb'))
    else:    
        # create a new model to start training
        mlp_model = NeuralNeworks(512, 64, 64, 10)
        mlp_model_aug = NeuralNeworks(512, 64, 64, 10)
        err = np.ndarray(())
        err_aug = np.ndarray(())

    # train the models
    err = np.append(err, train_and_save_model(mlp_model, Data.x_train, Data.y_train, mlp_filepath , epochs, lr, patience)) 
    err_aug = np.append(err_aug, train_and_save_model(mlp_model_aug, Data.x_train_aug, Data.y_train_aug, mlp_filepath , epochs, lr, patience)) 
    
    # write the errors back 
    # dumping the error data
    pickle.dump(err, open('TrainedModels/etc/error_vs_epoch', 'wb'))
    pickle.dump(err_aug, open('TrainedModels/etc/error_vs_epoch_aug', 'wb'))


if __name__ == '__main__':
    data = Data()
    train_mlp_models(data, 25, 0.01, 5)