import pickle
from Utils.GetData import Data

# defining the file paths
svm_filepath = 'TrainedModels/svm_model'
svm_aug_filepath = 'TrainedModels/svm_model_aug'

knn_filepath = 'TrainedModels/knn_model'
knn_aug_filepath = 'TrainedModels/knn_model_aug'

logistic_filepath = 'TrainedModels/logistic_model'
logistic_aug_filepath = 'TrainedModels/logistic_model_aug'

dtc_filepath = 'TrainedModels/dtc_model'
dtc_aug_filepath = 'TrainedModels/dtc_model_aug'

def train_svm_model(x_train, y_train_raw, save_path):
    from sklearn import svm

    # creating the model
    svm_model = svm.SVC()

    # training the model
    svm_model.fit(x_train, y_train_raw)

    # saving the trained model
    pickle.dump(svm_model, open(save_path, 'wb'))

    print("SVM Model Trained")

def train_knn_model(x_train, y_train, save_path):
    from sklearn.neighbors import KNeighborsClassifier

    KNN_model = KNeighborsClassifier()

    KNN_model.fit(x_train, y_train)

    # saving the trained model
    pickle.dump(KNN_model, open(save_path, 'wb'))

    print("KNN Model Trained")


def train_logistic_model(x_train, y_train_raw, save_path, max_iter):
    from sklearn.linear_model import LogisticRegression

    logistic_model = LogisticRegression(max_iter=max_iter)

    # training
    logistic_model.fit(x_train, y_train_raw)

    # saving the trained model
    pickle.dump(logistic_model, open(save_path, 'wb'))

    print("linear regression Model Trained")

def train_dtc_model(x_train, y_train, save_path, max_depth):
    from sklearn.tree import DecisionTreeClassifier

    dtc_model = DecisionTreeClassifier(max_depth= max_depth)

    # training the model
    dtc_model.fit(x_train, y_train)

    # saving the trained model
    pickle.dump(dtc_model, open(save_path, 'wb'))

    print("Descision Tree Model Trained")

def train_all_models(Data:Data):
    train_svm_model(Data.x_train, Data.y_train_raw, svm_filepath)
    train_knn_model(Data.x_train, Data.y_train, knn_filepath)
    train_logistic_model(Data.x_train, Data.y_train_raw, logistic_filepath, max_iter = 1000)
    train_dtc_model(Data.x_train, Data.y_train, dtc_filepath, max_depth = 10)

def train_all_models_aug(Data:Data):
    train_svm_model(Data.x_train_aug, Data.y_train_raw_aug, svm_aug_filepath)
    train_knn_model(Data.x_train_aug, Data.y_train_aug, knn_aug_filepath)
    train_logistic_model(Data.x_train_aug, Data.y_train_raw_aug, logistic_aug_filepath, max_iter = 1000)
    train_dtc_model(Data.x_train_aug, Data.y_train_aug, dtc_aug_filepath, max_depth = 10)

if __name__ == '__main__':
    data = Data()

    train_all_models(data)
    train_all_models_aug(data)