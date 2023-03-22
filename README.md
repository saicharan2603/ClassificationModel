# Assignment 1

CS776A: Deep Learning for Computer Vision.

Inorder to train the models execute **Main.py**, The python file inturn executes
1. Model/Train_MLP_Models.py
2. Model/Train_ML_Models.py

After, completing training the models, the models were pickled and stored in "**TrainedModels**" folder and the error vs epoch data was stored in "**TrainedModels**". later, these models were used for prediction.

*Note: The MLP models and corresponding error data files were renamed inorder to avoid overwriting the trained models.*

> ## Prerequisite for training the model.
>
>download the data from http://www.cs.toronto.edu/~kriz/cifar.html and extract into *Data* folder.
>
>The *Data* folder must contain the following files without any subfolders
>
>- data_batch_1
>- data_batch_2
>- data_batch_3
>- data_batch_4
>- data_batch_5
>- test_batch
>- batches.meta
>
>The *Data* folder is also available in the onedrive link shared.

Required downloads: (for testing our trained models)
> 
>The ready to use data (train data, augment data, test data after feature vector generation) for the models is pickled in the form of "*data.pkl*".This can be downloaded from one drive link as an alternative to avoid computation required in generating the feature vector
>
>To use the "*data.pkl*" for training or prediction download and place the file inside same folder in which Main.py file exist. 
>
>All of the trained models (including ML model and MLP models) are stored using pickle in the *TrainedModels* folder and is available to download at: https://iitk-my.sharepoint.com/:f:/g/personal/saicharanm22_iitk_ac_in/EjtCfbvsqghBmm15gtVKjf8Bko1qfAZlcq3f7uK0Wopraw?e=EVz5Bg
>


**Results.ipynb** contains all the test results
 

<br>

<br>

## Contents in different folders

The **Model** folder contains all the files
1. MLP Model
2. File to train the model
3. File to train ML Models


> **MLP Model** - contains code for Multilayer perceptron
The class **Neural Networks** consists of 4 main methods
>
> 1. forward pass
       - This method uses forward propogation to get the y_hat
> 2. backprop
        - used backpropogation to update the weights
> 3. fit
        - uses *forward_pass* and *backprop* to train the model
> 4. predict
        - uses *forward_pass* to  predict y_hat and then, the highest probability index is determined to generate y_pred vector(0, 1, 2, ... ,9).

The **Utils Folder** contains

1. *Data_Utils* - to unpickle and import the data from "Data" folder
2. *GetData* - contains class *Data* which generates all the required feature vectors for original data as well as the augmented data set 
3. *Info* - prints the information of the pickled dictionary

The **PreProcessor** folder contains the *ImagePreProcessor.py*, *Augment.py* and *FeatureVectorGenerator.py* for image processing, data augmentation and feature vector generation after upsampling respectively.


