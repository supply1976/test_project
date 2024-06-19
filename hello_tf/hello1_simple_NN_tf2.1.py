import os, sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


def data_pool(size=1000):
    """ A data pool to generate training and testing dataset
    use this to test you ML model code correctness and accuracy
    size: number of data
    """
    # x as input data (array) with shape=[size,2], means each data has two features
    x = np.random.rand(size, 2)
    # y as ground true, evaluated by given formula
    y = np.sqrt(-1 * np.log(x[:, 0])) * np.cos(x[:, 1])
    return x, y


def fit_and_build_model():
    # provide training dataset, fit model and save model
    train_X, train_Y = data_pool()
    _, input_dim = train_X.shape
    # define the ML model form
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=64, activation='relu', input_dim=input_dim))
    model.add(keras.layers.Dense(units=1, activation=None))
    print(model.summary())
    # compile the model, define the optimizer and loss function
    model.compile(optimizer='adam', loss='mse')
    # fit the model by feeding training data, use 80% of training data to train.
    model.fit(x=train_X, y=train_Y, verbose=1, epochs=500, batch_size=128, validation_split=0.2)
    # save teh trained model to HDF5 format (.h5) that can be loaded later
    model.save("my_model_1.h5")


def check_model():
    # load the trained model and see the result
    # creating another test data from data pool and do the prediction
    test_X, true_Y = data_pool(size=100)
    my_model = keras.models.load_model("my_model_1.h5")
    pred_Y = my_model.predict(test_X)
    result = pd.DataFrame({'Y_predict': pred_Y.flatten(), 'Y_true': true_Y.flatten()})
    print(result.head())


if __name__ == '__main__':
    fit_and_build_model()
    check_model()
