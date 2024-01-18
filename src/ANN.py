
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from scipy import stats
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

# Construct neural network model for ML dimension reduction
# training workflow - 
# 1. training and test spliting (with fix random seed!)
# 2. training ANN (with fixed intialization seed! tf.random.set_seed())
# 3. tune learning rate, layer number and epoch time to fit the model. The trained model will be reproducible!
def model_tanh(num_input, num_output, custom_loss_function, l1_reg = 1e-4, 
               l2_reg = 1e-3,learning_rate = 2.5e-3, nlayer=4, tf_rand=0):
    model = keras.Sequential()

    for i in range(nlayer):
        model.add(keras.layers.Dense(num_input, kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                     bias_regularizer=regularizers.l2(l2_reg), 
                                     activity_regularizer=regularizers.l2(l2_reg), activation='selu'))
    
    model.add(keras.layers.Dense(num_output,activation = 'tanh'))
    # initialization seed for ANN weights. Fix in order to reproduce the ANN model
    tf.random.set_seed(tf_rand)
    learning_rate = learning_rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=100,
        decay_rate=0.99)
    
    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
   
    model.compile(optimizer=opt,
                  loss=custom_loss_function,
                  metrics=[custom_loss_function])
    # print(model.summary())
    return model


# ANN-assisted surogate  with input weights
def ML_dimension_reduction_v4(X, Y, prior_min_Y, prior_max_Y, 
                           custom_loss_function, mistfit_weights, num_input = None,
                           l1_reg = 1e-4, l2_reg = 1e-3,learning_rate = 2.5e-3, 
                           num_epoch = 2000, batch_size = 125, rand=0, nlayer=4, tf_rand=0):
    
    # X: number of sample x number of features, theta
    # Y: number of sample x number of features, mistmatch
    num_X = X.shape[1]
    num_Y = Y.shape[1]
    
    # rescale X and Y
    ## remove the mean of X
    X_zero_mean = X-np.mean(X,axis = 0)
    
    # rescale Y into [-1,1]
    Y_scaled = ((Y - prior_min_Y)/(prior_max_Y-prior_min_Y))*2 -1
    
    # split the train and test dataset
    X_train, X_test, y_train, y_test, w_train, w_test= train_test_split(X_zero_mean, 
                                                                        Y_scaled, 
                                                                        mistfit_weights,
                                                                        test_size=0.1, 
                                                                        random_state = rand)
    
    # construct nn model: S()
    if num_input is None: 
        num_input = num_X
    S = model_tanh(num_input, num_Y, custom_loss_function,l1_reg = l1_reg, 
                   l2_reg = l2_reg, learning_rate = learning_rate, nlayer=nlayer, 
                   tf_rand=tf_rand)
    
    # train model: S()

        # use weights from y (mistmatch)
#         distance_to_Y_obs = np.sqrt(np.sum((y_train-Y_obs.T)**2,axis = 1))
    
    history = S.fit(X_train, y_train, epochs=num_epoch, batch_size=batch_size, 
                    validation_data=(X_test, y_test), 
                    verbose = 0, callbacks=[TqdmCallback(verbose=0)], 
                    sample_weight=w_train)
    print(S.summary())
    # ML dimension 
    S_X = S.predict(X_zero_mean)
#     S_X_obs = S.predict(X_obs.reshape(1,num_X)-np.mean(X,axis = 0))
    
    return S, S_X, S.predict(X_train), y_train, S.predict(X_test), y_test, history