import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load Data from MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Filter out 0s and 1s from the MNIST datasaet
train_filter = np.where((y_train == 0 ) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))

#Assign subarray of 0s and 1s into training sets
X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

#Reshape data into a simpler shape
X_train = X_train.reshape(12665, 784)
X_test = X_test.reshape(2115, 784)
y_train = y_train.reshape(12665, 1)
y_test = y_test.reshape(2115, 1)

# Pre-processing/normalizing of X_train & X_test between (0, 1)
X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

#sequential model creation using dense layers 
model = Sequential(
    [
        Dense(25, activation='relu', name="layer1"),
        Dense(15, activation='relu', name="layer2"),
        Dense(1, activation='sigmoid', name="layer3"),
    ], name = "recog_model",
)


#Fit model using BinaryCrossentropy loss function
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
)

#fit model using taining set and determine epoch
model.fit(
    X_train,y_train,
    epochs=20
)

def predict(array):
    prediction = model.predict(array.reshape(1,784))
    
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    return yhat