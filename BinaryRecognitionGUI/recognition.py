import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense

cwd = Path.cwd()
file_X = Path.joinpath(cwd, "dataset", "X_train.npy")
file_y = Path.joinpath(cwd, "dataset", "y_train.npy")

# Load Data from npy files
X_train = np.load(file_X)
y_train = np.load(file_y)

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