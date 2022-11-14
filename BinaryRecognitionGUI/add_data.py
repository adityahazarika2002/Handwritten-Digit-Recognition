import numpy as np
from pathlib import Path

cwd = Path.cwd()
file_X = Path.joinpath(cwd, "dataset", "X_train.npy")
file_y = Path.joinpath(cwd, "dataset", "y_train.npy")

# Load dataset from npy files
X_train = np.load(file_X)
y_train = np.load(file_y)

# Add user input to numpy array
def add_input(img_array, label):
    X_train = np.load(file_X)
    y_train = np.load(file_y)
    X_train = np.concatenate((X_train, img_array), axis=0)
    y_train = np.concatenate((y_train, [[label]]), axis=0)
    np.save(file_X, X_train)
    np.save(file_y, y_train)

