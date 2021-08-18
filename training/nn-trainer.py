import sys

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# Reading Data
df = pd.read_csv('baseline.csv')
df.dropna(inplace=True)
data = df.values

# Select first five columns as input variables
X = data[:, 3:8]
# Select last column as target variable i.e. option price which in this case is the Black-Scholes option price
Y = data[:, 8]

X = np.asarray(X).astype(np.float32)
Y = np.asarray(Y).astype(np.float32)

# Scaling dataset
std_scalar = preprocessing.StandardScaler()
X_scale = std_scalar.fit_transform(X)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_scale = min_max_scaler.fit_transform(X)

# Split main dataset into train and test+validation
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)

# Split test+validation dataset with 50% split
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential()

model.add(Dense(32, input_shape=(5,)))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

# model.add(Dense(16))
# model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('linear'))

# opt = SGD(lr=0.001)
model.compile(optimizer='adam', loss="mse", metrics=[])

print(model.summary())

history = model.fit(X_train, Y_train, batch_size=16, epochs=100, validation_data=(X_val, Y_val))
          # callbacks=[EarlyStopping(restore_best_weights=True)])

scores = model.evaluate(X_test, Y_test)
# print(scores)

model.save('nn_artifacts/eubopa_model.h5')
print("Model Successfully saved to disk...")