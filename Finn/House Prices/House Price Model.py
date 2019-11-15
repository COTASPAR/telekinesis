# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:44:08 2019

@author: Finn
"""

# Data processing
import pandas as pd
from sklearn import preprocessing  # Used for normalisation

from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers  # Helps the model generalise better

df = pd.read_csv('housepricedata.csv')  # Read in CSV file
dataset = df.values  # Convert data to arrays
X = dataset[:,0:10] # Split data into features
                    # [rows, columns]
Y = dataset[:,10]   # Split data into labels
min_max_scaler = preprocessing.MinMaxScaler()   # Get min max from sklearn
X_scale = min_max_scaler.fit_transform(X)   # Apply min max to features
from sklearn.model_selection import train_test_split  # Organises data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
# ^ split data into training and testing; split testing into testing and validation

# Modelling

# model -> model.compile -> model.fit -> model.evaluate

from tensorflow.keras.models import Sequential  # Defines the model structure
from tensorflow.keras.layers import Dense  # 'Dense' - fully connected network
model = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),   # Probability of 0.3 of the neurons dropping out. They aren't trained 
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])  # Num of neurons, activation function, (num of inputs)

# Weight decay:
# Include squared values of parameters and weigh them by 0.01
# Prevents weights growing too large. Applied only to high value parameters

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) # Prints out statistics

hist = model.fit(X_train, Y_train,
          batch_size=64, epochs=100,
          validation_data=(X_val, Y_val))   # Batch size is size of minibatch 

model.evaluate(X_test, Y_test)

# Visualizing data
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])  # What to plot
plt.plot(hist.history['val_loss'])  # What to plot
plt.title('Model loss') # Graph title
plt.ylabel('Loss')  # Axis title
plt.xlabel('Epoch') # Axis title
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()




