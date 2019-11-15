# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:19:57 2019
Word-level tweet classifier
@author: Finn
"""

# Data processing
import pandas as pd   # Data operations
tyson = pd.read_excel('BillNye_user_tweets.xlsx')  # Read in xlsx file
tyson['Class'] = 0  # Tyson tweets will be class 0
nye = pd.read_excel('neiltyson_user_tweets.xlsx')  # Read in xlsx file
nye['Class'] = 1  # Nye tweets will be class 1
tweetscombined = tyson.append(nye)  # Combine both classes into one file
from sklearn.utils import shuffle   # So we can randomize the data order
rawdata = shuffle(tweetscombined)   # Randomize the order
tweets = rawdata['Text']    # Just get tweet text
classes = rawdata['Class']  # Just get class
twitterdata = pd.DataFrame(data = tweets)   # Create new dataframe ready to hold all (currently just tweets)
twitterdata['Class'] = classes   # Dataframe holds classes
from sklearn.feature_extraction.text import CountVectorizer   # Get vectorizer methods
vectorizer = CountVectorizer(min_df=0, lowercase=False) # Define vectorizer
vectorizer.fit(twitterdata['Text'].values)  # Apply vectorizer to text, give each unique word an ID
tweetslookupdict = vectorizer.vocabulary_   # Store lookup dict
textonehotvector = vectorizer.transform(tweets).toarray()   # Convert tweets to one hot vector
from sklearn.model_selection import train_test_split   # Split data
X_train, X_test, Y_train, Y_test = train_test_split(textonehotvector, twitterdata['Class'].values, test_size=0.5)    # Split

# Modelling - neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers  # Helps the model generalise better

model = Sequential([Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                    Dropout(0.3),
                    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                    Dropout(0.3),
                    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                    Dropout(0.3),
                    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))])

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
                     epochs=10, 
                     validation_data=(X_test, Y_test),
                     batch_size=100)

model.evaluate(X_test, Y_test)

# Visualizing training
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])  # What to plot
plt.plot(hist.history['val_loss'])  # What to plot
plt.title('Model loss') # Graph title
plt.ylabel('Loss')  # Axis title
plt.xlabel('Epoch') # Axis title
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


