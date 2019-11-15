# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:19:57 2019

@author: Finn
"""


import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


tyson = pd.read_excel('BillNye_user_tweets.xlsx')  # Read in xlsx file
nye = pd.read_excel('neiltyson_user_tweets.xlsx')  # Read in xlsx file
tweetscombined = tyson.append(nye)  # Combine both classes into one file
rawdata = shuffle(tweetscombined)   # Randomize the order
dataset = rawdata.values  # Convert to arrays
X = dataset[:,1:2]  # Get only the tweet text
X_train, X_test = train_test_split(X, test_size=0.5)    # Split


vectorizer = CountVectorizer(min_df=0, lowercase=False) # Use vectorizer to build conversion dict
corpusdict = vectorizer.fit_transform(rawdata['Text']) # Apply to tweet column only
#print(vectorizer.vocabulary_)
corpusdict.toarray()





'''
X = dataset[:,0:10] # Split data into features
                    # [rows, columns]
Y = dataset[:,10]   # Split data into labels
from sklearn.model_selection import train_test_split  # Organises data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)'''