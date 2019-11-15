# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:19:57 2019
Word-level tweet classifier
@author: Finn
"""



#from sklearn import preprocessing

# Data processing
import pandas as pd   # Data operations
tyson = pd.read_excel('BillNye_user_tweets.xlsx')  # Read in xlsx file
nye = pd.read_excel('neiltyson_user_tweets.xlsx')  # Read in xlsx file
tweetscombined = tyson.append(nye)  # Combine both classes into one file
from sklearn.utils import shuffle   # So we can randomize the data order
rawdata = shuffle(tweetscombined)   # Randomize the order
tweets = rawdata['Text']    # Just get the tweets
from sklearn.feature_extraction.text import CountVectorizer   # Get vectorizer methods
vectorizer = CountVectorizer(min_df=0, lowercase=False) # Define vectorizer
vectorizer.fit(tweets)  # Apply vectorizer to dataset, give each unique word an ID
lookupdict = vectorizer.vocabulary_   # Store lookup dict
onehotvector = vectorizer.transform(tweets).toarray()   # Convert tweets to one hot vector

# Data organising
from sklearn.model_selection import train_test_split   # Split data
X_train, X_test = train_test_split(onehotvector, test_size=0.5)    # Split
print(X_train)


'''
dataset = rawdata.values  # Convert to arrays
X = dataset[:,1:2]  # Get only the tweet text
vectorizer = CountVectorizer(min_df=0, lowercase=False) # Use vectorizer to build conversion dict
corpusdict = vectorizer.fit_transform(X)

print(corpusdict)
#print(vectorizer.vocabulary_)
#corpusdict.toarray()
#X_train, X_test = train_test_split(X, test_size=0.5)    # Split


X = dataset[:,0:10] # Split data into features
                    # [rows, columns]
Y = dataset[:,10]   # Split data into labels
from sklearn.model_selection import train_test_split  # Organises data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)'''