from sklearn import feature_extraction as fe
from sklearn.metrics import accuracy_score as ac
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import os.path
import pickle

import gensim

stop_words = set(stopwords.words('english')+list(punctuation))
path1 = 'aclImdb/train/'
path2 = path1+ 'training/'

path3 = 'aclImdb/test/'
path4 = path3 + 'testing/'


print("train file input and preprocess")

with open(path1 +'temp.txt') as f:
	lines = f.readlines()

X_train = []
Y_train = []
count = 0
for l in lines:
	# if count > 100:
	# 	break
	# count += 1
	x, y = l.split(' ')
	Y_train.append(y)
	temp = open(path2 + x, 'r')
	temp = temp.read()
	tokens=wordpunct_tokenize(str(temp))
	tokens = [w for w in tokens if not w in stop_words]
	X_train.append(tokens)


print("train file input and preprocess")

with open(path3 +'temp.txt') as f:
	lines = f.readlines()

X_test = []
Y_test = []

count = 0
for l in lines:
	# if count > 100:
	# 	break
	# count += 1
	x, y = l.split(' ')
	Y_test.append(y)
	temp = open(path4 + x, 'r')
	temp = temp.read()
	tokens=wordpunct_tokenize(str(temp))
	tokens = [w for w in tokens if not w in stop_words]
	X_test.append(tokens)


x_train =  X_train
y_train = Y_train
x_test = X_test
y_test = Y_test

data = x_train + x_test
model = gensim.models.Word2Vec(data, size  = 100, window =5, min_count =5, workers =4)
model.save("w2v.bin")



# creating and training logistic regression model
# print("training begins")
# BNBC = BernoulliNB()
# BNBC.fit(X_train_dtm, y_train)
# print("test begins")
# y_predicted = BNBC.predict(X_test_dtm)

# print(ac(y_test, y_predicted))