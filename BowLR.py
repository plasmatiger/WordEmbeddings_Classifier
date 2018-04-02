from sklearn import feature_extraction as fe
from sklearn.metrics import accuracy_score as ac
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import os.path
import pickle

path1 = 'aclImdb/train/'
path2 = path1+ 'training/'

path3 = 'aclImdb/test/'
path4 = path3 + 'testing/'


print("train file input and preprocess")

with open(path1 +'temp.txt') as f:
	lines = f.readlines()

X_train = []
Y_train = []
for l in lines:
	x, y = l.split(' ')
	Y_train.append(y)
	temp = open(path2 + x, 'r')
	temp = temp.read()
	X_train.append(temp)


print("train file input and preprocess")

with open(path3 +'temp.txt') as f:
	lines = f.readlines()

X_test = []
Y_test = []
for l in lines:
	x, y = l.split(' ')
	Y_test.append(y)
	temp = open(path4 + x, 'r')
	temp = temp.read()
	X_test.append(temp)


x_train =  X_train
y_train = Y_train
x_test = X_test
y_test = Y_test

print("bow initiated")

vect = fe.text.CountVectorizer(max_features = 2000)
X_train_dtm = vect.fit_transform(x_train)
pd.DataFrame(X_train_dtm.toarray(), columns=vect.get_feature_names())
X_test_dtm = vect.transform(x_test)
pd.DataFrame(X_test_dtm.toarray(), columns=vect.get_feature_names())





# creating and training logistic regression model
print("training begins")
logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)
print("test begins")
y_predicted = logreg.predict(X_test_dtm)

print(ac(y_test, y_predicted))