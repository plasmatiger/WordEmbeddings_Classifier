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

import gensim

stop_words = set(stopwords.words('english')+list(punctuation))
path1 = 'aclImdb/train/'
path2 = path1+ 'training/'

path3 = 'aclImdb/test/'
path4 = path3 + 'testing/'

print("loading w2v")
model = gensim.models.Word2Vec.load("w2v.bin")

print(model)
print(list(model.wv.vocab)[1], list(model.wv.vocab)[2])
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
	doc = [word for word in tokens if word in model.wv.vocab]
	doc_mean = np.mean(model.wv[doc], axis=0)
	X_train.append(doc_mean)


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
	doc = [word for word in tokens if word in model.wv.vocab]
	doc_mean = np.mean(model.wv[doc], axis=0)
	X_test.append(doc_mean)


x_train =  X_train
y_train = Y_train
x_test = X_test
y_test = Y_test





# vect = fe.text.CountVectorizer(max_features = 2000)
# X_train_dtm = vect.fit_transform(x_train)
# pd.DataFrame(X_train_dtm.toarray(), columns=vect.get_feature_names())
# X_test_dtm = vect.transform(x_test)
# pd.DataFrame(X_test_dtm.toarray(), columns=vect.get_feature_names())





# creating and training logistic regression model
print("training begins")
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("test begins")
y_predicted = logreg.predict(X_test)

print(ac(y_test, y_predicted))