
# coding: utf-8

# In[33]:


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

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD, Adam
from keras.layers import Dense
from keras.utils import np_utils


# In[25]:


print("Loading Glove Model")
f = open("glove.6B/glove.6B.50d.txt",'r')
model = {}
for line in f:
	splitLine = line.split()
	word = splitLine[0]
	embedding = np.array([float(val) for val in splitLine[1:]])
	model[word] = embedding
print("Done.",len(model)," words loaded!")


# In[26]:


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
	temp = open(path2 + x, 'r')
	temp = temp.read()
	tokens=wordpunct_tokenize(str(temp))
	tokens = [w for w in tokens if not w in stop_words]
	doc = [word for word in tokens if word in model.keys()]
	doc_em = [model[d] for d in doc]
	# print(np.array(doc_em))
	if not doc_em:
		continue

	doc_mean = np.mean(doc_em, axis=0)
	# print(doc_mean.shape)
	# doc_mean = np.reshape(doc_mean, (1,50))
	X_train.append(doc_mean)
	Y_train.append(y)


# In[27]:


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
	temp = open(path4 + x, 'r')
	temp = temp.read()
	tokens=wordpunct_tokenize(str(temp))
	tokens = [w for w in tokens if not w in stop_words]
	doc = [word for word in tokens if word in model.keys()]
	doc_em = [model[d] for d in doc]
	# print(doc_em)
	# print(doc_em.shape)
	if not doc_em:
		continue
	doc_mean = np.mean(doc_em, axis=0)
	# print(doc_mean.shape)
	# doc_mean = np.reshape(doc_mean, (1,50))
	# print(doc_mean.shape)
	X_test.append(doc_mean)
	Y_test.append(y)


# In[28]:


x_train =  np.array(X_train)
print(x_train.shape)
y_train = np.array(Y_train)
x_test = np.array(X_test)
y_test = np.array(Y_test)


# In[29]:


# creating and training logistic regression model
print("keras model prep")
model = Sequential()
model.add(Dense(25, input_dim = 50,
	activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# In[38]:


print("training begins")
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
model.compile(loss="binary_crossentropy", optimizer=adam,
	metrics=["accuracy"])
model.fit(x_train, y_train, nb_epoch=30, batch_size=128,
	verbose=1)


# In[39]:


print("test begins")
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(x_test, y_test,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

