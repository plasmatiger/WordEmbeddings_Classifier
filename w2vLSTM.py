
# coding: utf-8

# In[1]:


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
from keras.layers import Activation, LSTM
from keras.optimizers import SGD, Adam
from keras.layers import Dense
from keras.utils import np_utils


# In[2]:


stop_words = set(stopwords.words('english')+list(punctuation))
path1 = 'aclImdb/train/'
path2 = path1+ 'training/'

path3 = 'aclImdb/test/'
path4 = path3 + 'testing/'

print("loading w2v")
model = gensim.models.Word2Vec.load("w2v.bin")

print(model)
print(list(model.wv.vocab)[1], list(model.wv.vocab)[2])


# In[3]:


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


# In[4]:


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


# In[19]:


x_train =  np.array(X_train)
x_train = x_train.reshape((x_train.shape[0],1, x_train.shape[1]))
print(x_train.shape)
y_train = np.array(Y_train)
y_train = y_train.reshape((y_train.shape[0]))
x_test = np.array(X_test)
x_test = x_test.reshape((x_test.shape[0],1, x_test.shape[1]))
y_test = np.array(Y_test)
y_test = y_test.reshape((y_test.shape[0]))
# In[21]:


# creating and training logistic regression model
print("keras model prep")
model = Sequential()
model.add(LSTM(100, input_shape = (1,100),
	activation="tanh"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# In[61]:


print("training begins")
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss="binary_crossentropy", optimizer=adam,
	metrics=["accuracy"])
model.fit(x_train, y_train, nb_epoch=10, batch_size=128,
	verbose=1)


# In[39]:


print("test begins")
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(x_test, y_test,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
