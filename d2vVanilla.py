
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
from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import os.path
import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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


# In[3]:


print("loading d2v")
model= Doc2Vec.load("d2v.model")


# In[4]:


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
	# tokens=wordpunct_tokenize(str(temp))
	# tokens = [w for w in tokens if not w in stop_words]
	# doc = [word for word in tokens if word in model.wv.vocab]
	# doc_mean = np.mean(model.wv[doc], axis=0)
	temp = word_tokenize(temp.lower())
	v = model.infer_vector(temp)
	X_train.append(v)


# In[5]:



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
	# tokens=wordpunct_tokenize(str(temp))
	# tokens = [w for w in tokens if not w in stop_words]
	# doc = [word for word in tokens if word in model.wv.vocab]
	# doc_mean = np.mean(model.wv[doc], axis=0)
	temp = word_tokenize(temp.lower())
	v = model.infer_vector(temp)
	X_test.append(v)


# In[6]:


x_train =  np.array(X_train)
print(x_train.shape)
y_train = np.array(Y_train)
x_test = np.array(X_test)
y_test = np.array(Y_test)


# In[7]:


print("keras model prep")
model = Sequential()
model.add(Dense(50, input_dim = x_train.shape[1],
	activation="relu"))
model.add(Dense(20,
	activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# In[8]:


print("training begins")
sgd = SGD(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(x_train, y_train, nb_epoch=50, batch_size=128,
	verbose=1)


# In[9]:


print("test begins")
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(x_test, y_test,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

