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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
	# tokens=wordpunct_tokenize(str(temp))
	# tokens = [w for w in tokens if not w in stop_words]
    #sent = ' '.join(tokens)
	X_train.append(temp)


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
    #sent = ' '.join(tokens)
	X_test.append(temp)


x_train =  X_train
y_train = Y_train
x_test = X_test
y_test = Y_test

print("data is getting created")
data = x_train + x_test

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), 
tags=str(i)) for i, _d in enumerate(data)]

print("tagged data created, model initiated")

model = Doc2Vec(size=50,
                alpha=0.025, 
                min_alpha=0.025,
                min_count=1)

print("model vocab") 
model.build_vocab(tagged_data)

print("epoch now will start")
for epoch in range(10):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
