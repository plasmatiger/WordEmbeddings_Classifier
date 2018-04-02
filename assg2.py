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
# nltk.download()
import os.path
import pickle

path1 = 'aclImdb/train/'
path2 = path1+ 'training/'

path3 = 'aclImdb/test/'
path4 = path3 + 'testing/'

# inp = open(path1 + 'temp.txt', 'r')
# inpu = inp.read()


def preprocess(document):
	stop_words = set(stopwords.words('english')+list(punctuation))
	stop_word_removed = [i.lower() for i in wordpunct_tokenize(document) if i.lower() not in stop_words]
	tagged_words_nltk = nltk.pos_tag(stop_word_removed)
	tagged_words_wordnet = []
	for i in tagged_words_nltk:
		if i[1].startswith('J'):
			tagged_words_wordnet.append((i[0],wordnet.ADJ))
		if i[1].startswith('R'):
			tagged_words_wordnet.append((i[0],wordnet.ADV))
		if i[1].startswith('V'):
			tagged_words_wordnet.append((i[0],wordnet.VERB))
		else:
			tagged_words_wordnet.append((i[0],wordnet.NOUN))
	
	processed_document = [WordNetLemmatizer.lemmatize(i[0],i[1]) for i in tagged_words_wordnet]
	return processed_document	

print("train file input and preprocess")

with open(path1 +'temp.txt') as f:
	lines = f.readlines()

X_train = []
Y_train = []
count = 0
for l in lines:
	count += 1
	if count%100 == 0:
		print(count/250, "train done")
	#print(l)
	x, y = l.split(' ')
	# print(x)
	# print(y)
	Y_train.append(y)
	temp = open(path2 + x, 'r')
	temp = temp.read()
	temp = preprocess(temp)
	X_train.append((temp))

# print(len(X_train))
# print(len(Y_train))

# print(X_train[-1])

# inp = open(path3 + 'temp.txt', 'r')
# inpu = inp.read()

print("train file input and preprocess")

with open(path3 +'temp.txt') as f:
	lines = f.readlines()

X_test = []
Y_test = []
count = 0
for l in lines:
	count += 1
	if count%100 == 0:
		print(count/250, "test done")
	#print(l)
	x, y = l.split(' ')
	# print(x)
	# print(y)
	Y_test.append(y)
	temp = open(path4 + x, 'r')
	temp = temp.read()
	temp = preprocess(temp)
	X_test.append((temp)

with open('train_pos.pkl','wb+') as f:
	pickle.dump(X_train,f)

with open('test_pos.pkl','wb+') as f:
	pickle.dump(X_test,f)

x_train = X_train[:1000]+X_train[-1000:-1]
y_train = Y_train[:1000]+Y_train[-1000:-1]

x_test = X_train[5000:5500]+X_train[-5500:-5000]
y_test = Y_train[5000:5500]+Y_train[-5500:-5000]

# x_train =  X_train
# y_train = Y_train
# x_test = X_test
# y_test = Y_test

print("bow initiated")
vect = fe.text.CountVectorizer(max_features = 2000)
X_train_dtm = vect.fit(x_train)
X_train_dtm = vect.transform(X_train_dtm)
pd.DataFrame(X_train_dtm.toarray(), columns=vect.get_feature_names())
X_test_dtm = vect.transform(x_test)
pd.DataFrame(X_test_dtm.toarray(), columns=vect.get_feature_names())

print(X_train_dtm.shape)
print(X_test_dtm.shape)



# creating and training logistic regression model
print("training begins")
logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)
print("test begins")
y_predicted = logreg.predict(X_test_dtm)

# print(y_predicted)

print(ac(y_test, y_predicted))