import pickle
with open('train_pos.pkl','rb') as f:
	x_train = pickle.load(f)

print(x_train.shape)

with open('test_pos.pkl','rb') as f:
	x_test = pickle.load(f)

print(x_test.shape)