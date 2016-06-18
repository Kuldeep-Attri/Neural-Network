import cPickle

file1 = "/users/kuldeepsharma/Desktop/cifar-10-batches-py/data_batch_1"
fo = open(file1, 'rb')
dict1 = cPickle.load(fo)
fo.close()
X1 = dict1['data']
Y1 = dict1['labels']

file1 = "/users/kuldeepsharma/Desktop/cifar-10-batches-py/data_batch_2"
fo = open(file1, 'rb')
dict1 = cPickle.load(fo)
fo.close()
X2 = dict1['data']
Y2 = dict1['labels']

file1 = "/users/kuldeepsharma/Desktop/cifar-10-batches-py/data_batch_3"
fo = open(file1, 'rb')
dict1 = cPickle.load(fo)
fo.close()
X3 = dict1['data']
Y3 = dict1['labels']

file1 = "/users/kuldeepsharma/Desktop/cifar-10-batches-py/data_batch_4"
fo = open(file1, 'rb')
dict1 = cPickle.load(fo)
fo.close()
X4 = dict1['data']
Y4 = dict1['labels']

file1 = "/users/kuldeepsharma/Desktop/cifar-10-batches-py/data_batch_5"
fo = open(file1, 'rb')
dict1 = cPickle.load(fo)
fo.close()
X5 = dict1['data']
Y5 = dict1['labels']

file1 = "/users/kuldeepsharma/Desktop/cifar-10-batches-py/test_batch"
fo = open(file1, 'rb')
dict1 = cPickle.load(fo)
fo.close()
X_test = dict1['data']
Y_test = dict1['labels']

import numpy as np

X_train = np.concatenate((X1,X2,X3,X4,X5),axis=0)
Y_train = np.concatenate((Y1,Y2,Y3,Y4,Y5),axis=0)

Y_test = np.asarray(Y_test) 
W = np.random.rand(10,3072)

out_ = np.dot(W,np.transpose(X_train))

l_rate = 1e-14

# Define loss function and use back propagation..
iter_ = 200
for j in range(iter_):
	loss =0
	v= 0 
	out_ = np.dot(W,np.transpose(X_train))
	count = X_train.shape[0]
	for i in range(count):
		A = out_[:,i]
		A = A.reshape(10,1)
		B = X_train[i,:]
		B = B.reshape(1,3072)
		
		delta = np.dot(A,B)
		delta[Y_train[i],:]=0
		#W = W - l_rate*delta
		v = 0.95*(v) - (l_rate)*delta
		W = W +v
		A[Y_train[i]]=0
		A = np.square(A)
		loss += np.sum(A) 
	print j
	j = j+1
	print (loss)

a = W[0,:]
im = a.reshape(32,32,3)
'''
from PIL import Image
max_ = np.amax(im)
min_ = np.amin(im)

diff_ = max_-min_
print diff_
temp_ = np.ones(3072)
temp_ = temp_.reshape(32,32,3)
temp_ = min_*temp_ 
im = np.subtract(im,temp_)
im = im/diff_
print im
im = im*255
im  = im.astype(int)
print im

im = Image.fromarray(im,'RGB')
im.save('my.png')
 
#out_test = np.dot(W,np.transpose(X_test))
#for i in range(X.tesy.shape[0]):

out_test = np.dot(W,np.transpose(X_test))
labels = np.zeros(10000)
for i in range(x_test.shape[0]):
'''

	


