import cPickle
import numpy as np


## Here I am creating the Cifar-10 dataset. 

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

X_train = np.concatenate((X1,X2,X3,X4,X5),axis=0) # Creating X_train array (50000*3072)
Y_train = np.concatenate((Y1,Y2,Y3,Y4,Y5),axis=0) # Creating Y_train array (50000*1)

Y_test = np.asarray(Y_test)

## Initialsing Parammeters 
# '''''''''''''''''''''''''''''''
W1 = 0.01*np.random.rand(3072,1000)
W2 = 0.01*np.random.rand(1000,10)
v1=0
v2=0
n_iter = 10 # number of iterationx
num_examples = 50000
l_rate = 1e-7
reg = 1e-2

#''''''''''''''''''''''''''''''''
for iter_ in range(n_iter):
	hidden = np.dot(X_train,W1)
	hidden = np.maximum(hidden,0)
	out_ = np.dot(hidden,W2)

	## To avoid overflow
	for i in range(50000):
		a = np.amax(out_[i,:])
		out_[i,:] = out_[i,:] -a 
	
	# Softmax layer
	out_ = np.exp(out_)
	b = np.sum(out_,axis=1,keepdims=True)
	probs = (out_)/b
	

	probs = probs + 1e-15 
	corect_logprobs = -np.log(probs[range(num_examples),Y_train])
	data_loss = np.sum(corect_logprobs)/num_examples
	reg_loss2 = 0.5*reg*np.sum(np.transpose(W2)*np.transpose(W2))
	reg_loss1 = 0.5*reg*np.sum(np.transpose(W1)*np.transpose(W1))
	loss = data_loss + reg_loss1 + reg_loss2
	print "iteration %d: loss %f" % (iter_, loss)
	
	# Storing derivatives for backpropagation
	dscores = probs
	dscores[range(num_examples),Y_train] -= 1
	dscores /= num_examples
	# Computing delta i.e. dW/d(param)
	# Using Momentum + SGD 
	dhidden = np.dot(dscores, W2.T)
	dhidden[hidden <= 0] = 0

	delta = np.dot(np.transpose(hidden),dscores)
	dW = np.dot(X_train.T, dhidden)
	delta = delta + reg*W2
	dW = dW + reg*W1
	#v2 = 0.9*(v2) - (l_rate)*delta
	#W2 = W2 +v2
	W2 = W2 - l_rate*(delta)
	#v1 = 0.9*(v1) - (l_rate)*dW
	#W1 = W1 +v1
	W1 = W1 - l_rate*(dW)

## Computing Accuracy
'''
hidden = np.dot(X_t,W1)
hidden = np.maximum(hidden,0)
scores = np.dot(hidden, W2) 
predicted_class = np.argmax(scores, axis=1)
print 'The fucking Training accuracy: %.2f' % (np.mean(predicted_class == Y_train))
'''
hidden = np.dot(X_test,W1)
hidden = np.maximum(hidden,0)
scores = np.dot(hidden, W2)  
predicted_class = np.argmax(scores, axis=1)
print 'Test accuracy: %.2f' % (np.mean(predicted_class == Y_test))

	


	


