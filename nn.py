import numpy as np
import random
import matplotlib.pyplot as plt

class neural_net:
	def __init__(self, num_layer, input_size, output_size, list_of_layer_size):
		self.num_layer = num_layer
		self.input_size = input_size
		self.list_of_layer_size = list_of_layer_size
		self.output_size = output_size


def creat_data(num_of_point):
	data_1 = np.random.rand(num_of_point/2,2)
	data_1 =  1*data_1 + 5 
	label_1 = np.ones((num_of_point/2,1))
	# plt.scatter(data_1[:,0], data_1[:,1],color='red')

	data_2 = np.random.rand(num_of_point/2,2)
	data_2 = -1*data_2 - 5
	label_2 = np.zeros((num_of_point/2,1))
	# plt.scatter(data_2[:,0], data_2[:,1],color='blue')

	# plt.show()
	data = np.concatenate((data_1, data_2), axis=0)
	labels = np.concatenate((label_1, label_2), axis=0)

	return data, labels


if __name__ == '__main__':

	number_of_data = 100
	data, labels = creat_data(number_of_data)
	data_temp = np.concatenate((data,labels), axis =1)
	np.random.shuffle(data_temp)

	data = data_temp[:,:2]
	labels = data_temp[:,2:]

	train_data = data[:(0.8*number_of_data),:]
	train_label = labels[:(0.8*number_of_data),:]

	test_data =  data[(0.8*number_of_data):,:]
	test_label = labels[(0.8*number_of_data):,:]

	net = neural_net(2,2,2,[3,3]) # this is where you set the parameters for the neural network
	
	_iters = 10

	weights =[]
	grads = []
	for i_ in xrange(net.num_layer + 1):
		if i_ == 0:
			weights.append(np.random.rand(net.input_size,net.list_of_layer_size[i_]))
		if i_>0 and i_<net.num_layer:
			weights.append(np.random.rand(net.list_of_layer_size[i_-1], net.list_of_layer_size[i_]))
		if i_ == net.num_layer:
			weights.append(np.random.rand(net.list_of_layer_size[i_-1], net.output_size))		
	

	for i_ in xrange(_iters):
		hiddens = []
		for j_ in xrange(net.num_layer):
			if j_== 0:	
				hiddens.append(np.dot(train_data,weights[j_]))
				hiddens[-1] = np.maximum(0,hiddens[-1]) # Applying ReLU here ... :)
			if j_>0 and j_<net.num_layer:
				hiddens.append(np.dot(hiddens[-1],weights[j_]))
				hiddens[-1] = np.maximum(0,hiddens[-1]) # Applying ReLU here ... :)

		output = np.dot(hiddens[-1],weights[net.num_layer])
		

		#########################################################
		max_ = np.max(output,axis=1)
		max_ = max_.reshape(train_data.shape[0],1)
		output = output - max_

		#########################################################
		prob = np.exp(output)/(np.sum(np.exp(output),axis=1)).reshape(train_data.shape[0],1)
		print prob.shape

		loss = np.sum(-np.log(prob[range(train_data.shape[0]),train_label.reshape(1,train_label.shape[0]).astype(int)]))
		print "loss value is after iter --> ", i_," loss --> ",loss

		## back prop will start from here.. ##
		d_output = prob
		d_output[range(train_data.shape[0]),train_label.reshape(1,train_label.shape[0]).astype(int)]

		for i_ in xrange(net.num_layer + 1):

			print 





		








