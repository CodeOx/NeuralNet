import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
import string
from sklearn import preprocessing

#################################
# Neural Netwotk ##############

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# input : parameters for a neural network
# returns : empty neural net, i.e., a list of weight matrices
def create_network(num_inputs, hidden_units_list, num_output):
	np.random.seed(0)
	w = [0]*(len(hidden_units_list)+1)
	units_left_layer = num_inputs
	std = float(1/np.sqrt(num_inputs))

	for i in range(len(hidden_units_list)) :
		units_right_layer = hidden_units_list[i]
		
		w[i] = np.random.normal(loc=float(0), scale=std, size=(units_left_layer + 1, units_right_layer))	# + 1 for bias
		
		units_left_layer = units_right_layer
	
	w[len(hidden_units_list)] = np.random.normal(loc=float(0), scale=std, size=(units_left_layer + 1, num_output))	# + 1 for bias
	
	return w

# input : a neural_network, an input (list of vectors) consistent with the network
# returns : list of output vectors for each layer of the network
def forward_multi(neural_net, input):
	m = len(input)
	output_list = []
	
	layer_input = np.hstack([np.ones((m,1)), input]) 		 	# transpose of input vector, 
	
	for w in neural_net:
		layer_output = np.matmul(layer_input, w) 				# transpose of output vector
		layer_output = sigmoid(layer_output)
		layer_input = np.hstack([np.ones((m,1)), layer_output])		# append 1 for bias
		
		output_list.append(layer_output.T)

	return output_list

# returns : gradient for each parameter
def backward_multi(neural_net, input, output, target_output):
	m = len(target_output)
	t = target_output.T
	inp = np.hstack([np.ones((input.shape[0],1)), input]).T

	num_layers = len(neural_net)
	gradient = [0]*num_layers
	gradient_list = []
	delta = [0]*num_layers
	
	layer_output = output[num_layers-1]
	delta[num_layers-1] = layer_output * (1 - layer_output) * (layer_output - t)

	for i in range(num_layers-2, -1, -1):
		layer_output = output[i]
		delta[i] = layer_output * (1 - layer_output) * (np.dot(neural_net[i+1], delta[i+1])[1:,:])

	z = output[0].shape[1]
	for i in range(num_layers-1, 0, -1):
		augmented_layer_output = np.vstack([[1]*z, output[i-1]])
		#gradient[i] = np.matmul(augmented_layer_output, delta[i].T)
		gradient[i] = np.dot(delta[i],augmented_layer_output.T).T

	augmented_layer_output = inp
	gradient[0] = np.dot(delta[0],augmented_layer_output.T).T

	return gradient

def getError(output, y):
	error = 0.0
	y_inp = y.T
	y_pred = output[len(output)-1]

	y_err = y_inp - y_pred
	error_list = y_err * y_err
	error = np.sum(error_list)

	return error


# input : neural network to train, x,y for training
# returns : trained neural network
def train(neural_net, x, y, batch_size, neta):
	max_iterations = 50000	# max iterations
	error_threshold = 0.00001

	error_old = -1.0
	it = 0 					# epochs

	while(it < max_iterations):
		error_new = 0.0
		for s in range(len(y)/batch_size):
			it += 1
			start = s * batch_size
			
			output = forward_multi(neural_net, x[start:start+batch_size])
			net_gradient = backward_multi(neural_net, x[start:start+batch_size], output, y[start:start+batch_size])

			neta_gradient = [neta*g for g in net_gradient]
			neural_net = [n-g for (n,g) in zip(neural_net, neta_gradient)]

			error_new += getError(output, y[start:start+batch_size])

		error_new /= len(y)
		print error_new
		#if (abs(error_old - error_new) < error_threshold and error_old >= 0.0) :
		#	return neural_net
		error_old = error_new	

	return neural_net

def predict1(neural_net,X):
	n = len(neural_net)
	a = X.T; 
	m = np.shape(a)[1]
	a = np.vstack((np.ones((1,m)),a))
	for i in range(0,n):
		if (i == n-1):
			z = np.dot(neural_net[i].T,a)
			a = sigmoid(z)
		else:
			z = np.dot(neural_net[i].T,a)
			a = sigmoid(z)
			a = np.vstack((np.ones((1,np.shape(a)[1])),a))

	e = (np.max(a,axis=0) == a)*1
	e=e.T
	e = np.ndarray.tolist(e)
	ans = list(range(m))
	for i in range(0,m):
		ans[i] = e[i].index(1)
	ans = np.array([ans]).T
	return ans

def getAccuracy(Y_predict, Y_input):
	m = len(Y_input)
	count = 0
	for i in range(m):
		flag = 1
		for j in range(10):
			if Y_predict[i][j] != Y_input[i][j]:
				flag = 0
				break
		count += flag

	return float(count)/m

def accuracy(Y_inp, Y_pred):
	count = 0
	for i in range(len(Y_inp)):
		if Y_inp[i] == Y_pred[i]:
			count += 1
	return float(count)/len(Y_inp)

#################################
# I/O ###########################

train_file = "F:/ml/A3/Q2/data/poker-hand-training-true.data"
test_file = "F:/ml/A3/Q2/data/poker-hand-testing.data"

dfTr = pd.read_csv(train_file, header=None)
dfTe = pd.read_csv(test_file, header=None)

#################################
# Random UnderSampling ##########

# dfTr_cl = [dfTr[dfTr[10] == i] for i in range(10)]
# dfTr_count = [len(k) for k in dfTr_cl]
# dfTr_sample = [dfTr_cl[i].sample(min(2000, dfTr_count[i])) for i in range(10)]
# dfTr = pd.concat(dfTr_sample)
# print dfTr

#################################
# One hot encoding ##############

X_Tr = dfTr.iloc[:,:10]
X_Te = dfTe.iloc[:,:10]
Y_Tr = dfTr.iloc[:,10:]
Y_Te = dfTe.iloc[:,10:]

X_combined = pd.concat([X_Tr,X_Te],keys=[0,1])
X_combined = pd.get_dummies(X_combined, columns=[0,1,2,3,4,5,6,7,8,9])

Y_combined = pd.concat([Y_Tr,Y_Te],keys=[0,1])
Y_combined = pd.get_dummies(Y_combined, columns=[10])

X_Tr, X_Te = X_combined.xs(0), X_combined.xs(1)
Y_Tr, Y_Te = Y_combined.xs(0), Y_combined.xs(1)

X_Tr = X_Tr.values
X_Te = X_Te.values

Y_Tr = Y_Tr.values
Y_Te = Y_Te.values

# Standardizing the data
X_Tr = preprocessing.scale(X_Tr) 

#################################
# running network  ##############

net = create_network(85, [25], 10)
trained_net = train(net, X_Tr, Y_Tr, 100, 0.1)

Y_Tr_predict = predict1(trained_net, X_Tr)
#Y_Te_predict = predict1(trained_net, X_Te)

print "training accuracy = ", accuracy(dfTr.iloc[:,10:].values, Y_Tr_predict)
#print "testing accuracy = ", accuracy(dfTe.iloc[:,10:].values, Y_Te_predict)