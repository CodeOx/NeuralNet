import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
import string
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import time

#################################
# Neural Netwotk ##############

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	z = np.zeros_like(x)
	return np.maximum(z, x)

def activation(x, type='sigmoid'):
	if type == 'relu':
		return relu(x)
	return sigmoid(x)

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
def forward_multi(neural_net, input, actv='sigmoid'):
	m = len(input)
	output_list = []
	
	layer_input = np.hstack([np.ones((m,1)), input]) 		 	# transpose of input vector, 
	
	for i in range(len(neural_net)-1):
		w = neural_net[i]
		layer_output = np.matmul(layer_input, w) 				# transpose of output vector
		layer_output = activation(layer_output,actv)
		layer_input = np.hstack([np.ones((m,1)), layer_output])		# append 1 for bias
		
		output_list.append(layer_output.T)

	w = neural_net[len(neural_net)-1]
	layer_output = np.matmul(layer_input, w) 				# transpose of output vector
	layer_output = activation(layer_output,'sigmoid')
	output_list.append(layer_output.T)

	return output_list

# returns : gradient for each parameter
def backward_multi(neural_net, input, output, target_output, actv='sigmoid'):
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
		if actv == 'sigmoid': grad = layer_output * (1 - layer_output)
		if actv == 'relu': grad = (layer_output > 0) * 1
		delta[i] = grad * (np.dot(neural_net[i+1], delta[i+1])[1:,:])

	z = output[0].shape[1]
	for i in range(num_layers-1, 0, -1):
		augmented_layer_output = np.vstack([[1]*z, output[i-1]])
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
def train(neural_net, x, y, batch_size, neta, learning_type='fixed', actv='sigmoid'):
	max_iterations = 50000	# max iterations = 200 epochs
	tol = 0.0001

	error_old = -1.0
	it = 0 					# epochs

	while(it < max_iterations):
		error_new = 0.0
		for s in range(len(y)/batch_size):
			it += 1
			start = s * batch_size
			
			output = forward_multi(neural_net, x[start:start+batch_size], actv)
			net_gradient = backward_multi(neural_net, x[start:start+batch_size], output, y[start:start+batch_size], actv)

			neta_gradient = [neta*g for g in net_gradient]
			neural_net = [n-g for (n,g) in zip(neural_net, neta_gradient)]

			error_new += getError(output, y[start:start+batch_size])

		error_new /= len(y)
		# print error_new
		if ((error_old - error_new) < tol and error_old >= 0.0 and learning_type == 'variable') :
			neta /= 5.0

		error_old = error_new	

	return neural_net

def predict1(neural_net,X,actv='sigmoid'):
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
			a = activation(z, actv)
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

# train_file = "F:/ml/A3/Q2/data/poker-hand-training-true-processed.data"
# test_file = "F:/ml/A3/Q2/data/poker-hand-testing-processed-small.data"

train_file = sys.argv[2]
test_file = sys.argv[3]

dfTr = pd.read_csv(train_file, header=None)
dfTe = pd.read_csv(test_file, header=None)

X_Tr = dfTr.iloc[:,:85].values
X_Te = dfTe.iloc[:,:85].values
Y_Tr = dfTr.iloc[:,85:].values
Y_Te = dfTe.iloc[:,85:].values

# Standardizing the data
X_Tr = preprocessing.scale(X_Tr)
X_Te = preprocessing.scale(X_Te)

#################################
# running network  ##############

part = 6

# part (c)
if part == 3:
	f = open('logs_part_c.csv', 'w+')

	hl_units = [5, 10, 15, 20, 25]
	learning_rate = 0.1

	for h in hl_units:
		start = time.time()

		net = create_network(85, [h], 10)
		trained_net = train(net, X_Tr, Y_Tr, 128, 0.1, 'fixed', 'sigmoid')

		end =  time.time()

		Y_Tr_predict = predict1(trained_net, X_Tr, 'sigmoid')
		Y_Te_predict = predict1(trained_net, X_Te, 'sigmoid')

		Y_Tr1 = [np.argmax(a) for a in Y_Tr]
		Y_Te1 = [np.argmax(a) for a in Y_Te]

		print 'number of hidden units = ' + str(h)
		print 'confusion matrix ='
		print confusion_matrix(Y_Te1, Y_Te_predict)

		f.write(str(accuracy(Y_Tr1, Y_Tr_predict)) + ",")
		f.write(str(accuracy(Y_Te1, Y_Te_predict)) + ",")
		f.write(str(end - start) + "\n")

	f.close()

# part (d)
if part == 4:
	f = open('logs_part_d.csv', 'w+')

	hl_units = [5, 10, 15, 20, 25]
	learning_rate = 0.1

	for h in hl_units:
		start = time.time()

		net = create_network(85, [h,h], 10)
		trained_net = train(net, X_Tr, Y_Tr, 128, 0.1)

		end =  time.time()

		Y_Tr_predict = predict1(trained_net, X_Tr)
		Y_Te_predict = predict1(trained_net, X_Te)

		Y_Tr1 = [np.argmax(a) for a in Y_Tr]
		Y_Te1 = [np.argmax(a) for a in Y_Te]

		print 'number of hidden units = ' + str(h)
		print 'confusion matrix ='
		print confusion_matrix(Y_Te1, Y_Te_predict)

		f.write(str(accuracy(Y_Tr1, Y_Tr_predict)) + ",")
		f.write(str(accuracy(Y_Te1, Y_Te_predict)) + ",")
		f.write(str(end - start) + "\n")

	f.close()

# part (e)
if part == 5:
	hl_units = [5, 10, 15, 20, 25]
	learning_rate = 0.1

	f = open('logs_part_e1.csv', 'w+')

	for h in hl_units:
		start = time.time()

		net = create_network(85, [h], 10)
		trained_net = train(net, X_Tr, Y_Tr, 128, 0.1, 'variable', 'sigmoid')

		end =  time.time()

		Y_Tr_predict = predict1(trained_net, X_Tr)
		Y_Te_predict = predict1(trained_net, X_Te)

		Y_Tr1 = [np.argmax(a) for a in Y_Tr]
		Y_Te1 = [np.argmax(a) for a in Y_Te]

		print 'number of hidden units = ' + str(h)
		print 'confusion matrix ='
		print confusion_matrix(Y_Te1, Y_Te_predict)

		f.write(str(accuracy(Y_Tr1, Y_Tr_predict)) + ",")
		f.write(str(accuracy(Y_Te1, Y_Te_predict)) + ",")
		f.write(str(end - start) + "\n")

	f.close()

	f = open('logs_part_e2.csv', 'w+')

	for h in hl_units:
		start = time.time()

		net = create_network(85, [h,h], 10)
		trained_net = train(net, X_Tr, Y_Tr, 128, 0.1, 'variable', 'sigmoid')

		end =  time.time()

		Y_Tr_predict = predict1(trained_net, X_Tr)
		Y_Te_predict = predict1(trained_net, X_Te)

		Y_Tr1 = [np.argmax(a) for a in Y_Tr]
		Y_Te1 = [np.argmax(a) for a in Y_Te]

		print 'number of hidden units = ' + str(h)
		print 'confusion matrix ='
		print confusion_matrix(Y_Te1, Y_Te_predict)

		f.write(str(accuracy(Y_Tr1, Y_Tr_predict)) + ",")
		f.write(str(accuracy(Y_Te1, Y_Te_predict)) + ",")
		f.write(str(end - start) + "\n")

	f.close()

# part (f)
if part == 6:
	hl_units = [5, 10, 15, 20, 25]
	learning_rate = 0.1

	f = open('logs_part_f1.csv', 'w+')

	for h in hl_units:
		start = time.time()

		net = create_network(85, [h], 10)
		trained_net = train(net, X_Tr, Y_Tr, 128, 0.1, 'variable', 'relu')

		end =  time.time()

		Y_Tr_predict = predict1(trained_net, X_Tr, 'relu')
		Y_Te_predict = predict1(trained_net, X_Te, 'relu')

		Y_Tr1 = [np.argmax(a) for a in Y_Tr]
		Y_Te1 = [np.argmax(a) for a in Y_Te]

		print 'number of hidden units = ' + str(h)
		print 'confusion matrix ='
		print confusion_matrix(Y_Te1, Y_Te_predict)

		f.write(str(accuracy(Y_Tr1, Y_Tr_predict)) + ",")
		f.write(str(accuracy(Y_Te1, Y_Te_predict)) + ",")
		f.write(str(end - start) + "\n")

	f.close()

	f = open('logs_part_f2.csv', 'w+')

	for h in hl_units:
		start = time.time()

		net = create_network(85, [h,h], 10)
		trained_net = train(net, X_Tr, Y_Tr, 128, 0.1, 'variable', 'relu')

		end =  time.time()

		Y_Tr_predict = predict1(trained_net, X_Tr, 'relu')
		Y_Te_predict = predict1(trained_net, X_Te, 'relu')

		Y_Tr1 = [np.argmax(a) for a in Y_Tr]
		Y_Te1 = [np.argmax(a) for a in Y_Te]

		print 'number of hidden units = ' + str(h)
		print 'confusion matrix ='
		print confusion_matrix(Y_Te1, Y_Te_predict)

		f.write(str(accuracy(Y_Tr1, Y_Tr_predict)) + ",")
		f.write(str(accuracy(Y_Te1, Y_Te_predict)) + ",")
		f.write(str(end - start) + "\n")

	f.close()

if part == 7:
	config = sys.argv[1]
	f = open(config, 'r')
	lines = [line.rstrip('\n') for line in f]
	print lines
	hl_units = [int(a) for a in lines[4].split()]
	print hl_units

	net = create_network(int(lines[0]), hl_units, int(lines[1]))
	trained_net = train(net, X_Tr, Y_Tr, int(lines[2]), 0.1, lines[6], lines[5])

	Y_Tr_predict = predict1(trained_net, X_Tr, lines[5])
	Y_Te_predict = predict1(trained_net, X_Te, lines[5])

	Y_Tr1 = [np.argmax(a) for a in Y_Tr]
	Y_Te1 = [np.argmax(a) for a in Y_Te]

	print 'confusion matrix ='
	print confusion_matrix(Y_Te1, Y_Te_predict)

	print "Training accuracy = ", accuracy(Y_Tr1, Y_Tr_predict)
	print "Tesing accuracy = ", accuracy(Y_Te1, Y_Te_predict)

	f.close()