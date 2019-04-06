import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import string

#################################
# Neural Netwotk ##############

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# input : a vector [[a1,a2,a3,...]]
# returns : D = [[a1(1-a1),0,0,0,..], [0,a2(1-a2),0,0,..], ...]
def convert_vector_to_D(input):
	one_minus_input = 1 - input
	product = input * one_minus_input
	identity = np.eye(len(input))
	D = identity * product
	return D

# input : parameters for a neural network
# returns : empty neural net, i.e., a list of weight matrices
def create_network(num_inputs, hidden_units_list, num_output):
	#np.random.seed(0)
	w = [0]*(len(hidden_units_list)+1)
	units_left_layer = num_inputs
	for i in range(len(hidden_units_list)) :
		units_right_layer = hidden_units_list[i]
		w[i] = np.random.uniform(-0.1, 0.1, (units_left_layer + 1, units_right_layer))	# + 1 for bias
		units_left_layer = units_right_layer
	w[len(hidden_units_list)] = np.random.uniform(-0.1, 0.1, (units_left_layer + 1, num_output))
	return w

# input : a neural_network, an input (vector) consistent with the network
# returns : list of output vectors for each layer of the network
def forward(neural_net, input):
	output_list = []
	
	i = np.reshape(input, (len(input), 1))
	layer_input = np.vstack([i, [1]]).T 		 	# transpose of input vector, 

	for w in neural_net:
		layer_output = np.matmul(layer_input, w) 				# transpose of output vector
		layer_output = sigmoid(layer_output)
		layer_input = np.vstack([layer_output.T, [1]]).T		# append 1 for bias
		
		output_list.append(layer_output.T)

	return output_list

# input : a neural_network, an input (list of vectors) consistent with the network
# returns : list of output vectors for each layer of the network
def forward_multi(neural_net, input):
	m = len(input)
	output_list = []
	
	layer_input = np.hstack([input, np.ones((m,1))]) 		 	# transpose of input vector, 
	
	for w in neural_net:
		layer_output = np.matmul(layer_input, w) 				# transpose of output vector
		layer_output = sigmoid(layer_output)
		layer_input = np.hstack([layer_output, np.ones((m,1))])		# append 1 for bias
		
		output_list.append(layer_output.T)

	return output_list

# returns : gradient for each parameter
def backward(neural_net, input, output, target_output):
	t = np.reshape(target_output, (len(target_output), 1))
	inp = np.reshape(input, (len(input), 1))

	num_layers = len(neural_net)
	gradient = [0]*num_layers
	delta = [0]*num_layers
	
	layer_output = output[num_layers-1]
	delta[num_layers-1] = layer_output * (1 - layer_output) * (layer_output - t)

	for i in range(num_layers-2, -1, -1):
		layer_output = output[i]
		delta[i] = layer_output * (1 - layer_output) * np.matmul(neural_net[i+1][:-1,:], delta[i+1])

	for i in range(num_layers-1, 0, -1):
		augmented_layer_output = np.vstack([output[i-1], [1]])
		gradient[i] = np.matmul(augmented_layer_output, delta[i].T)

	augmented_layer_output = np.vstack([inp, [1]])
	gradient[0] = np.matmul(augmented_layer_output, delta[0].T)

	return gradient

def column(array, k):
	c = array[:,k]
	return c.reshape((len(c),1))

# returns : gradient for each parameter
def backward_multi(neural_net, input, output, target_output):
	m = len(target_output)
	t = target_output.T
	inp = np.hstack([input, np.ones((input.shape[0],1))]).T

	num_layers = len(neural_net)
	gradient = [0]*num_layers
	gradient_list = []
	delta = [0]*num_layers
	
	layer_output = output[num_layers-1]
	delta[num_layers-1] = layer_output * (1 - layer_output) * (layer_output - t) * -1

	for i in range(num_layers-2, -1, -1):
		layer_output = output[i]
		delta[i] = layer_output * (1 - layer_output) * np.matmul(neural_net[i+1][:-1,:], delta[i+1]) * -1

	for k in range(m):
		for i in range(num_layers-1, 0, -1):
			augmented_layer_output = np.vstack([column(output[i-1],k), [1]])
			gradient[i] = np.matmul(augmented_layer_output, column(delta[i],k).T)

		augmented_layer_output = column(inp,k)
		gradient[0] = np.matmul(augmented_layer_output, column(delta[0],k).T)

		gradient_list.append(gradient)

	sum_gradient = []
	for j in range(len(neural_net)):
		sum_gradient.append(gradient_list[0][j].copy()/m)

	for i in range(1,m):
		for j in range(len(neural_net)):
			sum_gradient[j] += gradient_list[i][j]/m

	return sum_gradient

def getError(output, y):
	error = 0.0
	y_inp = y.T
	y_pred = output[len(output)-1].copy()

	y_err = y_inp - y_pred
	error_list = y_err * y_err
	error = np.sum(error_list)

	return error


# input : neural network to train, x,y for training
# returns : trained neural network
def train(neural_net, x, y, batch_size, neta):
	max_iterations = 500 	# max epochs
	error_threshold = 0.00001

	error_old = -1.0
	it = 0 					# epochs
	while(it < max_iterations):
		it += 1
		error_new = 0.0
		for s in range(len(y)/batch_size):
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
		print it

	return neural_net

# input : a neural_network, an input consistent with the network
# returns : multi dimensional output
def predict(neural_net, input):
	i = np.reshape(input, (len(input), 1))
	layer_input = np.vstack([i, [1]]).T 		 	# transpose of input vector, 

	for w in neural_net:
		layer_output = np.matmul(layer_input, w) 				# transpose of output vector
		layer_output = sigmoid(layer_output)
		layer_input = np.vstack([layer_output.T, [1]]).T		# append 1 for bias

	index = np.argmax(layer_output)
	layer_output = [0]*layer_output.shape[1]
	layer_output[index] = 1

	return layer_output

def predict_multi(neural_net, X):
	predicted = []
	for x1 in X:
		predicted.append(predict(neural_net, x1))

	return predicted

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


# n = create_network(2, [1], 2)
# print n
# input = [[1.0,1.0], [1.0,1.0]]
# #print np.array([[0.5], [1]])
# #o = forward(n, input)
# #print o
# #print backward(n, input, o, [1.0])
# print "****"
# a = forward_multi(n, input)
# print a

# n1 = train(n, np.array(input), np.array([[0.0, 1.0], [0.0, 1.0]]), 2, 0.1)
# print n1
# print 
# print "---------"
# print backward_multi(n, np.array(input), a, np.array([[0.0, 1.0], [0.0, 1.0]]))

# print predict(n1, input[0])
# o = forward(n, [[1,2],[1,3]])
# print np.array(o)

n = [np.array([[0.15,0.25], [0.20,0.30], [0.35,0.35]]), np.array([[0.40,0.50], [0.45,0.55], [0.60,0.60]])]
y = [[0.01,0.99]]
x = [[0.05,0.10]]
print n
o = forward_multi(n, x)
print o
b = backward_multi(n, np.array(x), o, np.array(y))
print b 
neta = 0.5
neta_gradient = [neta*g for g in b]
neural_net = [n-g for (n,g) in zip(n, neta_gradient)]
print neural_net
#################################
# I/O ###########################

train_file = "F:/ml/A3/Q2/data/poker-hand-training-true.data"
test_file = "F:/ml/A3/Q2/data/poker-hand-testing.data"

dfTr = pd.read_csv(train_file, header=None)
dfTe = pd.read_csv(test_file, header=None)

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

#################################
# running network  ##############

# net = create_network(85, [20], 10)
# trained_net = train(net, X_Tr, Y_Tr, 128, 0.1)

# Y_Tr_predict = predict_multi(trained_net, X_Tr)
# Y_Te_predict = predict_multi(trained_net, X_Te)

# print "Training accuracy = ", getAccuracy(Y_Tr_predict, Y_Tr)*100
# print "Testing accuracy = ", getAccuracy(Y_Te_predict, Y_Te)*100

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(hidden_layer_sizes=(20,), activation='logistic', solver='sgd', alpha=0.0, batch_size=128, learning_rate_init=0.1, max_iter=500, tol=0, random_state=1, nesterovs_momentum=False)
# print clf.get_params()
# clf.fit(X_Tr, Y_Tr) 
# print getAccuracy(clf.predict(X_Tr), Y_Tr)