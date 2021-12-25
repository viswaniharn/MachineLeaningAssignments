import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


featureIndices = []

def sig_derivative(z):
    return (z*(1-z))

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return (1.0 / (1.0 + np.exp(-z)))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    
    def get_data(typ):
        shuff = []
        for i in range(10):
            train_mat = mat[typ + str(i)]
            labels = np.full((train_mat.shape[0],1),i)
            labeled_train_mat = np.concatenate((train_mat, labels), axis=1)
            shuff.append(labeled_train_mat)  
            
        all_labeled = np.concatenate((shuff[0],shuff[1],shuff[2],shuff[3],shuff[4],shuff[5],shuff[6],shuff[7],shuff[8],shuff[9]), axis=0)
        
        np.random.shuffle(all_labeled)
        
        return all_labeled
    
    all_labeled_train = get_data('train')

    train_data = all_labeled_train[0:50000,][:,0:784]
    train_label = all_labeled_train[0:50000,][:,784]

    train_data = train_data/255.0

    validation_data = all_labeled_train[50000:60000,:][:,0:784] 
    validation_label = all_labeled_train[50000:60000,:][:,784]
    
    validation_data = validation_data/255.0
    
    all_labeled_test = get_data('test')
    
    test_data = all_labeled_test[:,0:784]
    test_label = all_labeled_test[:,784]
    
    test_data = test_data/255.0
    
    # Feature selection
    # Your code here.
    
    comb = np.concatenate((train_data, validation_data), axis=0)
    ref = comb[0,:]
    bool_value_columns = np.all(comb == ref, axis=0)
    
    fc = 0
    global featureIndices
    
    for i in range(len(bool_value_columns)):
        if bool_value_columns[i] == False:
            fc = fc + 1
            featureIndices.append(i)
            #print(i,end = "   ")
            
    #print("\n Number of features selected: ", fc)
            
    fin = comb[:, ~bool_value_columns]
    
    train_data = fin[0:train_data.shape[0], :]
    validation_data = fin[train_data.shape[0]:, :]
    
    test_data = test_data[:,~bool_value_columns]
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    n = training_data.shape[0]
    
    b1 = np.full((n,1),1)
    training_bias = np.concatenate((b1, training_data), axis = 1)
    
    hidden = np.dot(training_bias, np.transpose(w1))
    
    sig = sigmoid(hidden)

    hidden_n = sig.shape[0]
    
    hidden_bias = np.full((hidden_n,1),1)
    sig_biased = np.concatenate((hidden_bias, sig), axis = 1)
    
    second_op = np.dot(sig_biased, np.transpose(w2))
    output = sigmoid(second_op)
    
    y = np.full((n, n_class), 0)
    
    for i in range(n):
        y[i][training_label[i]] = 1
    
    log_output = np.log((output))
    log_o_diff = np.log((1.0-output))

    err = np.sum( np.multiply(y,log_output) + np.multiply((1.0-y),log_o_diff) )/((-1)*n)
    
    delt = output - y
    
    grad2 = np.dot(delt.T, sig_biased)
    
    temp = np.dot(delt, w2)

    temp = temp * sig_derivative(sig_biased)
    
    grad1 = (np.dot(np.transpose(temp), training_bias))[1:, :]
    
    reg_term = lambdaval * (np.sum(w1 ** 2) + np.sum(w2 ** 2)) / (2*n)
    
    obj_val = err + reg_term


    grad1_reg = (grad1 + lambdaval*w1)/n
    grad2_reg = (grad2 + lambdaval*w2)/n

    obj_grad = np.concatenate((grad1_reg.flatten(), grad2_reg.flatten()),0)   
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    n = data.shape[0]
    
    bias = np.full((n,1),1)
    data = np.concatenate((bias, data), axis=1)
    
    z = sigmoid(np.dot(data, w1.T))
    
    n_hid = z.shape[0]
    
    bias1 = np.full((n_hid,1),1)
    z = np.concatenate((bias1, z), axis=1)
    
    output_layer = sigmoid(np.dot(z, w2.T))
    
    labels = np.argmax(output_layer, axis=1)
    
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 30

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

parameters = {"selected_features": featureIndices, "n_hidden": 20, "w1": w1, "w2": w2, "lambda": 30}
pickle.dump(parameters, open('params.pickle', 'wb'))