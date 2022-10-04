__author__ = 'eddy_huang'
from three_layer_neural_network import *

def generate_data_choice():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200, noise=0.20)
    return X, y

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, nn_input_dim, nn_layer_dim, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        Constructs a deep neural network
        :param nn_input_dim: input dimension
        :param nn_layer_dim: a list describe the hidden layers dimension
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param actF: abstract activation function
        :param diff_actF: abstract function of derivative of activation
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''

        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.nn_layer_dim = nn_layer_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.nn_layer_num = len(nn_layer_dim)
        # print(self.nn_layer_num)
        self.actF= lambda x: self.actFun(x, type=self.actFun_type)
        self.diff_actF = lambda x: self.diff_actFun(x, type=self.actFun_type)

        # Initialize the network
        np.random.seed(seed)

        # initializes a N-dimension array to store hidden layers
        self.singleLayers = []
        input_dim = nn_input_dim
        for i in nn_layer_dim:
            self.singleLayers.append(Layer(i, input_dim, self.actFun_type))
            input_dim = i

        # Initialize the weights and biases in the output layer
        self.WOut = np.random.randn(self.nn_layer_dim[-1], self.nn_output_dim) / np.sqrt(self.nn_layer_dim[-1])
        self.bOut = np.zeros((1, self.nn_output_dim))

    def feedforward(self, X, actFun):
        '''
        feedforward builds a n-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        x = X
        for layer in self.singleLayers:
            layer.feedforward(x, actFun)
            x = layer.a

        # output layer
        zOut = np.dot(x, self.WOut) + self.bOut
        exp_scores = np.exp(zOut)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return None

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2, ... dL/dn, dL/bn in two lists
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        deltaLast = self.probs
        deltaLast[range(num_examples), y] -= 1
        # last hidden layer output
        a = self.singleLayers[-1].a
        z = self.singleLayers[-1].z

        # output layer
        dWOut = np.dot(a.T, deltaLast)
        dbOut = np.sum(deltaLast, axis=0)
        deltaOut = self.diff_actF(z) * (np.dot(deltaLast, self.WOut.T))
        dW = [dWOut]
        db = [dbOut]

        # backprop the hidden layer(s)
        delta = deltaOut
        for l in range(self.nn_layer_num - 1, -1, -1) :
            if l != 0:
                a = self.singleLayers[l-1].a
                z = self.singleLayers[l-1].z
            else:
                a = X
                z = X
            layer_dW, layer_db = self.singleLayers[l].backprop(z, a, delta, self.diff_actF)
            delta = self.singleLayers[l].deltaPrev
            dW.insert(0, layer_dW)
            db.insert(0, layer_db)

        return dW, db

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, self.actF)
        # Calculating the loss

        data_loss = -np.sum(np.log(self.probs[range(num_examples), y]))
        # Add regulatization term to loss (optional)
        # Mean square (L2)
        WSE = np.sum(np.square(self.WOut))
        for l in self.singleLayers:
            WSE += np.sum(np.square(l.W))
        data_loss += self.reg_lambda / 2 * WSE
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, self.actF)

            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW[-1] += self.reg_lambda * self.WOut
            for l in range(0, self.nn_layer_num):
                layer = self.singleLayers[l]
                dW[l] += self.reg_lambda * layer.W

            # Update
            mod = 1.0 / self.nn_layer_num
            # output layer
            self.WOut += -epsilon * dW[-1] * mod
            self.bOut += -epsilon * db[-1] * mod
            # hidden layer(s)
            for l in range(0, self.nn_layer_num):
                layer = self.singleLayers[l]
                layer.W += -epsilon * dW[l] * mod
                layer.b += -epsilon * db[l] * mod

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))
        return None

class Layer(object):
    '''
    This is one single layer base class
    '''
    def __init__(self, layer_output_dim, layer_input_dim, actFun_type='tanh'):
        """
        :param layer_input_dim: input dimension
        :param layer_output_dim: the number of hidden units
        :param actFun_type: type of activation function to be used
        """

        self.layer_input_dim = layer_input_dim
        self.layer_output_dim = layer_output_dim
        self.actFun_type = actFun_type

        # Initialize weight and bias
        self.W = np.random.randn(self.layer_input_dim, self.layer_output_dim) / np.sqrt(self.layer_input_dim)
        self.b = np.zeros((1, self.layer_output_dim))

    def feedforward(self, X, actFun):
        """
        :param X: activation from previous layer
        :param actFun: the activation function passed as an anonymous function
        :return: None
        """
        self.z = np.dot(X, self.W) + self.b
        self.a = actFun(self.z)
        return None

    def backprop(self, z, a, delta, diff_actFun):
        """
        :param a: the activation from the previous layer
        :param z: the z from the previous layer
        :deltaPrev: the delta from the previous layer
        :param diff_actFun: the differentiated activation function as an anonymous function
        :return: gradient of weight and bias for the layer
        """
        self.deltaPrev = diff_actFun(z) * (np.dot(delta, self.W.T))
        self.db = np.sum(delta, axis=0, keepdims=True)
        self.dW = np.dot(a.T, delta)
        return self.dW, self.db

if __name__ == "__main__":
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    model = DeepNeuralNetwork(nn_input_dim=2, nn_layer_dim=[5, 5, 5, 5, 5], nn_output_dim=2, actFun_type='relu')

    # generate and visualize choice of dataset
    # X, y = generate_data_choice()
    # model = DeepNeuralNetwork(nn_input_dim=2, nn_layer_dim=[50, 50, 50, 50, 50], nn_output_dim=2, actFun_type='relu')

    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)
