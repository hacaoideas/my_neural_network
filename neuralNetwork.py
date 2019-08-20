"""
>>> import numpy as np
>>> myNet = neuralNetwork(3,4,3,0.3)
>>> res = myNet.query(np.array([3,2,1]))
>>> assert(res.shape == (3, 1))


"""



import numpy as np
import scipy.special

class neuralNetwork:

    #initialize the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #learning rate
        self.lr = learningrate

        #setting up the network array
        #need to review the rules determining size of network arrays
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #other people set up weights by a normal distribution
        #centering around zero 
        #with standard deviation that is related to the number 
        #of incoming links into a node
        #this can be an improvement to make into the constructor so that it can
        #accept methods to generate initial weights

        #activation function
        self.activation_function = lambda x: scipy.special.expit(x)


    #train the neural network
    def train():
        pass

    #query the neural network
    def query(self, inputs_list):
        #convert the inputs list into 2d array
        inputs = np.array(inputs_list, ndmin=2).T 

        #calculate signal into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        #calculate the signal emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signal into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        #calculate the signals emerging from final output layer
        final_output = self.activation_function(final_inputs)

        return final_output




if __name__ == "__main__":
    import doctest
    doctest.testmod()
