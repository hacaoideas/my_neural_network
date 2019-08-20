import numpy as np

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
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hjnodes))

        #other people set up weights by a normal distribution
        #centering around zero 
        #with standard deviation that is related to the number 
        #of incoming links into a node
        #this can be an improvement to make into the constructor so that it can
        #accept methods to generate initial weights
        

    #train the neural network
    def train():
        pass

    #query the neural network
    def query():
        pass