"""

>>> import numpy as np
>>> myNet = neuralNetwork(3,4,3,0.3)
>>> res = myNet.query(np.array([3,2,1]))
>>> assert(res.shape == (3, 1))


"""



import numpy as np
import scipy.special
import pickle as pk

def _expit(x):
    return scipy.special.expit(x)

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
        #only object defined as def can be pickled. 
        self.activation_function = _expit


    #train the neural network
    def train(self, inputs_list, targets_list):
        #conver inputs list to 2d array
        #ndmin prevents the newly created array
        #to take shape like (10,1)
        #instead, the array must be shaped (1,10) and transposed to (10,1)
        inputs = np.array(inputs_list, ndmin=2).T 
        targets = np.array(targets_list, ndmin=2).T 

        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signal into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        #calculate signal emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #error is the (target - actual)
        output_errors = targets - final_outputs

        #hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        #update the weights for links between hidden and output layers
        self.who += self.lr*np.dot((output_errors*final_outputs*(1.0 - final_outputs)),np.transpose(hidden_outputs))

        #update the weights for linked between hidden and input layers
        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), np.transpose(inputs))


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

    #This function persists the neural Network innards into file
    def persist(self):
        with open('../temp/neuralNetwork.pkl', 'wb') as dumpfile:
            pk.dump(self, dumpfile)

    #This function recreates the neural network from file
    @classmethod
    def recreate(cls):
        with open('../temp/neuralNetwork.pkl', 'rb') as dumpfile:
            obj = pk.load(dumpfile)
            return obj


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    nw1 = neuralNetwork(3,4,3,0.3)
    print(nw1.__dict__)
    nw1.persist()
    print("====")
    nw2 = neuralNetwork.recreate()
    print(nw2.__dict__)
