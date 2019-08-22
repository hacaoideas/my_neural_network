import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import loader


class examiner:
    """This class object examines the data, and make decision to feed the data
    into the neuralNetwork"""

    def __init__(self, ld):
        self.data_only = ld.data_only
        self.label_only = ld.label_only
                
    
    
    def plotone(self, row):
        #This load the first line and plot it for testing purpose only
        fig = plt.figure()
        plt.imshow(self.data_only[row,:].reshape(28,28), cmap='Greys', interpolation=None)
        fig.suptitle(self.label_only[row])
        plt.savefig('../temp/sample_plot.png')


    def getone(self, row):
        #This one return a pair of input and target for the neuralNetwork.train()
        #Numpy array is immutable, it's pointless trying to shrink or edit it. 
        #looping through getting the items is the most efficient way
        target = np.zeros((10))
        target[self.label_only[row]] = 1
        input = self.data_only[row,:]
        input_i = input.flatten()
        input_i = np.interp(input, (input.min(), input.max()),(0,1))
        return (input_i, target)






if __name__ == "__main__":
    ex = examiner(loader.loader('../data/mnist_train_100.csv'))
    ex.plotone(5)
    l,d = ex.getone(5)
    print(l)
    print(d)
    
    