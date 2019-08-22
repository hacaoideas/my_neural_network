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
                
    
    #This load the first line and plot it for testing purpose only
    def plotone(self, row):
        plt.imshow(self.data_only[row,:].reshape(28,28), cmap='Greys', interpolation=None)
        plt.savefig('../temp/sample_plot.png')




if __name__ == "__main__":
    ex = examiner(loader.loader('../data/mnist_train_100.csv'))

    ex.plotone(7)
    print(ex.label_only[7])
    