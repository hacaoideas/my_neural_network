import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import loader


class examiner:
    """This class object loads and examines the data"""

    #This will create the examiner and
    #loads the data and
    # def __init__(self, path_data_set):
    #     self.raw_data = pd.read_csv(path_data_set, sep=',', header=None)
    #     self.label_only = np.atleast_2d(self.raw_data)[:,0]
    #     self.data_only = np.atleast_2d(self.raw_data)[:,1:]
        
        
    #     self.data = np.array(self.raw_data)

    def __init__(self, ld):
        self.data_only = ld.data_only
        self.label_only = ld.label_only
                


    
    #This load the first line and plot it
    def plotone(self, row):
        plt.imshow(self.data_only[row,:].reshape(28,28), cmap='Greys', interpolation=None)
        plt.savefig('../temp/sample_plot.png')




if __name__ == "__main__":
    ld = loader.loader('../data/mnist_train_100.csv')
    ex = examiner(ld)
    
    ex.plotone(6)
    print(ex.label_only[6])
    