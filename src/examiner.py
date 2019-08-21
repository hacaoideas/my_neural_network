import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


class examiner:
    """This class object loads and examines the data"""

    #This will create the examiner and
    #loads the data and
    def __init__(self, path_data_set):
        self.raw_data = pd.read_csv(path_data_set, sep=',', header=None)
        self.data = np.array(self.raw_data)

                


    
    #This load the first line and plot it
    def plotone(self):
        image_array = self.data[0,1:].reshape(28,28)
        plt.imshow(image_array, cmap='Greys', interpolation=None)
        plt.show()


        

    
    #These implements iterator protocol and return the
    #next pair of inputs and targets
    def __iter__(self):
        self._current = 0
        return self

    def next(self):
        if self._current >= self.data.shape[0]:
            raise StopIteration
        else:
            self._current += 1
            return self.data[self._current-1,1:]


if __name__ == "__main__":
    ex = examiner('../data/mnist_test_10.csv')
    ex.plotone()
    for i in ex:
        print(i)
    