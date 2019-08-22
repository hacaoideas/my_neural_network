import pandas as pd 
import numpy as np 

class loader:
    """This class does all the loading
    and is responsible for handling the exception"""
    def __init__(self, data_path):
        self.raw = np.array(pd.read_csv(data_path, sep=',', header=None))
        self.label_only = self.raw[:,0]
        self.data_only = self.raw[:,1:]
