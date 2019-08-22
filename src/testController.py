import neuralNetwork
import examiner
import loader
import numpy as np
import matplotlib.pyplot as plt 

class testController:
    def __init__(self):
        #initializing the testController, load test data and examine, recreate the network from file
        self.ld = loader.loader('../data/mnist_test_10.csv')
        self.ex = examiner.examiner(self.ld)
        self.nw = neuralNetwork.neuralNetwork.recreate()

    def test(self, row):
        test_data, test_target = self.ex.getone(row)
        
        #expected value
        exp_value = np.argmax(test_target)

        #result value
        res = self.nw.query(test_data)
        fig = plt.figure()
        plt.imshow(test_data.reshape(28,28), cmap='Greys', interpolation=None)
        res_value = np.argmax(res)
        fig.suptitle(f"expected {exp_value}, get {res_value}")
        plt.savefig('../temp/result.png')
        plt.show()
        


if __name__ == "__main__":
    
    tc = testController()
    tc.test(3)