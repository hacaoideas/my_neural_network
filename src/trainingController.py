import neuralNetwork
import examiner
import loader

class trainingController:
    def __init__(self):
        #initialize the controller, creating the loader, examine data and create the network
        self.ld = loader.loader('../data/mnist_train_100.csv')
        self.ex = examiner.examiner(ld)
        self.nw = neuralNetwork.neuralNetwork(784,100,10,0.3)

    def train(self):
        TRAIN_SET_SIZE = self.ex.label_only.shape[0]
        for i in range(TRAIN_SET_SIZE):
            print(i)
            self.nw.train(*self.ex.getone(i)) #explicitly tell python to unpack tuple
            if i % 20 == 0:
                self.nw.persist()



if __name__ == "__main__":
    pass
    

    