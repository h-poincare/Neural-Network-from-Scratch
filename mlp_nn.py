import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time 


def yLabels(data_labels, num_labels):
    y_vector = np.zeros((len(num_labels), data_labels.shape[0]))
    for i,dl in enumerate(data_labels):
        for j,nl in enumerate(num_labels):
            if (dl == nl):
                y_vector[j, i] = 1
            else:
                y_vector[j, i] = 0
    return y_vector


class NeuralNetwork():
    def __init__(self, n_inputs, n_hidden_units, n_outputs):
        self.n_inputs = n_inputs
        self. n_hidden_units = n_hidden_units
        self.n_outputs = n_outputs

        self.net = self.create_net()
        self.weights= self.init_weights(self.net)
        

    #function returns net structure 
    def create_net(self):
        net = []
        first_layer = np.zeros((self.n_inputs, 1))
        output_layer = np.zeros((self.n_outputs, 1))
        net.append(first_layer)
        for _,hidden_unit in enumerate(self.n_hidden_units):
            hidden_layer = np.zeros((hidden_unit, 1))
            net.append(hidden_layer)
        net.append(output_layer)

        return net

    def init_weights(self, net):
        num_thetas = len(net) - 1
        weights = []
        for i in range(num_thetas):
            theta = np.random.randn(len(net[i + 1]), len(net[i]) + 1)
            weights.append(theta)
        return weights
    
    def sigmoid(self, z, dx=False):
        if dx:
            if type(z) == np.matrixlib.defmatrix.matrix:
                return np.multiply(z, (1 - z))
            else:
                return (z) * (1 - z)
        else:
            return 1 / (1 + np.exp(-z))
    
    def feedforward(self, x, j):
        one = np.array([[1]])
        self.net[0] = x[[j]].T
        self.net[0] = np.append(one, self.net[0], axis=0)
        for i,theta in enumerate(self.weights):
            z = np.dot(theta, self.net[i])
            a = self.sigmoid(z)
            self.net[i + 1] = a
            if i + 1 < len(self.net)-1:
                self.net[i + 1] = np.append(one, self.net[i + 1], axis=0)

        output = self.net[len(self.net)-1]

        return output
        
    def backprop(self, y, output, j):
        delta_list = []
        self.weightsGrad = [np.zeros((self.weights[t].shape)) for t in range(len(self.weights))]
        delta = output - y[[j]].T
        delta_list.append(delta)
        for k in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(self.weights[k].T, delta) * self.sigmoid(self.net[k], dx=True)
            delta = delta[1:len(delta)]
            delta_list.append(delta)
        # compute the gradient of the cost function
        delta_list = list(reversed(delta_list))
        for m in range(len(self.weightsGrad)):
            self.weightsGrad[m] = np.dot(delta_list[m], self.net[m].T)

        return self.weightsGrad

    def gradientDescent(self, l_rate):
        for i in range(len(self.weights)):
            self.weights[i] += -(l_rate * self.weightsGrad[i])
        return self.weights
    
    def accuracy(self, y, output, j):
        if np.argmax(y[[j]].T) == np.argmax(output.T, axis=-1):
            return 1
        else:
            return 0

    def train(self, x_train, y_train, x_test, y_test, epochs=20 
              , batch_size=500, l_rate=0.01, viz=False):
        
        self.epochs = epochs 
        self.batch_size = batch_size
        num_batches = x_train.shape[0] // self.batch_size
        epoch_list = []
        error_hist = []
        x_vals = 0
        for i in range(self.epochs):

            epoch_list.append(x_vals)
            #shuffle 
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]

            for j in range(num_batches):
                # batch
                begin = j * self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                #Forward prop
                output = self.feedforward(x, j)
                # Backprop
                _ = self.backprop(y, output, j)
                # Optimize
                self.gradientDescent(l_rate=l_rate)

            #Make predictions.. evaluate model performance
            #Train 
            train_pred = 0
            for ind,x in enumerate(x_train):
                output = self.feedforward(x_train, ind)
                train_acc_count = self.accuracy(y_train, output, ind)
                train_pred += train_acc_count
                
            train_acc = train_pred / x_train.shape[0]
            error_hist.append(1-train_acc)
            error_hist[i] = 1-train_acc

            if i % 100 == 0: 
                print(f"TRAINING: EPOCH {i}  :   TRAIN ACCURACY = {100*(train_acc)}")
            

            if viz == True:
                plt.plot(epoch_list, error_hist, color='blue')
                plt.ylim(0, 1)
                plt.pause(0.000000001)
                plt.title(f"Neural Network: Error Over {self.epochs} Epochs")    
                plt.xlabel("Epochs")
            x_vals +=1
        plt.show()


        #Test performance
        test_pred = 0
        for ind,x in enumerate(x_test):
            output = self.feedforward(x_test, ind)
            test_acc_count = self.accuracy(y_test, output, ind)
            test_pred += test_acc_count

        test_acc = test_pred / x_test.shape[0]
        print(f"TEST ACCURACY = {100*(test_acc)}")


if __name__ == '__main__':
    Xdata = np.load("data/digit_images.npy")
    ydata = np.load("data/digit_labels.npy")

    ylabels = yLabels(ydata, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = ylabels.T
    Xtrain, Xtest, ytrain, ytest = train_test_split(Xdata, y, test_size=0.33, random_state=42)
    nn1 = NeuralNetwork(64, [25, 25], 10)
    nn1.train(Xtrain, ytrain, Xtest, ytest, 15_000, Xtrain.shape[0], 0.01, viz=False)

   